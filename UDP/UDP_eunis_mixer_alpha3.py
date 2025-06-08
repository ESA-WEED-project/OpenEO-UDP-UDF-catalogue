"""
UDP to run the inference module on openEO with given DataCube and ML model
This version for the alpha2 release. There are a few limitation
1) The text_concat on the filename prefix is not seriazable somehow in the save_results node, so needs to be adapted manually in the json.
        add module in json
        "textconcat4": {
      "process_id": "text_concat",
      "arguments": {
        "data": [
          "Alpha2_EUNIS-habitat-proba-cube_year",
          {"from_parameter": "year"},
          "_",
          {"from_parameter": "area_name"},
          "_",
          {"from_parameter": "onnx_model_id"},
          "_",
          {"from_parameter": "scenarioId"}]
      }
    },
and replace file_name with {"from_node": "textconcat4"} in saveresult1 and as well
            "product_tile": {"from_parameter": "area_name"},
            "time_start": {"from_node": "textconcat1"},
            "time_end": {"from_node": "textconcat2"}


"""
import os
import sys
import pathlib
import openeo
import json
import requests
import re
import pandas as pd
import datetime
from datetime import datetime, timedelta
import traceback
# WEED project developments"
from openeo.metadata import metadata_from_stac
from openeo.api.process import Parameter
from openeo.processes import text_concat
from openeo.rest.udp import build_process_dict
# TODO remove direct path
sys.path.append('/home/smetsb/PycharmProjects/eo_processing/src')
from eo_processing.config import get_job_options, get_collection_options, get_standard_processing_options
from eo_processing.utils.helper import init_connection, getUDFpath
from eo_processing.utils.metadata import get_base_metadata
from eo_processing.utils.geoprocessing import reproj_bbox_to_ll

# Establish connection to OpenEO instance (note that authentication is not necessary to just build the UDP)
provider = 'cdse' #this udp works only on cdse
connection = openeo.connect(url="openeo.dataspace.copernicus.eu")
# set up the UDP parameters
param_bbox = Parameter(name="bbox",
    description="""Limits the data to process to the specified bounding box or polygons.\n\nFor raster data, the process loads the pixel into the data cube if the point\n
    at the pixel center intersects with the bounding box or any of the polygons\n(as defined in the Simple Features standard by the OGC).\n\nFor vector data, 
    the process loads the geometry into the data cube if the geometry\nis fully within the bounding box or any of the polygons (as defined in the\nSimple 
    Features standard by the OGC). Empty geometries may only be in the\ndata cube if no spatial extent has been provided.\n\nEmpty geometries are ignored.\n\nSet this parameter to null to set no limit for the spatial extent.""",
    schema= {"title": "Bounding Box",
          "type": "object",
          "subtype": "bounding-box",
          "required": ["west", "south","east","north"],
          "properties": {
            "west": {
              "description": "West (lower left corner, coordinate axis 1).",
              "type": "number"
            },
            "south": {
              "description": "South (lower left corner, coordinate axis 2).",
              "type": "number"
            },
            "east": {
              "description": "East (upper right corner, coordinate axis 1).",
              "type": "number"
            },
            "north": {
              "description": "North (upper right corner, coordinate axis 2).",
              "type": "number"
            },
            "crs": {
              "description": "Coordinate reference system of the extent, specified as as [EPSG code](http://www.epsg-registry.org/) or [WKT2 CRS string](http://docs.opengeospatial.org/is/18-010r7/18-010r7.html).",
              "anyOf": [
                {
                  "title": "EPSG Code",
                  "type": "integer",
                  "subtype": "epsg-code",
                  "minimum": 1000,
                  "examples": [
                    3035
                  ]
                },
                {
                  "title": "WKT2",
                  "type": "string",
                  "subtype": "wkt2-definition"
                }
              ],
              "default": 3035
            }
          }
        },)

param_year = Parameter.integer(
    name="year",
    default=2024,
    description="The year for which to generate the habitat map. (default: 2024)",
)

param_digitalId = Parameter.string(
    name="digitalId",
    description="Digital ID of client",
)
param_scenarioId = Parameter.string(
    name="scenarioId",
    description="Id of the scenario/session",
)#combine with dash iso /

param_name = Parameter.string(
    name="area_name",
    description="Name of the AOI",
    optional = True,
    default = "AOI"
)

def query_stac(spatial_extent, temporal_extent, collections=["inference-alpha2-prepared-v101"]) -> bool|None:

    # search endpoint of the WEED STAC API
    search_endpoint = "https://catalogue.weed.apex.esa.int/search"
    # create the search string
    search_payload = {
        "limit": 20,
        "collections": collections,
        "filter-lang": "cql-json",
        "bbox": spatial_extent,
        #"datetime": f"{temporal_extent[0]}T00:00:00Z/{temporal_extent[1]}T23:59:59Z",
    }
    # execute the search
    try:
        r = requests.post(search_endpoint, json=search_payload, timeout=(3, 5))
    except requests.exceptions.Timeout:
        raise RuntimeError("Timeout while searching for feature items")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error while searching: {e}")

    # handle response - here we have the rule that cubes should be UNIQUE
    if r.status_code == 200:
        search_results = r.json()
        if len(search_results["features"]) >= 1:
            #print(search_results["features"])
            return search_results
        elif len(search_results["features"]) == 0:
            return None
    elif r.status_code == 400:
        print("Bad Request  validation errors:")
        raise RuntimeError(json.dumps(r.json(), indent=2))
    else:
        print(f"Unexpected status {r.status_code}:")
        raise RuntimeError(r.text)

def parse_prob_classes_fromStac(metadata):

    #print("Parsing prob classes from Stac")
    filename = metadata['features'][0]['id']  # 'alpha-2_proba-cube_year2024_34σEH13_EUNIS2021plus-EU-v1-2024-MED_v101.tif'

    filename_match = re.search((r"year(?P<year>\d{4})_"r"(?P<tileID>[^_]+).*"r"_v(?P<version>\d+)\.tif$"), filename)
    if filename_match:
        version = filename_match.group("version")
        year = filename_match.group("year")
        tile = filename_match.group("tileID")
    else:
        raise ValueError("Filename does not match expected pattern.")

    #now retrieve the bandnames to extract class information
    band_names=[]
    for i in range(len(metadata['features'][0]['assets'][filename]['eo:bands'])):
        band_names.append(metadata['features'][0]['assets'][filename]['eo:bands'][i]['name'])
    # Parse band names into structured format
    band_info = []
    pattern = re.compile(r"Level([\w\d]+)_class-([\w\d]+)_habitat-([\w\d]+)-(\d+)")
    for band_nr, band_name in enumerate(band_names, start=1):
        match = pattern.search(band_name.replace(" ",""))  #make sure no white spaces pending
        if match:
            level, class_name, habitat, raster_code = match.groups()
            band_info.append((band_nr, level, class_name, habitat, int(raster_code)))
        else:
            print('skipping {}'.format(band_name))
    # Create DataFrame
    df = pd.DataFrame(band_info, columns=["band_nr", "level", "model", "habitat", "raster_code"])

    return df, tile

###################################
# set the request year for the data
start = text_concat([param_year, "01", "01T00:00:00Z" ], separator="-")
end = text_concat([param_year, "12", "31T23:59:59Z"], separator="-")


file_namea = text_concat(["Alpha3_EUNIS-extent-map_year",param_year ],separator="")
file_name = text_concat([file_namea,param_name,param_scenarioId], separator="_")
s3_prefix = text_concat([param_digitalId,param_scenarioId], separator="-")

# convert the row name into a openEO bbox dict giving the spatial extent of the job
processing_extent = param_bbox
# define job_options, processing_options,  and collection_options
job_options = get_job_options(provider=provider, task='eunis_mixer')
collection_options = get_collection_options(provider=provider)
processing_options = get_standard_processing_options(provider=provider, task='feature_generation')

# updates to job_options
job_options.update({"allow_empty_cubes": True})  # that cubes in areas with no data are created for a STAC and not an error
job_options.update({"export-workspace-enable-merge": True})  # that items for STAC catalog are merged if in same S3 prefix
job_options.update({'etl_organization_id': '4938'})  # billing to specific organization

#### increasing memory for alpha-2 for testing
job_options.update({"driver-memory": "4G",
                    "driver-memoryOverhead": "4G",
                    "executor-memory": "4G",
                    "executor-memoryOverhead": "2500m",
                    "python-memory": "5G",
                    "logging-threshold": "debug"})

# first check if there is an item that matches the criteria
# bbox is array of numbers in sequesce lon_min, lat_min, lon_max, lat_max in WGS84
extent_ll = reproj_bbox_to_ll(processing_extent).bounds
s_extent = [extent_ll[0], extent_ll[1], extent_ll[2], extent_ll[3]]
s_extent_dict = {"west": extent_ll[0], "south": extent_ll[1], "east": extent_ll[2], "north": extent_ll[3],
                 "crs": "EPSG:4326"}
t_extent = (start, (datetime.strptime(end, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d'))

# s_extent= {"west":21.229899387831, "south":38.48744551412938, "east": 21.458646069348923, "north":38.668357705483636, "crs": "EPSG:4326"}
collections = ["inference-alpha2-prepared-v101"]
result_query = query_stac(s_extent, t_extent, collections)
if result_query is None:
    raise RuntimeError("No STAC items found that match the criteria in WEED catalog {collections}")

# use returned metadata to build up the class dictionary
df, tileID = parse_prob_classes_fromStac(result_query)

print('Load datacube from stac for tile {}'.format(tileID))
# create a connection to backend
#connection = init_connection(backend)
data_cube = connection.load_stac("https://catalogue.weed.apex.esa.int/collections/inference-alpha2-prepared-v101",
                                 spatial_extent=s_extent_dict)
                                # temporal_extent TODO add once openeo has fixed issue
# keep temporal dimension to store in STAC
# we do not yet resample as we follow the inference spatial resolution
# filter spatial the whole cube
data_cube = data_cube.filter_bbox(processing_extent)

### run hierarchical probability merger
udf = openeo.UDF.from_file(
    getUDFpath('udf_max_occurence_hierarchical_merger.py'),
    context={"tile": tileID, "level_info": df.to_dict()})

# Apply the UDF to the data cube
data_cube = data_cube.apply_dimension(process=udf, dimension="bands")

### create job progress graph including storage to S3
# prepare metadata for rater output
file_meta = get_base_metadata(project='WEED')
file_meta.update(description=f'EUNIS habitat map level3 (highest probability of occurrence).',
                 tiling_grid='Global_20km',
                 product_tile=f'{tileID}',
                 raster_coding=str(df.set_index('habitat')['raster_code'].to_dict()),
                 time_start=start,
                 time_end=end)
# ToDo: prepare also needed band metadata

saved_cube = data_cube.save_result(format="GTiff",
                                   options={
                                       'separate_asset_per_band': False,
                                       'filename_prefix': file_name,
                                       'file_metadata': file_meta
                                   })

# save to S3 and directly publish results in STAC catalog on WEED STAC.api if export workspace is configured
cube_workspace = saved_cube.export_workspace(workspace="esa-weed-apex-stac-api-workspace",
                                    merge=s3_prefix)

description = """
EUNIS hierarchical mixer for the habitat maps for the alpha3 release.
"""
spec = build_process_dict(
    process_id="udp_eunis_mixer_alpha3",
    summary="Generates the alpha 3 eunix extent map result based on highest probabilities.",
    description=description.strip(),
    parameters=[
        param_bbox,
        param_year,
        param_digitalId,
        param_scenarioId,
        param_name
    ],
    process_graph=cube_workspace,
    default_job_options=job_options
)

# dump to json file to be usable as UDP
this_script = pathlib.Path(__file__)
json_file = os.path.normpath(os.path.join(this_script.parent, 'json', this_script.name.lower().split('.')[0] + '.json'))
print(f"Writing UDP to {json_file}")
with open(json_file, "w", encoding="UTF8") as f:
    json.dump(spec, f, indent=2)
