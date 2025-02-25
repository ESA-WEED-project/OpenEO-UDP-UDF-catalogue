"""
UDP to run the inference module on openEO with given DataCube and ML model
This version for the alpha1 release. There are a few limitation
1) processing options should be added manually in the outcome json

  "default_job_options": {
    "driver-memory": "1000m",
    "driver-memoryOverhead": "1000m",
    "executor-memory": "1500m",
    "executor-memoryOverhead": "1500m",
    "python-memory": "4000m",
    "max-executors": 20,
    "soft-errors": "true",
    "udf-dependency-archives": [
      "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_dependencies_1.16.3.zip#onnx_deps"
    ]}

2) it seems that the param object is not yet suitable to rename band name on. So for the moment not set should be retrieved from models later on.
3) The text_concat on the filename prefix is not seriazable somehow, so needs to be adapted manually in the json.
        add module in json
    "textconcat4": {
      "process_id": "text_concat",
      "arguments": {
        "data": [
          "Alpha1_EUNIS-habitat-proba-cube_year",
          {"from_parameter": "year"},
          "_",
          {"from_parameter": "area_name"}]
      }
    },
and replace file_name with textconcat4


"""
import json
import openeo
from openeo.api.process import Parameter
from openeo.processes import text_concat, add
from openeo.rest.udp import build_process_dict
from openeo.rest.datacube import THIS
from eo_processing.config.settings import get_collection_options, get_standard_processing_options, get_job_options
from eo_processing.openeo.processing import generate_master_feature_cube
from eo_processing.utils.helper import getUDFpath
import os
import pathlib

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
    default=2021,
    description="The year for which to generate the habitat map. (default: 2021)",
)
param_onnx_models = Parameter(
    name="onnx_models",
    description="List of onnx_models used to run.",
    schema={"type": "list"},
)
param_output_band_names = Parameter.array(
    name="output_band_names",
    description="Array of str handing over all needed ecosystem characteristics layers to run model training."
)

param_epsg = Parameter.integer(
    name="target_epsg",
    default=3035,
    description="The desired output projection system, which is EPSG:3035 by default.",
)
param_resolution = Parameter.number(
    name="target_res",
    default=10.0,
    description="The desired resolution, specified in units of the projection system, which is meters by default.",
)
param_digitalId = Parameter.string(
    name="digitalId",
    description="Digital ID of client",
)
param_scenarioId = Parameter.string(
    name="scenarioId",
    description="Id of the scenario/session",
)
param_name = Parameter.string(
    name="area_name",
    description="Name of the AOI",
)

# set the request year for the data
start = text_concat([param_year, "01", "01"], separator="-")
end = text_concat([add(param_year, 1), "01", "01"], separator="-")

file_name = text_concat(["Alpha1_EUNIS-habitat-proba-cube_year",param_year,"_",param_name],separator="")
model_urls = param_onnx_models
output_band_names = param_output_band_names
s3_prefix = text_concat([param_digitalId,param_scenarioId], separator="/")

# convert the row name into a openEO bbox dict giving the spatial extent of the job
processing_extent = param_bbox
# define job_options, processing_options,  and collection_options
job_options = get_job_options(provider=provider, task='inference')
collection_options = get_collection_options(provider=provider)
processing_options = get_standard_processing_options(provider=provider, task='feature_generation')
# adapt the epsg to the processing grid
processing_options.update(target_crs = param_epsg)

#### create the feature cube
# define the S1/S2 processed feature cube (Note: do not set spatial extent since we hand it over in the end)
data_cube = generate_master_feature_cube(connection,
                                         None,
                                         start,
                                         end,
                                         **collection_options,
                                         **processing_options)

# now we merge in the NON ON-DEMAND processed features (DEM and WENR features)
# load the DEM from a CDSE collection
DEM = connection.load_collection(
    "COPERNICUS_30",
    bands=["DEM"])
# reduce the temporal domain since copernicus_30 collection is "special" and feature only are one time stamp
DEM = DEM.reduce_dimension(dimension='t', reducer=lambda x: x.last(ignore_nodata=True))
# resample the cube to 10m and EPSG of corresponding 20x20km grid tile
DEM = DEM.resample_spatial(projection=processing_options['target_crs'],
                           resolution=processing_options['resolution'],
                           method="bilinear")
# merge into the S1/S2 data cube
data_cube = data_cube.merge_cubes(DEM)

# load the WERN features from public STAC
WENR = connection.load_stac("https://stac.openeo.vito.be/collections/wenr_features")
# drop the time dimension

WENR.metadata=None
# resample the cube to 10m and EPSG of corresponding 20x20km grid tile
WENR = WENR.resample_spatial(projection=processing_options['target_crs'],
                             resolution=(processing_options['resolution']),
                             method="near")

try:
    WENR = WENR.drop_dimension('t')
except:
    # workaround if we still have the client issues with the time dimensions for STAC dataset with only one time stamp
    WENR.metadata = WENR.metadata.add_dimension("t", label=None, type="temporal")
    WENR = WENR.drop_dimension('t')

# merge into the S1/S2 data cube
data_cube = data_cube.merge_cubes(WENR)

# filter spatial the whole cube
data_cube = data_cube.filter_bbox(processing_extent)

#### run multi-model inference
#we pass the model url as context information within the UDF
udf  = openeo.UDF.from_file(
        getUDFpath('udf_catboost_inference.py'),
        context={"model_list": model_urls})

# Apply the UDF to the data cube.
proba_cube = data_cube.apply_dimension(process=udf, dimension = "bands")

# extra posprocessing (band label renaming and scaling)
#proba_cube = proba_cube.rename_labels(dimension="bands",target=param_output_band_names) This does not seem to work
proba_cube = proba_cube.linear_scale_range(0,100, 0,100)

#### create job progress graph including storage to S3
saved_cube = proba_cube.save_result(format="GTiff",
                                    options={
                                        'separate_asset_per_band': True,
                                        'filename_prefix': "file_name",
                                    })

#generate S3_prefix

cube_workspace = saved_cube.process("export_workspace",
                                        arguments={
                                            'data': THIS,
                                            'workspace': 'esa-weed-workspace',
                                            'merge': s3_prefix
                                        })


description = """
Inference for the habitat maps for the alpha1 release.
"""
spec = build_process_dict(
    process_id="udp_inference_module_alpha1",
    summary="Generates the alpha 1 inference result based on inputs."
            "Returns a single band per probability.",
    description=description.strip(),
    parameters=[
        param_bbox,
        param_year,
        param_onnx_models,
        param_output_band_names,
        param_digitalId,
        param_scenarioId,
        param_name,
        param_epsg,
        param_resolution,
    ],

    process_graph=cube_workspace,
)

# dump to json file to be usable as UDP
this_script = pathlib.Path(__file__)
json_file = os.path.normpath(os.path.join(this_script.parent, 'json', this_script.name.lower().split('.')[0] + '.json'))
print(f"Writing UDP to {json_file}")
with open(json_file, "w", encoding="UTF8") as f:
    json.dump(spec, f, indent=2)
