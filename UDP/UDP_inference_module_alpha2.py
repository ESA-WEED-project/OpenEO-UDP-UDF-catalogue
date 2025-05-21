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
import pathlib
import openeo
from openeo.metadata import metadata_from_stac
import json
# WEED project developments"
from eo_processing.config import get_job_options, get_collection_options, get_standard_processing_options
from eo_processing.utils.helper import getUDFpath
from eo_processing.utils.metadata import get_base_metadata
from openeo.api.process import Parameter
from openeo.processes import text_concat
from openeo.rest.udp import build_process_dict
from eo_processing.openeo.processing import generate_master_feature_cube

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
param_onnx_model = Parameter(
    name="onnx_model_id",
    description="1 onnx_model_id used to run.",
    schema={"type": "string"},
) #iso url , it's just the ID TD

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
)

# set the request year for the data
start = text_concat([param_year, "01", "01T00:00:00Z" ], separator="-")
end = text_concat([param_year, "12", "31T00:00:00Z"], separator="-")


file_namea = text_concat(["Alpha2_EUNIS-habitat-proba-cube_year",param_year ],separator="")
file_name = text_concat([file_namea,param_name,param_onnx_model,param_scenarioId], separator="_")
s3_prefix = text_concat([param_digitalId,param_scenarioId], separator="-")

# convert the row name into a openEO bbox dict giving the spatial extent of the job
processing_extent = param_bbox
# define job_options, processing_options,  and collection_options
job_options = get_job_options(provider=provider, task='inference')
collection_options = get_collection_options(provider=provider)
processing_options = get_standard_processing_options(provider=provider, task='feature_generation')

# updates to job_options
job_options.update({"allow_empty_cubes": True})  # that cubes in areas with no data are created for a STAC and not an error
job_options.update({"export-workspace-enable-merge": True})  # that items for STAC catalog are merged if in same S3 prefix
job_options.update({'etl_organization_id': '4938'})  # billing to specific organization

#### increasing memory for alpha-2 for testing
job_options.update({"driver-memory": "4G",
                    "driver-memoryOverhead": "4G",
                    "executor-memory": "2G",
                    "executor-memoryOverhead": "1000m",
                    "python-memory": "5G",
                    "logging-threshold": "debug"})
#### end of delete

#### create-on-demand the feature datacube
# define the S1/S2 processed feature cube (Note: do not set spatial extent since we hand it over in the end TODO boox: None ?)
data_cube = generate_master_feature_cube(connection,
                                         processing_extent,
                                         start,
                                         end,
                                         **collection_options,
                                         **processing_options)

# now we merge in the NON ON-DEMAND processed features (DEM and WENR features)
# load the DEM from a CDSE collection
DEM = connection.load_collection("COPERNICUS_30",
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
WENR = connection.load_stac("https://catalogue.weed.apex.esa.int/collections/wern_features",
                            bands=metadata_from_stac(
                                "https://catalogue.weed.apex.esa.int/collections/wern_features").band_names,
                            spatial_extent=processing_extent)
# resample the cube to 10m and EPSG of corresponding 20x20km grid tile
WENR = WENR.resample_spatial(projection=processing_options['target_crs'],
                             resolution=processing_options['resolution'],
                             method="near")
# drop the time dimension
try:
    WENR = WENR.drop_dimension('t')
except:
    # workaround if we still have the client issues with the time dimensions for STAC dataset with only one time stamp
    WENR.metadata = WENR.metadata.add_dimension("t", label=None, type="temporal")
    WENR = WENR.drop_dimension('t')
# merge into the S1/S2 data cube
data_cube = data_cube.merge_cubes(WENR)

# load the GLOBES features from public STAC
GLOBES = connection.load_stac("https://catalogue.weed.apex.esa.int/collections/Globes-V1",
                              bands=metadata_from_stac(
                                  "https://catalogue.weed.apex.esa.int/collections/Globes-V1").band_names,
                              spatial_extent=processing_extent,
                              temporal_extent=[start, end])
# resample the cube to 10m and EPSG of corresponding 20x20km grid tile
GLOBES = GLOBES.resample_spatial(projection=processing_options['target_crs'],
                                 resolution=processing_options['resolution'],
                                 method="near")
# drop the time dimension
try:
    GLOBES = GLOBES.drop_dimension('t')
except:
    # workaround if we still have the client issues with the time dimensions for STAC dataset with only one time stamp
    GLOBES.metadata = GLOBES.metadata.add_dimension("t", label=None, type="temporal")
    GLOBES = GLOBES.drop_dimension('t')
# merge into the S1/S2 data cube
data_cube = data_cube.merge_cubes(GLOBES)

# filter spatial the whole cube
data_cube = data_cube.filter_bbox(processing_extent)

#### run multi-model inference
# we pass the modelID as context information within the UDF
udf = openeo.UDF.from_file(
    getUDFpath('udf_catboost_inference.py'),
    context={"model_id": param_onnx_model})

# Apply the UDF to the data cube.
proba_cube = data_cube.apply_dimension(process=udf, dimension="bands")
proba_cube = proba_cube.linear_scale_range(0, 100, 0, 100)

#### create job progress graph including storage to S3
# prepare metadata for raster output (GTiff, NetCDF)
file_meta = get_base_metadata(project='WEED')
file_meta.update(description=f'habitat occurrence probability by model inference',
                 tiling_grid='global_grid20',
                 product_tile='from_param_name',
                 time_start='from_start',
                 time_end='from_end')
# ToDo: prepare also needed band metadata

saved_cube = proba_cube.save_result(format="GTiff",
                                    options={
                                        'separate_asset_per_band': False,
                                        'filename_prefix': "file_name",
                                        'file_metadata': file_meta,
                                    })

# save to S3 and directly publish results as STAC catalog on WEED STAC.api if export workspace is configured for this
# TODo: add the export_workspace for S3 saving AND directly STAC catalog creation and publishing into the VITO vault
cube_workspace = saved_cube.export_workspace(workspace="esa-weed-apex-stac-api-workspace",
                                             merge=s3_prefix)



description = """
Inference for the habitat maps for the alpha2 release.
"""
spec = build_process_dict(
    process_id="udp_inference_module_alpha2",
    summary="Generates the alpha 2 inference result based on inputs."
            "Returns a single band per probability.",
    description=description.strip(),
    parameters=[
        param_bbox,
        param_year,
        param_onnx_model,
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
