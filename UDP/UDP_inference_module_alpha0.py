"""
UDP to run the inference module on openEO with given DataCube and ML model
This version for the alpha0 release is still a dummy and only loads a pre-processed map.

"""
import json
import openeo
from openeo.api.process import Parameter
from openeo.processes import if_, eq, text_concat, add
from openeo.rest.udp import build_process_dict
import os
import pathlib

# Establish connection to OpenEO instance (note that authentication is not necessary to just build the UDP)
connection = openeo.connect(url="openeo.dataspace.copernicus.eu")
# set up the UDP parameters
param_geo = Parameter.geojson(
    name="geometry",
    description="Geometry as GeoJSON feature(s).",
)
param_year = Parameter.integer(
    name="year",
    default=2021,
    description="The year for which to generate the habitat map. (default: 2021)",
)
param_topology = Parameter.string(
    name="topology",
    default='EUNIS2012',
    description="The topology to run the habitat mapping for. (default: EUNIS2012)",
)
param_topology_level = Parameter.integer(
    name="topology_level",
    default=1,
    description="The topology level to run the habitat mapping for. (default: 1)",
)
param_td = Parameter.geojson(
    name="reference_data",
    description="Geometry of reference data as GeoJSON feature(s).",
)
param_feature_ids = Parameter(
    name="remapping_dict",
    default=['cgls_dem_30', 'temp', 'watercontent'],
    description="List of str handing over all needed ecosystem characteristics layers to run model training.",
    schema={"type": "list"},
)
param_model = Parameter.string(
    name="ml_model_type",
    default='CatBoost',
    description="The name of the ML model type to use for the inference. (default: CatBoost)",
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

# set the request year for the data
start = text_concat([param_year, "01", "01"], separator="-")
end = text_concat([add(param_year, 1), "01", "01"], separator="-")

# specify the needed data locations
cube_L1 = connection.load_disk_collection(format="GTiff",
                                          glob_pattern="/data/users/Public/buchhornm/prepared_SK_alpha0_habitat-maps/Slovakia_v5_*_L1.tif",
                                          options=dict(date_regex='.*_(\d{4})(\d{2})(\d{2})_L1.tif'))
cube_L2 = connection.load_disk_collection(format="GTiff",
                                          glob_pattern="/data/users/Public/buchhornm/prepared_SK_alpha0_habitat-maps/Slovakia_v5_*_L2.tif",
                                          options=dict(date_regex='.*_(\d{4})(\d{2})(\d{2})_L2.tif'))
cube_L3 = connection.load_disk_collection(format="GTiff",
                                          glob_pattern="/data/users/Public/buchhornm/prepared_SK_alpha0_habitat-maps/Slovakia_v5_*_L3.tif",
                                          options=dict(date_regex='.*_(\d{4})(\d{2})(\d{2})_L3.tif'))

# filter thematic
cube = if_(eq(param_topology_level, 1), cube_L1, if_(eq(param_topology_level, 2), cube_L2, cube_L3))

# filter spatial and temporal
cube = cube.filter_spatial(geometries=param_geo)
cube = cube.filter_temporal([start, end])

# warp to specified projection and resolution if needed
cube = cube.resample_spatial(resolution=param_resolution, projection=param_epsg, method="near")


description = """
Loads the preprocessed habitat map for SK for the alpha0 release.
"""
spec = build_process_dict(
    process_id="udp_inference_module_alpha0",
    summary="Loads the prepared 2021 habitat map for SK in EUNIS2012 topology for requested topology level. "
            "Returns a single band RasterCube.",
    description=description.strip(),
    parameters=[
        param_geo,
        param_year,
        param_topology,
        param_topology_level,
        param_td,
        param_feature_ids,
        param_model,
        param_epsg,
        param_resolution,
    ],
    process_graph=cube,
)

# dump to json file to be usable as UDP
this_script = pathlib.Path(__file__)
json_file = os.path.normpath(os.path.join(this_script.parent, 'json', this_script.name.lower().split('.')[0] + '.json'))
print(f"Writing UDP to {json_file}")
with open(json_file, "w", encoding="UTF8") as f:
    json.dump(spec, f, indent=2)
