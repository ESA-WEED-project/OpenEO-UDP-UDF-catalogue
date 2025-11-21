"""
UDP to run the inference module on openEO with given DataCube and ML model
This version for the alpha2 release. There are a few limitation
1) The text_concat on the filename prefix is not seriazable somehow in the save_results node, so needs to be adapted manually in the json.
        add module in json
    "textconcat2": {
      "process_id": "text_concat",
      "arguments": {
        "data": [
          {
            "from_parameter": "year"
          },
          "01",
          "01T00:00:00Z"
        ],
        "separator": "-"
      }
    },
    "textconcat3": {
      "process_id": "text_concat",
      "arguments": {
        "data": [
          {
            "from_parameter": "year"
          },
          "12",
          "31T23:59:59Z"
        ],
        "separator": "-"
      }
    },
        "textconcat4": {
      "process_id": "text_concat",
      "arguments": {
        "data": [
          "Alpha3_EUNIS-extent-map_year",
          {"from_parameter": "year"},
          "_",
          {"from_parameter": "area_name"},
          "_",
          {"from_parameter": "scenarioId"}]
      }
    },
and replace file_name with {"from_node": "textconcat4"} in saveresult1 and as well
            "product_tile": {"from_parameter": "area_name"},
            "time_start": {"from_node": "textconcat2"},
            "time_end": {"from_node": "textconcat3"}

create concat of STAC url
"textconcat5": {
      "process_id": "text_concat",
      "arguments": {
        "data": [
          "https://catalogue.weed.apex.esa.int/collections/",
          {"from_node": "textconcat"}
        ],
        "separator": "/"
        }
    }


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
from eo_processing.config import get_job_options, get_collection_options, get_standard_processing_options
from eo_processing.utils.helper import init_connection, getUDFpath
from eo_processing.utils.metadata import get_base_metadata

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

param_onnx_model = Parameter.string(
    name="onnx_model",
    description="ONNX model path or identifier",
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

###################################
s3_prefix = text_concat([param_digitalId,param_scenarioId], separator="-")

# convert the row name into a openEO bbox dict giving the spatial extent of the job
processing_extent = param_bbox

job_options = {"driver-memory": "512m",
                    "driver-memoryOverhead": "512m",
                    "executor-memory": "512m",
                    "executor-memoryOverhead": "512m",
                    "logging-threshold": "debug",
               "etl_organization_id": "4938"}


geojson = {"type":"FeatureCollection",
           "features": [
               {"type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[-74.03, 40],
                                                                [-74.000, 40],
                                                                [-74.000, 40.03],
                                                                [-74.03, 40.03],
                                                               [-74.03, 40]]]
                            },
               "properties": {"digitalId": "param_digitalId" ,
                              "scenarioId":"param_scenarioId",
                              "onnx_model": "param_onnx_model",
                              "year" : "param_year",
                              "spatial_extent" : "param_bbox"
                             }}
           ]
          }


cube = connection.load_geojson(json.dumps(geojson))

result = cube.save_result(format = "geojson",options={'filename_prefix': "s3_prefix"})
cube_workspace = result.export_workspace(workspace="esa-weed-test-workspace", merge = "test_bert_vector")



description = """
UDP starter.
"""

spec = build_process_dict(
    process_id="udp_starter",
    summary="Generates the starter script for Aries.",
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
