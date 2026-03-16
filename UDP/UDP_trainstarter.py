"""
UDP to run the inference module on openEO with given DataCube and ML model
This version for the alpha2 release. There are a few limitation
1) at moment just replace whole grpah with
{
  "process_graph": {
    "textconcat1": {
      "process_id": "text_concat",
      "arguments": {
        "data": [
          {
            "from_parameter": "digitalId"
          },
          {
            "from_parameter": "scenarioId"
          }
        ],
        "separator": "-"
      }
    },
    "loadgeojson1": {
      "process_id": "load_geojson",
      "arguments": {
        "data": {
          "type": "FeatureCollection",
          "features": [
            {
              "type": "Feature",
              "geometry": {
                "type": "Polygon",
                "coordinates": [
                  [
                    [
                      -74.03,
                      40
                    ],
                    [
                      -74.0,
                      40
                    ],
                    [
                      -74.0,
                      40.03
                    ],
                    [
                      -74.03,
                      40.03
                    ],
                    [
                      -74.03,
                      40
                    ]
                  ]
                ]
              },
              "properties": {
                "digitalId": {"from_parameter": "digitalId"},
                "scenarioId": {"from_parameter": "scenarioId"},
                "rdm_table": {"from_parameter": "rdm_table"},
                "year": {"from_parameter": "year"},
                "spatial_extent": {"from_parameter": "geometry"},
                "dt_url": {"from_parameter": "dt_url"},
                "workflow":"training"
              }
            }
          ]
        },
        "properties": []
      }
    },
    "saveresult1": {
      "process_id": "save_result",
      "arguments": {
        "data": {
          "from_node": "loadgeojson1"
        },
        "format": "geojson",
        "options": {
          "filename_prefix": {
            "from_node": "textconcat1"
          }
        }
      }
    },
    "exportworkspace1": {
      "process_id": "export_workspace",
      "arguments": {
        "data": {
          "from_node": "saveresult1"
        },
        "merge": "test_bert_vector",
        "workspace": "esa-weed-test-workspace"
      },
      "result": true
    }
  },
  "id": "udp_trainstarter",
  "summary": "Generates the training starter script for Aries.",
  "description": "UDP training starter.",
  "default_job_options": {
    "driver-memory": "512m",
    "driver-memoryOverhead": "512m",
    "executor-memory": "512m",
    "executor-memoryOverhead": "512m",
    "logging-threshold": "debug",
    "etl_organization_id": "4938"
  },
  "parameters": [
    {
      "name": "geometry",
      "description": "GeoJSON Geometry Object.",
      "schema": {
        "type": "object",
        "coordinates": []
      }
    },
    {
      "name": "year",
      "description": "The year for which to generate the habitat map. (default: 2024)",
      "schema": {
        "type": "integer"
      },
      "default": 2024,
      "optional": true
    },
    {
      "name": "rdm_table",
      "description": "RDM postGres table name",
      "schema": {
        "type": "string"
      },
      "default": "global_training",
      "optional": true
    },
    {
      "name": "digitalId",
      "description": "Digital ID of client",
      "schema": {
        "type": "string"
      }
    },
    {
      "name": "scenarioId",
      "description": "Id of the scenario/session",
      "schema": {
        "type": "string"
      }
    },
    {
      "name": "area_name",
      "description": "Name of the AOI",
      "schema": {
        "type": "string"
      },
      "optional": true,
      "default": "AOI"
    },
    {
      "name": "dt_url",
      "description": "url of the digital twin",
      "schema": {
        "type": "string"
      },
      "optional": true,
      "default": "https://services.integratedmodelling.org/runtime/main/api/v1/dt/ESA_INSTITUTIONAL.rvr3s2juw0"
    }
  ]
}
"""
import os
import pathlib
import openeo
import json

from openeo.api.process import Parameter
from openeo.processes import text_concat
from openeo.rest.udp import build_process_dict


# Establish connection to OpenEO instance (note that authentication is not necessary to just build the UDP)
provider = 'cdse' #this udp works only on cdse
connection = openeo.connect(url="openeo.dataspace.copernicus.eu")
# set up the UDP parameters
param_geometry = Parameter(name="geometry",
    description="""GeoJSON Geometry Object.""",
    schema= {"type": "object",
          "coordinates": []
    })

param_year = Parameter.integer(
    name="year",
    default=2024,
    description="The year for which to generate the habitat map. (default: 2024)",
)

param_rdm_table = Parameter.string(
    name="rdm_table",
    description="RDM postGres table name",
    default = "global_training"
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
param_dt_url = Parameter.string(
    name="dt_url",
    description="url of the digital twin",
    optional = True,
    default = "https://services.integratedmodelling.org/runtime/main/api/v1/dt/ESA_INSTITUTIONAL.rvr3s2juw0"
)

###################################
s3_prefix = text_concat([param_digitalId,param_scenarioId], separator="-")



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
                              "spatial_extent" : "geometry",
                              "dt_url":"param_dt_url"
                             }}
           ]
          }


cube = connection.load_geojson(json.dumps(geojson))

result = cube.save_result(format = "geojson",options={'filename_prefix': "s3_prefix"})
cube_workspace = result.export_workspace(workspace="esa-weed-test-workspace", merge = "test_bert_vector")



description = """
UDP training starter.
"""

spec = build_process_dict(
    process_id="udp_trainstarter",
    summary="Generates the training starter script for Aries.",
    description=description.strip(),
    parameters=[
        param_geometry,
        param_year,
        param_rdm_table,
        param_digitalId,
        param_scenarioId,
        param_name,
        param_dt_url
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
