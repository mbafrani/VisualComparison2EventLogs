from flask import Flask, render_template, request, make_response, jsonify, redirect, url_for
import base64, os
from tempfile import NamedTemporaryFile
import uuid
from copy import copy
from EMDMeasurment.ComparisonMethods import produce_visualizations_from_event_logs_paths
from flask_cors import CORS
import json
import traceback


app = Flask(__name__)
CORS(app, expose_headers=["x-suggested-filename"])


logs_dictio = {}


@app.route("/visualizations.html")
def visualizations_page():
    return ""


@app.route("/visualizationsService", methods=["GET"])
def visualizationsService():
    uid1 = request.args.get("uid1")
    uid2 = request.args.get("uid2")
    if uid1 is None:
        uid1 = "log1"
    if uid2 is None:
        uid2 = "log2"

    log_path1 = logs_dictio[uid1]
    log_path2 = logs_dictio[uid2]

    resp = produce_visualizations_from_event_logs_paths(log_path1, log_path2)

    for key in resp:
        resp[key] = open(resp[key], "r").read()

    return jsonify(resp)


if __name__ == "__main__":
    logs_dictio["log1"] = "C:/running-example.xes"
    logs_dictio["log2"] = "C:/running-example.xes"
    app.run(host='0.0.0.0')
