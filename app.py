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


@app.route('/')
def empty_path():
    return redirect(url_for('upload_page'))


@app.route('/index.html')
def index():
    return redirect(url_for('upload_page'))


@app.route("/comparison.html")
def comparison_page():
    return render_template("comparison.html")


@app.route("/upload.html")
def upload_page():
    return render_template("upload.html")


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

    return jsonify(resp)


@app.route("/uploadService", methods=["POST"])
def upload():
    uuids = []
    for file in request.files:
        tmp_file = NamedTemporaryFile()
        tmp_file.close()
        fo = request.files[file]
        fo.save(tmp_file.name)
        this_uuid = str(uuid.uuid4())
        logs_dictio[this_uuid] = tmp_file.name
        uuids.append(this_uuid)
    return {"uuid1": uuids[0], "uuid2": uuids[1]}


if __name__ == "__main__":
    if not os.path.exists(os.path.join("static", "temp")):
        os.mkdir(os.path.join("static", "temp"))
    logs_dictio["log1"] = "C:/running-example.xes"
    logs_dictio["log2"] = "C:/running-example.xes"
    app.run(host='0.0.0.0')
