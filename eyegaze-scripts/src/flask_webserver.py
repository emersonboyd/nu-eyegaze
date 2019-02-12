from flask import Flask, request
from flask_cors import CORS, cross_origin
from label_image import label_image
from google.cloud import storage
import os
import socket
import sys

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/hello', methods=["POST"])
@cross_origin()
def print_hello():
    return 'hello'

@app.route('/inf', methods=["POST"])
@cross_origin()
def run_model():

    labels = label_image(request.files)
    return labels


if __name__ == '__main__':
    print("Starting Webserver")
    app.run(host="0.0.0.0", port=5000, debug=True)