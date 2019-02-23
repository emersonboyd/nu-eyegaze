from flask import Flask, request
from google.cloud import storage
import os
import socket
import sys
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

@app.route('/hello', methods=["POST"])
@cross_origin()
def get_hello():
    return "hello"

