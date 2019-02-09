from flask import Flask, request
from flask_cors import CORS, cross_origin
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