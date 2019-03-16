from flask import Flask, request
from flask_cors import CORS, cross_origin
from label_image import label_image, get_response_string_with_image_path
from google.cloud import storage
import os
import socket
import sys

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = ''

@app.route('/hello', methods=["POST"])
@cross_origin()
def print_hello():
    return 'hello'

@app.route('/inf', methods=["POST"])
@cross_origin()
def run_model():

    image_left = request.files['image_left']
    image_left.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image_left.jpg'))
    image_right = request.files['image_right']
    image_right.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image_right.jpg'))
    #labels = label_image(request.files)
    labels = get_response_string_with_image_paths('image_left.jpg')
    return labels


if __name__ == '__main__':
    print("Starting Webserver")
    app.run(host="0.0.0.0", port=5000, debug=True)
