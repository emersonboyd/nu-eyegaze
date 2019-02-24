import numpy as np
import json
import requests
import random
import time

def send_image(image_name, ip="35.185.63.125", port="5000", command="inf"):
    """
    Send an image to the Google Cloud Server

    """

    img = open(image_name, 'rb')
    post_command = "http://" + ip + ":" + port + "/" + command
    print(post_command)
    files = {'media': img}
    request = requests.post(post_command, files=files)
    time.sleep(.1)
    return request.text
