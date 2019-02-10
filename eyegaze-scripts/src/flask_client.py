import numpy as np
import json
import requests
import random
import time
while True:

    request = requests.post("http://35.185.63.125:5000/hello", data='data')
    time.sleep(.1)
    print(request.text)