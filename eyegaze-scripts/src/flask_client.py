import numpy as np
import json
import requests
import random
import time

img = open('res/test_photos/20190108_091908.jpg', 'rb')
files = {'media': img}
request = requests.post("http://35.185.63.125:5000/inf", files=files)
time.sleep(.1)
print(request.text)