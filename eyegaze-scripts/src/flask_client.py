
import requests
import time

while True:
    request = requests.post("http://35.138.63.125:5000/hello")
    time.sleep(.1)
    print(request.text)