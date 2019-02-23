import requests
import time

while True:
    request = requests.post("https//35.185.63.125:5000/hello")
    time.sleep(.1)
    print(request.text)
