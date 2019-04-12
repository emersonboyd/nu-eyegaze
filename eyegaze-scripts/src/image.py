                                                                                                           #!/usr/bin/python3
import os
from time import sleep
#import picamera
#import pygame
from datetime import datetime as dt
import RPi.GPIO as GPIO
import sys
#sys.path.insert(0, 'home/pi/ivport-v2/')
import test_ivport_quad
from flask_client import send_images
#import vlc
import pygame
import ivport
import socket
from urllib import request
from util import parse_server_response

import util


socket.setdefaulttimeout(23)  # timeout in seconds

url = 'http://google.com/'

def check_wifi(url):
    try :
        response = request.urlopen(url)
    except request.HTTPError as e:
        #print('The server couldnt fulfill the request. Reason: ' + str(e.code))
        return 'wifi disconnected'
    except request.URLError as e:
        #print('We failed to reach a server. Reason: ' + str(e.reason))
        return 'wifi disconnected'
    else :
        html = response.read()
        #print('Wifi connected!')
        return 'WiFi connected!'


def init_button():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(40, GPIO.IN, pull_up_down = GPIO.PUD_UP) #button to PIN 33/GPIO 13

print('Initializing button')
init_button()

# init picam
try:
    iv = ivport.IVPort(ivport.TYPE_QUAD2, iv_jumper='A')
    iv.close()
    # (3280,2464)
    iv.camera_open(camera_v2=True, resolution=(1920,1080))
    print('IVPort initialized')
except:
    print('probably out of resources(other than memory) error')

#print('Initializing Camera.')
#camera = picamera.PiCamera()
#camera.start_preview()

while True:
    # take pic using 'enter' on keyboard
    # string = input()
    # if string == 'q':
        #exit()

    button_state = GPIO.input(40)
    if button_state == False:
        if check_wifi(url) == 'wifi disconnected':
            os.system('omxplayer -o local {}/wifi_disconnected.mp3'.format(res_dir))
            print('wifi disconnected')
            break
        print("button was pressed!:")
        # sleep(0.75) allow time to adjust to light levels
        # camera.capture('image' + str(datetime.datetime.now()) + '.jpg')
        #test_ivport_quad.picam_capture()

        # IN THIS SCRIPT, IMAGE1 IS DEFINITELY THE LEFT CAM FROM THE PERSPECTIVE OF THE USER WEARING THE GLASSES

        image_name_left = '/home/pi/Pictures/image_left'
        image_name_right = '/home/pi/Pictures/image_right'

        iv.camera_change(1)
        iv.camera_capture(image_name_left, use_video_port=False) #+ str(datetime.now()), use_video_port=False)
        print(str(dt.now()))
        print('image 1 complete')
        #sleep(5)
        iv.camera_change(3)
        #sleep(5)
        iv.camera_capture(image_name_right, use_video_port=False) # + str(datetime.now()), use_video_port=False)
        #print (str(datetime.now()))
        print('image 3 complete')

        print('sending left image to webserver...')
        #labels1 = send_image("image1_CAM1.jpg")
        response = send_images(image_name_left + '_CAM1.jpg', image_name_right + '_CAM3.jpg')
        print('Raw response from server: {}'.format(response.text))

        parsed_response = parse_server_response(response.text)
        print(parsed_response)

        res_dir = util.get_resources_directory()

        if len(parsed_response) == 0:
                os.system('omxplayer -o local {}/no_detection.mp3'.format(res_dir))

        for tup in parsed_response:
                class_audio_file_name = '{}/{}'.format(res_dir, tup[0])
                dist_audio_file_name = '{}/distance/{}'.format(res_dir, tup[1])
                angle_audio_file_name = '{}/angle/{}'.format(res_dir, tup[2])
                os.system('omxplayer -o local {}'.format(class_audio_file_name))
                os.system('omxplayer -o local {}'.format(dist_audio_file_name))
                os.system('omxplayer -o local {}'.format(angle_audio_file_name))

	#print('sending right image to webserver...')
        #labels2 = send_image("image2_CAM2.jpg")
        #print(labels2)
        #i = vlc.Instance('--verbose 3')

        #if labels1 == 'exit_sign':
        #    os.system("omxplayer -o local exit_sign.mp3")
        #elif labels1 == 'bathroom_sign':
        #    os.system("omxplayer -o local bathroom_sign.mp3")
        #else:
        #    print('No label, not running audio')


iv.close()
print('IVPort closed')
#camera.stop_preview()

