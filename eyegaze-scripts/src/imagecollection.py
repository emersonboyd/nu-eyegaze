#!/usr/bin/python3
import os
from time import sleep
#import picamera
#import pygame
import datetime
import RPi.GPIO as GPIO
import sys
from time import sleep
#sys.path.insert(0, 'home/pi/ivport-v2/')
import test_ivport_quad
#from flask_client import send_image
#import vlc
import pygame


def init_button():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(40, GPIO.IN, pull_up_down = GPIO.PUD_UP) #button to PIN 33/GPIO 13

print('Initializing button')
init_button()

#print('Initializing Camera.')
#camera = picamera.PiCamera()
#camera.start_preview()


while True:
    sleep(5) # sleep for five seconds
    # take pic using 'enter' on keyboard
    # string = input()
    # if string == 'q':
        #exit()
    button_state = GPIO.input(40)
    if button_state == False or True:
        # sleep(0.75) allow time to adjust to light levels
        # camera.capture('image' + str(datetime.datetime.now()) + '.jpg')
        print("button pressed")
        #test_ivport_quad.picam_capture()
        print("pictures taken")
        #print('sending left image to webserver...')
        #labels1 = send_image("image1_CAM1.jpg")
        #print(labels1)
        #print('sending right image to webserver...')
        #labels2 = send_image("image2_CAM2.jpg")
        #print(labels2)
        #i = vlc.Instance('--verbose 3')
        #if labels1 == 'exit_sign':
        os.system("omxplayer -o local ../res/exitsign_edit.mp3")
        os.system("omxplayer -o local ../res/distance/10feet.mp3")
        os.system("omxplayer -o local ../res/angle/1oclock.mp3")
        #else:
        #    os.system("omxplayer -o local bathroom_sign.mp3")

#camera.stop_preview()

