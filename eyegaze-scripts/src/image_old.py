                                                                                                           #!/usr/bin/python3

from time import sleep
#import picamera
#import pygame
import datetime
import RPi.GPIO as GPIO
import sys
#sys.path.insert(0, 'home/pi/ivport-v2/')
import test_ivport_quad

def init_button():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(33, GPIO.IN, pull_up_down = GPIO.PUD_UP) #button to PIN 33/GPIO 13


print('Initializing button')
init_button()

#print('Initializing Camera.')
#camera = picamera.PiCamera()
#camera.start_preview()

while True:
    # take pic using 'enter' on keyboard
    # string = input()
    # if string == 'q':
        #exit()
    button_state = GPIO.input(33)
    if button_state == False:
        # sleep(0.75) allow time to adjust to light levels
        # camera.capture('image' + str(datetime.datetime.now()) + '.jpg')
        test_ivport_quad.picam_capture()

#camera.stop_preview()




