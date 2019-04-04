# NU Eyegaze Capstone
Optical Character Recognition and Sign Detection Glasses for the Visually Impaired

## Opening the Virtual Environment
On the raspberry pi, run `pytyon3 -m virtualenv NUeyegaze/dev/bin/activate`

## Installing Packages
`python3 -m pip install  <package>`

## Running for The First Time
- Setup pi to run on School WiFi
- run init_ivport.py on startup(in .bashrc or something) before image.py
- enable Camera, SSH, VNC, SPI, I2C, Serial Port, 1 Wire, Remote GPIO under "interfaces" in RPi Configuration settings.  
- make sure "listen_for_button_press.sh" is configured. need button input program otherwise image.py won't respond to button press. 
