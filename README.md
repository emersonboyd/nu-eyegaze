# NU Eyegaze Capstone
Optical Character Recognition and Sign Detection Glasses for the Visually Impaired

[Project Description](https://docs.google.com/presentation/d/1Eew1FWWk76oY9wtT70f1i_0qqgfPZqYRX5wbpYT6G0Q/edit?usp=sharing)

[Youtube Video Demonstration](https://youtu.be/peDaGbRYc8M)

Group Members: 
Brendon Welsh
Emerson Boyd
Aleksandra Pasek
Nathaniel Hartwig
Melissa Chen

## Opening the Virtual Environment
On the raspberry pi, run `pytyon3 -m virtualenv NUeyegaze/dev/bin/activate`

## Installing Packages
`python3 -m pip install  <package>`

## Running for The First Time
- Setup pi to run on WiFi
- run init_ivport.py on startup(in .bashrc or something) before image.py
- enable Camera, SSH, VNC, SPI, I2C, Serial Port, 1 Wire, Remote GPIO under "interfaces" in RPi Configuration settings.  
- make sure "listen_for_button_press.sh" is configured. need button input program otherwise image.py won't respond to button press. 
