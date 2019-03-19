#!/usr/bin/env python

import subprocess
import pexpect

child = pexpect.spawn("bluetoothctl")
#child.expect('[bluetooth]#')
print("spawned bluetoothctl")
child.sendline("power on")
#child.expect('[Q13]#')
child.sendline("agent on")
#child.expect('[Q13]#')
child.sendline("default-agent")
#child.expect('[Q13]#')
child.sendline("connect EB:06:BF:35:3F:16")
print("connected device Q13")
#child.expect('[Q13]#')
child.sendline("quit")
child.close()
print("closed child process")

subprocess.Popen(["pacmd", "set-card-profile", "bluez_card.EB_06_BF_35_3F_16 a2dp_sink"], stdout=subprocess.PIPE) 
print("pacmd card profile set")
subprocess.Popen(["pacmd", "set-default-sink", "bluez_card.EB:06:BF:35:3F:16"], stdout=subprocess.PIPE)
print("pacmd default sink set")

