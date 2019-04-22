#!/usr/bin/env python

# Openbci2ft reads data from an OpenBCI device and writes that data to a FieldTrip buffer
#
# This module is part of the EEGsynth project (https://github.com/eegsynth/eegsynth)
#
# Copyright (C) 2019 EEGsynth project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import configparser
import argparse
import numpy as np
import os
import redis
import sys
import time
import threading
from scipy import signal as sp

sys.path.append('/Users/roboos/tmp/OpenBCI_Python')
import openbci

if hasattr(sys, 'frozen'):
    basis = sys.executable
elif sys.argv[0]!='':
    basis = sys.argv[0]
else:
    basis = './'
installed_folder = os.path.split(basis)[0]

# eegsynth/lib contains shared modules
sys.path.insert(0, os.path.join(installed_folder,'../../lib'))
import EEGsynth
import FieldTrip

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inifile", default=os.path.join(installed_folder, os.path.splitext(os.path.basename(__file__))[0] + '.ini'), help="optional name of the configuration file")
args = parser.parse_args()

config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
config.read(args.inifile)

try:
    r = redis.StrictRedis(host=config.get('redis','hostname'), port=config.getint('redis','port'), db=0)
    response = r.client_list()
except redis.ConnectionError:
    print("Error: cannot connect to redis server")
    exit()

# combine the patching from the configuration file and Redis
patch = EEGsynth.patch(config, r)
del config

# this determines how much debugging information gets printed
debug = patch.getint('general','debug')
delay = patch.getint('general','delay')

try:
    ftc_host = patch.getstring('fieldtrip','hostname')
    ftc_port = patch.getint('fieldtrip','port')
    if debug>0:
        print('Trying to connect to buffer on %s:%i ...' % (ftc_host, ftc_port))
    ft_output = FieldTrip.Client()
    ft_output.connect(ftc_host, ftc_port)
    if debug>0:
        print("Connected to output FieldTrip buffer")
except:
    print("Error: cannot connect to output FieldTrip buffer")
    exit()

# the variable naming follows the OpenBCI code
args.board = patch.getstring('openbci', 'board')
args.port = None
args.baud = 115200
args.filtering = False
args.log = False
args.aux = None
args.daisy = False
args.high_speed = False
args.ip_address = None

if args.board == "cyton":
    import openbci.cyton as bci
    args.port = patch.getstring('openbci', 'serial')
    board = bci.OpenBCICyton(port=args.port,
                             baud=args.baud,
                             daisy=args.daisy,
                             filter_data=args.filtering,
                             scaled_output=True,
                             log=args.log)
elif args.board == "ganglion":
    import openbci.ganglion as bci
    board = bci.OpenBCIGanglion(port=args.port,
                                filter_data=args.filtering,
                                scaled_output=True,
                                log=args.log,
                                aux=args.aux)
elif args.board == "wifi":
    import openbci.wifi as bci
    args.shield_name = patch.getstring('openbci', 'name')
    board = bci.OpenBCIWiFi(shield_name=args.shield_name,
                                ip_address=args.ip_address,
                                log=args.log,
                                high_speed=args.high_speed)
    # supported for Cyton are 250, 500, 1000, 2000, 4000, 8000, 16000
    # supported for Ganglion are 200, 400, 800, 1600, 3200, 6400, 12800, 25600
    if board.getBoardType() == "cyton":
        board.set_sample_rate(250)
    elif board.getBoardType() == "ganglion":
        board.set_sample_rate(200)

fsample = board.getSampleRate()
nchans = board.getNbEEGChannels() # board.getNbAUXChannels()

if debug > 0:
    print("fsample", fsample)
    print("nchans", nchans)

datatype = FieldTrip.DATATYPE_FLOAT32
ft_output.putHeader(nchans, float(fsample), datatype)

start = time.time()
count = 0

def streamSample(sample):
    global count
    global debug
    count += 1
    if debug>0 and (count % 100)==0:
        print("sample ", count)
    dat = np.array([sample.channel_data])
    ft_output.putData(dat.astype(np.float32))

print("STARTING STREAM")
board.start_streaming(streamSample)

if args.board == "wifi":
    board.loop()
