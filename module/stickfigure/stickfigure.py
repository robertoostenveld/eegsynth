#!/usr/bin/env python

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import configparser
import redis
import argparse
import numpy as np
import os
import time
import signal
import sys
import math
from color import color2rgba

if hasattr(sys, 'frozen'):
    path = os.path.split(sys.executable)[0]
    file = os.path.split(sys.executable)[-1]
elif sys.argv[0] != '':
    path = os.path.split(sys.argv[0])[0]
    file = os.path.split(sys.argv[0])[-1]
else:
    path = os.path.abspath('')
    file = os.path.split(path)[-1] + '.py'

# eegsynth/lib contains shared modules
sys.path.insert(0, os.path.join(path, '../../lib'))
import EEGsynth
import FieldTrip

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inifile", default=os.path.join(path, os.path.splitext(file)[0] + '.ini'), help="optional name of the configuration file")
args = parser.parse_args()

config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
config.read(args.inifile)

try:
    r = redis.StrictRedis(host=config.get('redis','hostname'), port=config.getint('redis','port'), db=0)
    response = r.client_list()
except redis.ConnectionError:
    raise RuntimeError("cannot connect to Redis server")

# combine the patching from the configuration file and Redis
patch = EEGsynth.patch(config, r)

# this can be used to show parameters that have changed
monitor = EEGsynth.monitor()

# get the options from the configuration file
debug = patch.getint('general', 'debug')
delay = patch.getfloat('general', 'delay')

# initialize graphical window
app = QtGui.QApplication([])

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

w = gl.GLViewWidget()
w.setWindowTitle('EEGsynth pose')

winx        = patch.getfloat('display', 'xpos')
winy        = patch.getfloat('display', 'ypos')
winwidth    = patch.getfloat('display', 'width')
winheight   = patch.getfloat('display', 'height')
w.setGeometry(winx, winy, winwidth, winheight)

distance  = patch.getint('camera', 'distance')
elevation = patch.getint('camera', 'elevation')
azimuth   = patch.getint('camera', 'elevation')
w.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)

w.show()

GLOBALSCALE = 100.
ZSHIFT = 100.

def vec2angle(x, y=None, z=None, degrees=True):
    if y == None:
        if len(x.shape)>1:
            xyz = x.flatten()
        else:
            xyz = x
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
    ax = np.atan2(sqrt(y^2+z^2),x)
    ay = np.atan2(sqrt(z^2+x^2),y)
    az = np.atan2(sqrt(x^2+y^2),z)
    if degrees:
         ax *= 180/np.pi
         ay *= 180/np.pi
         az *= 180/np.pi
    return np.array([ax, ay, az])

def transform(H, xyz):
    xyzw = np.ones((xyz.shape[0], xyz.shape[1]+1))
    xyzw[:,:-1] = xyz
    xyzw = np.matmul(xyzw, np.transpose(H))
    xyz = xyzw[:,:-1]
    return xyz

def scale(x, y, z):
    # See http://www.it.hiof.no/~borres/j3d/math/homo/p-homo.html
    S = np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
        ])
    return S

def translate(x, y, z):
    # See http://www.it.hiof.no/~borres/j3d/math/homo/p-homo.html
    T = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
        ])
    return T

def rotatex(v, degrees=True):
    # See http://www.it.hiof.no/~borres/j3d/math/homo/p-homo.html
    if degrees:
        v = math.pi * v / 180;
    c = math.cos(v)
    s = math.sin(v)
    Rx = np.array([
        [ 1,  0,  0,  0],
        [ 0,  c, -s,  0],
        [ 0,  s,  c,  0],
        [ 0,  0,  0,  1]
        ])
    return Rx

def rotatey(v, degrees=True):
    # See http://www.it.hiof.no/~borres/j3d/math/homo/p-homo.html
    if degrees:
        v = math.pi * v / 180;
    c = math.cos(v)
    s = math.sin(v)
    Ry = np.array([
        [ c,  0,  s,  0],
        [ 0,  1,  0,  0],
        [-s,  0,  c,  0],
        [ 0,  0,  0,  1]
        ])
    return Ry

def rotatez(v, degrees=True):
    # See http://www.it.hiof.no/~borres/j3d/math/homo/p-homo.html
    if degrees:
        v = math.pi * v / 180;
    c = math.cos(v)
    s = math.sin(v)
    Rz = np.array([
        [ c, -s,  0,  0],
        [ s,  c,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
        ])
    return Rz

def rotate(v, x, y, z, degrees=True):
    # See http://www.it.hiof.no/~borres/j3d/math/homo/p-homo.html
    R = np.identity(4)
    if x:
        R = np.matmul(rotatex(v, degrees), R)
    if y:
        R = np.matmul(rotatey(v, degrees), R)
    if z:
        R = np.matmul(rotatez(v, degrees), R)
    return R

class Mesh():
    def __init__(self):
        self.vertices = np.zeros((0, 3))
        self.faces  = np.zeros((0, 3))
        self.top    = None
        self.bottom = None
    def append(self, vertices, faces=None):
        if faces is None:
            mesh     = vertices
            vertices = mesh.vertices
            faces    = mesh.faces
        n = self.vertices.shape[0]
        self.vertices = np.concatenate((self.vertices, vertices))
        self.faces = np.concatenate((self.faces, faces+n))
    def rotate(self, v, x, y, z):
        self.vertices = transform(rotate(v, x, y, z), self.vertices)
        if not self.bottom is None:
            self.bottom = transform(rotate(v, x, y, z), self.bottom)
        if not self.top is None:
            self.top = transform(rotate(v, x, y, z), self.top)
    def rotatex(self, v):
        self.rotate(v, 1, 0, 0)
    def rotatey(self, v):
        self.rotate(v, 0, 1, 0)
    def rotatez(self, v):
        self.rotate(v, 0, 0, 1)
    def translate(self, x, y=None, z=None):
        if y == None:
            if len(x.shape)>1:
                xyz = x.flatten()
            else:
                xyz = x
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
        self.vertices = transform(translate(x, y, z), self.vertices)
        if not self.bottom is None:
            self.bottom = transform(translate(x, y, z), self.bottom)
        if not self.top is None:
            self.top = transform(translate(x, y, z), self.top)
    def scale(self, x, y=None, z=None):
        if y == None:
            xyz = x
            x = xyz
            y = xyz
            z = xyz
        self.vertices = transform(scale(x, y, z), self.vertices)
        if not self.bottom is None:
            self.bottom = transform(scale(x, y, z), self.bottom)
        if not self.top is None:
            self.top = transform(scale(x, y, z), self.top)

class Cube(Mesh):
    def __init__(self):
        Mesh.__init__(self)
        self.top    = np.array([[0, 0,  1]])
        self.bottom = np.array([[0, 0, -1]])
        self.vertices = np.array([
            [ -1.000,  -1.000,  -1.000],
            [ -1.000,   1.000,  -1.000],
            [  1.000,   1.000,  -1.000],
            [  1.000,  -1.000,  -1.000],
            [ -1.000,  -1.000,   1.000],
            [ -1.000,   1.000,   1.000],
            [  1.000,   1.000,   1.000],
            [  1.000,  -1.000,   1.000]
            ])
        self.faces = np.array([
            [   0,    1,    3],
            [   1,    2,    3],
            [   0,    4,    5],
            [   0,    5,    1],
            [   1,    5,    6],
            [   1,    6,    2],
            [   2,    6,    7],
            [   2,    7,    3],
            [   3,    7,    4],
            [   3,    4,    0],
            [   4,    5,    7],
            [   5,    6,    7]
            ])

class Cylinder(Mesh):
    def __init__(self):
        Mesh.__init__(self)
        self.top    = np.array([[0, 0,  1]])
        self.bottom = np.array([[0, 0, -1]])
        self.vertices = np.array([
            [  1.000,   0.000,  -1.000],
            [  0.866,   0.500,  -1.000],
            [  0.500,   0.866,  -1.000],
            [  0.000,   1.000,  -1.000],
            [ -0.500,   0.866,  -1.000],
            [ -0.866,   0.500,  -1.000],
            [ -1.000,   0.000,  -1.000],
            [ -0.866,  -0.500,  -1.000],
            [ -0.500,  -0.866,  -1.000],
            [ -0.000,  -1.000,  -1.000],
            [  0.500,  -0.866,  -1.000],
            [  0.866,  -0.500,  -1.000],
            [  1.000,   0.000,   1.000],
            [  0.866,   0.500,   1.000],
            [  0.500,   0.866,   1.000],
            [  0.000,   1.000,   1.000],
            [ -0.500,   0.866,   1.000],
            [ -0.866,   0.500,   1.000],
            [ -1.000,   0.000,   1.000],
            [ -0.866,  -0.500,   1.000],
            [ -0.500,  -0.866,   1.000],
            [ -0.000,  -1.000,   1.000],
            [  0.500,  -0.866,   1.000],
            [  0.866,  -0.500,   1.000],
            [  0.000,   0.000,  -1.000],
            [  0.000,   0.000,   1.000]
            ])
        self.faces = np.array([
            [   0,    1,   13],
            [   0,   13,   12],
            [   1,    2,   14],
            [   1,   14,   13],
            [   2,    3,   15],
            [   2,   15,   14],
            [   3,    4,   16],
            [   3,   16,   15],
            [   4,    5,   17],
            [   4,   17,   16],
            [   5,    6,   18],
            [   5,   18,   17],
            [   6,    7,   19],
            [   6,   19,   18],
            [   7,    8,   20],
            [   7,   20,   19],
            [   8,    9,   21],
            [   8,   21,   20],
            [   9,   10,   22],
            [   9,   22,   21],
            [  10,   11,   23],
            [  10,   23,   22],
            [  11,    0,   12],
            [  11,   12,   23],
            [  24,    1,    0],
            [  24,    2,    1],
            [  24,    3,    2],
            [  24,    4,    3],
            [  24,    5,    4],
            [  24,    6,    5],
            [  24,    7,    6],
            [  24,    8,    7],
            [  24,    9,    8],
            [  24,   10,    9],
            [  24,   11,   10],
            [  24,    0,   11],
            [  25,   12,   13],
            [  25,   13,   14],
            [  25,   14,   15],
            [  25,   15,   16],
            [  25,   16,   17],
            [  25,   17,   18],
            [  25,   18,   19],
            [  25,   19,   20],
            [  25,   20,   21],
            [  25,   21,   22],
            [  25,   22,   23],
            [  25,   23,   12]
            ])

class Sphere(Mesh):
    def __init__(self):
        Mesh.__init__(self)
        self.vertices = np.array([
            [  0.000,   0.000,   1.000],
            [  0.894,   0.000,   0.447],
            [  0.276,   0.851,   0.447],
            [ -0.724,   0.526,   0.447],
            [ -0.724,  -0.526,   0.447],
            [  0.276,  -0.851,   0.447],
            [  0.724,  -0.526,  -0.447],
            [  0.724,   0.526,  -0.447],
            [ -0.276,   0.851,  -0.447],
            [ -0.894,   0.000,  -0.447],
            [ -0.276,  -0.851,  -0.447],
            [  0.000,   0.000,  -1.000]
            ])
        self.faces = np.array([
            [   0,    1,    2],
            [   0,    2,    3],
            [   0,    3,    4],
            [   0,    4,    5],
            [   0,    5,    1],
            [   1,    7,    2],
            [   2,    8,    3],
            [   3,    9,    4],
            [   4,   10,    5],
            [   5,    6,    1],
            [   6,    7,    1],
            [   7,    8,    2],
            [   8,    9,    3],
            [   9,   10,    4],
            [  10,    6,    5],
            [  11,    7,    6],
            [  11,    8,    7],
            [  11,    9,    8],
            [  11,   10,    9],
            [  11,    6,   10]
            ])

class bodySegment(Cylinder):
    def __init__(self, length, diameter, color='white'):
        Cylinder.__init__(self)
        self.translate(0, 0, 1)
        self.scale(diameter, diameter, length/2)
        self.color = color2rgba[color]

class imuSensor(Cube):
    def __init__(self, x, y, z, color='white'):
        Cube.__init__(self)
        self.color = color2rgba[color]
        self.scale(x/2, y/2, z/2)
        cylinder = Cylinder()
        cylinder.scale(0.5, 0.5, 0.5)
        cylinder.scale(0.5*y, 0.5*y, 0.2*x)
        cylinder.rotate(90, 0, 1, 0)
        cylinder.translate(x/2, 0, 0) # it is sticking out on the positive x-side
        self.append(cylinder.vertices, cylinder.faces)

def update():
    for item in w.items:
        w.removeItem(item)

    g = gl.GLGridItem()
    g.scale(30/GLOBALSCALE, 30/GLOBALSCALE, 30/GLOBALSCALE)
    g.translate(0, 0, -ZSHIFT/GLOBALSCALE)
    w.addItem(g)

    #md = gl.MeshData.sphere(10,10)
    #g = gl.GLMeshItem(meshdata=md, smooth=True, computeNormals=True, drawFaces=True, drawEdges=False, shader='shaded')
    #g.scale(10/GLOBALSCALE, 10/GLOBALSCALE, 10/GLOBALSCALE)
    #g.translate(0, 0, -ZSHIFT/GLOBALSCALE)
    #w.addItem(g)

    sensor = imuSensor(35, 20, 15.5)
    sensor.rotatex(patch.getfloat('sensor', 'x', default=0))
    sensor.rotatey(patch.getfloat('sensor', 'y', default=0))
    sensor.rotatez(-patch.getfloat('sensor', 'z', default=0))

    xaxis = bodySegment(30, 2, 'red')
    xaxis.rotate(90, 0, 1, 0)
    yaxis = bodySegment(30, 2, 'green')
    yaxis.rotate(-90, 1, 0, 0)
    zaxis = bodySegment(30, 2, 'blue')
    zaxis.rotate(0, 0, 0, 0)

    # See https://www.researchgate.net/profile/Ashish_Singla4/publication/283532449/figure/fig1/AS:315590863540224@1452254133566/Dimensions-of-average-male-human-being-23_W640.jpg
    body = bodySegment(48.8, 5, 'wheat')
    body.translate(0, 0, 9.4+45+46)

    neck = bodySegment(25, 5, 'wheat')
    neck.rotate(0, 0, 1, 0)
    neck.translate(body.top)

    head = bodySegment(10, 15, 'wheat')
    head.rotate(90, 0, 1, 0)
    head.translate(neck.top)

    leftShoulder = bodySegment(20, 5, 'wheat')
    leftShoulder.rotate(-90, 1, 0, 0)
    leftShoulder.translate(body.top)

    rightShoulder = bodySegment(20, 5, 'wheat')
    rightShoulder.rotate(90, 1, 0, 0)
    rightShoulder.translate(body.top)

    upperLeftArm = bodySegment(30.2, 5, 'red')
    upperLeftArm.rotate(-170, 1, 0, 0)
    upperLeftArm.rotatex(patch.getfloat('upperleftarm', 'x', default=0))
    upperLeftArm.rotatey(patch.getfloat('upperleftarm', 'y', default=0))
    upperLeftArm.rotatez(patch.getfloat('upperleftarm', 'z', default=0))
    upperLeftArm.translate(leftShoulder.top)

    upperRightArm = bodySegment(30.2, 5, 'green')
    upperRightArm.rotate(170, 1, 0, 0)
    upperRightArm.rotatex(patch.getfloat('upperrightarm', 'x', default=0))
    upperRightArm.rotatey(patch.getfloat('upperrightarm', 'y', default=0))
    upperRightArm.rotatez(patch.getfloat('upperrightarm', 'z', default=0))
    upperRightArm.translate(rightShoulder.top)

    lowerLeftArm = bodySegment(26.9, 5, 'red')
    lowerLeftArm.rotate(-180, 1, 0, 0)
    lowerLeftArm.rotatex(patch.getfloat('lowerleftarm', 'x', default=0))
    lowerLeftArm.rotatey(patch.getfloat('lowerleftarm', 'y', default=0))
    lowerLeftArm.rotatez(patch.getfloat('lowerleftarm', 'z', default=0))
    lowerLeftArm.translate(upperLeftArm.top)

    lowerRightArm = bodySegment(26.9, 5, 'green')
    lowerRightArm.rotate(180, 1, 0, 0)
    lowerRightArm.rotatex(patch.getfloat('lowerrightarm', 'x', default=0))
    lowerRightArm.rotatey(patch.getfloat('lowerrightarm', 'y', default=0))
    lowerRightArm.rotatez(patch.getfloat('lowerrightarm', 'z', default=0))
    lowerRightArm.translate(upperRightArm.top)

    leftHip = bodySegment(14/2, 5, 'wheat')
    leftHip.rotate(-90, 1, 0, 0)
    leftHip.translate(body.bottom)

    rightHip = bodySegment(14/2, 5, 'wheat')
    rightHip.rotate(90, 1, 0, 0)
    rightHip.translate(body.bottom)

    upperLeftLeg = bodySegment(46, 5, 'wheat')
    upperLeftLeg.rotate(-170, 1, 0, 0)
    upperLeftLeg.translate(leftHip.top)

    upperRightLeg = bodySegment(46, 5, 'wheat')
    upperRightLeg.rotate(170, 1, 0, 0)
    upperRightLeg.translate(rightHip.top)

    lowerLeftLeg = bodySegment(45, 5, 'wheat')
    lowerLeftLeg.rotate(-180, 1, 0, 0)
    lowerLeftLeg.translate(upperLeftLeg.top)

    lowerRightLeg = bodySegment(45, 5, 'wheat')
    lowerRightLeg.rotate(180, 1, 0, 0)
    lowerRightLeg.translate(upperRightLeg.top)

    pieces = []
    # pieces.append(sensor)

    pieces.append(xaxis)
    pieces.append(yaxis)
    pieces.append(zaxis)

    pieces.append(body)
    pieces.append(neck)
    pieces.append(head)
    pieces.append(leftShoulder)
    pieces.append(rightShoulder)
    pieces.append(upperLeftArm)
    pieces.append(upperRightArm)
    pieces.append(lowerLeftArm)
    pieces.append(lowerRightArm)
    pieces.append(leftHip)
    pieces.append(rightHip)
    pieces.append(upperLeftLeg)
    pieces.append(upperRightLeg)
    pieces.append(lowerLeftLeg)
    pieces.append(lowerRightLeg)

    for piece in pieces:
        md = gl.MeshData(vertexes=piece.vertices, faces=piece.faces)
        g = gl.GLMeshItem(meshdata=md, color=piece.color, smooth=True, computeNormals=True, drawFaces=True, drawEdges=False, shader='shaded')
        g.translate(0, 0, -ZSHIFT/GLOBALSCALE)
        g.scale(1/GLOBALSCALE,1/GLOBALSCALE,1/GLOBALSCALE)
        w.addItem(g)

# keyboard interrupt handling
def sigint_handler(*args):
    QtGui.QApplication.quit()

signal.signal(signal.SIGINT, sigint_handler)

if True:
    # Set timer for update
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.setInterval(10)            # timeout in milliseconds
    timer.start(int(delay*1000))     # in milliseconds
else:
    update()

# Start
QtGui.QApplication.instance().exec_()
