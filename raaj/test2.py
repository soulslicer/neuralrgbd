# Torch
import torch

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

# Python
import numpy as np
import math
import time
import os, sys
import cv2



import util
import matplotlib.pyplot as plt

def gaussian(x, mu, sig, power=2.):
    return np.exp(-np.power(np.abs(x - mu), power) / (2 * np.power(sig, power)))

def uniform(x):
    return np.array([1./len(x)]*len(x))

def normalize(x):
    return x/np.sum(x)

import cv2

# Callback function for trackbar
def on_change(self):
    pass

# Creates window
cv2.namedWindow('Image')
cv2.createTrackbar('A', 'Image', 0, 256, on_change)
cv2.setTrackbarPos('A', 'Image', 256)
cv2.createTrackbar('B', 'Image', 0, 256, on_change)
cv2.setTrackbarPos('B', 'Image', 256)
cv2.createTrackbar('STD', 'Image', 0, 128, on_change)
cv2.setTrackbarPos('STD', 'Image', 30)
cv2.createTrackbar('POWER', 'Image', 0, 128, on_change)
cv2.setTrackbarPos('POWER', 'Image', 64)


# Infinite loop
plt.ion()
while(True):
    A = (cv2.getTrackbarPos('A', 'Image') - 128)/128.  # returns trackbar position
    B = (cv2.getTrackbarPos('B', 'Image') - 128) / 128.  # returns trackbar position
    STD = (cv2.getTrackbarPos('STD', 'Image')) / 128.  # returns trackbar position
    POWER = (cv2.getTrackbarPos('POWER', 'Image')) / 32.  # returns trackbar position

    #A = A*5
    B = B*10
    ##??

    print(A)
    print(B)
    print(STD)
    print(POWER)
    k = cv2.waitKey(1)




    d_candi = util.powerf(5, 40, 96, 1.5)

    #dist = gaussian(d_candi, 20, 0.5)*0.2 + uniform(d_candi)*0.8
    dist = gaussian(d_candi, 20, STD, POWER)*A + uniform(d_candi)*B
    dist = np.clip(dist, 0, 1)
    dist = normalize(dist)

    axes = plt.axes()
    axes.set_ylim([0, 0.5])
    plt.plot(d_candi, dist)
    plt.pause(0.01)
    plt.cla()
