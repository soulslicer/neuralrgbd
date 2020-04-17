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

# TEST FOR 0 OR NAN CASE

# VARIANCE
unc_img = np.array([
    [(3./6.)**2, 0.3],
    [0.3, 0]
]).astype(np.float32)
z_img = np.array([
    [20, 20],
    [20, 0]
]).astype(np.float32)
int_img = np.array([
    [(50./255.), 1.],
    [1., 0]
]).astype(np.float32)
mask = z_img > 0


def mapping(x):
    def ma(x, m):
        A = -1. / ((m) * ((0.5 / m) + x)) + 1.
        return A

    def mb(x, m, f):
        c = m / ((m * f + 0.5) ** 2)
        y = c * x + (1 - c)
        return y

    m=5
    f=0.45
    mask = x > f
    y = ~mask*ma(x, m=m) + mask*mb(x,m=m,f=f)
    return y

# axes = plt.axes()
# axes.set_ylim([-1, 1])
# x = np.linspace(0., 1., 100)
# y3 = mapping(x)
# plt.plot(x,y3)
# plt.show()

def mixed_model(d_candi, z_img, unc_img, A, B):
    mixed_dist = util.gen_soft_label_torch(d_candi, z_img, unc_img, zero_invalid=True, pow=2.)*A + util.gen_uniform(d_candi, z_img)*B
    mixed_dist = torch.clamp(mixed_dist, 0, np.inf)
    mixed_dist = mixed_dist / torch.sum(mixed_dist, dim=0)
    return mixed_dist

# Need a way to from intensity to A value


d_candi = util.powerf(5, 40, 96, 1.5)
# x = util.gen_soft_label_torch(d_candi, torch.tensor(z_img), torch.tensor(unc_img), zero_invalid=True)
# y = util.gen_uniform(d_candi, torch.tensor(z_img))
A = mapping(int_img)
print(A)
z = mixed_model(d_candi, torch.tensor(z_img), torch.tensor(unc_img), torch.tensor(A), torch.tensor(1.-A))
#gaussian(d_candi, z_img, unc_img, 2.)

axes = plt.axes()
axes.set_ylim([0, 0.5])
dist = z[:,1,1]
#print(dist)
plt.plot(d_candi, dist.numpy())
plt.show()


# Creates window
cv2.namedWindow('Image')
cv2.createTrackbar('A', 'Image', 0, 256, on_change)
cv2.setTrackbarPos('A', 'Image', 256)
cv2.createTrackbar('B', 'Image', 0, 256, on_change)
cv2.setTrackbarPos('B', 'Image', 256)
cv2.createTrackbar('STD', 'Image', 0, 128, on_change)
cv2.setTrackbarPos('STD', 'Image', 60)
cv2.createTrackbar('POWER', 'Image', 0, 128, on_change)
cv2.setTrackbarPos('POWER', 'Image', 64)

# MAKE WORK FOR REAL IMAGE?

# Infinite loop
plt.ion()
while(True):
    # A = (cv2.getTrackbarPos('A', 'Image') - 128)/128.  # returns trackbar position
    # B = (cv2.getTrackbarPos('B', 'Image') - 128) / 128.  # returns trackbar position
    STD = (cv2.getTrackbarPos('STD', 'Image')) / 128.  # returns trackbar position
    POWER = (cv2.getTrackbarPos('POWER', 'Image')) / 32.  # returns trackbar position

    TRUE_A = cv2.getTrackbarPos('A', 'Image')
    A = mapping(np.array([float(TRUE_A)/256.]))
    print((TRUE_A, float(TRUE_A)/256., A))

    # k = cv2.waitKey(1)
    # continue

    #A = A*5
    #B = B*1
    ##??

    # print(A)
    # print(B)
    # print(STD)
    # print(POWER)
    k = cv2.waitKey(1)


    print((A, 1-A))

    d_candi = util.powerf(5, 40, 128, 1.5)

    #dist = gaussian(d_candi, 20, 0.5)*0.2 + uniform(d_candi)*0.8
    dist = gaussian(d_candi, 10, STD, POWER)*A + uniform(d_candi)*(1-A)
    dist = np.clip(dist, 0, np.inf)
    dist = normalize(dist)

    axes = plt.axes()
    axes.set_ylim([0, 0.5])
    plt.plot(d_candi, dist)
    plt.pause(0.01)
    plt.cla()
