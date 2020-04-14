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

# Custom
import util
from util import Logger
import kitti
import batch_loader
import inverse_warp as iv
from model import KVNET
import losses
from light_curtain import LightCurtain

def normalize(field):
    minv, _ = field.min(1) # [1,384]
    maxv, _ = field.max(1)  # [1,384]
    return (field - minv)/(maxv-minv)


import matplotlib.pyplot as plt
# def plotfig(index):
#     dist_pred = dpv_plane_predicted[0,:,index].cpu().numpy()
#     dist_truth = dpv_plane_truth[0, :, index].cpu().numpy()
#     plt.figure()
#     plt.plot(np.array(d_candi), dist_pred)
#     plt.plot(np.array(d_candi), dist_truth)
#     plt.ion()
#     plt.pause(0.005)
#     plt.show()

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], color='g')
    ax.add_line(l)
    return l

data = np.load("testing.npy", allow_pickle=True).item()
dpv_plane_predicted = torch.tensor(data["dpv_plane_predicted"])
dpv_predicted = torch.tensor(data["dpv_predicted"])
depthmap_truth_np = data["depthmap_truth_np"]
rgbimg = torch.tensor(data["rgbimg"])
d_candi = data["d_candi"]
intr_up = torch.tensor(data["intr_up"])

lightcurtain = LightCurtain()
lightcurtain.init(data["lc"])

# GT DPV
dpv_truth = util.gen_soft_label_torch(d_candi, torch.tensor(depthmap_truth_np).cuda(), 0.2, zero_invalid=True).unsqueeze(0).cpu()

# Field
dpv_plane_predicted, debugmap = losses.gen_ufield(dpv_predicted.cuda(), d_candi, intr_up, None, img=None)
dpv_plane_predicted = dpv_plane_predicted.squeeze(0)

# Plan
lc_paths, field_visual = lightcurtain.plan(dpv_plane_predicted.cuda())
lc_paths = data["lc_paths"]

pixel = [107,140]
pixel = [235,144]

i=0
lc_outputs = []
for lc_path in lc_paths:
    output = lightcurtain.sense_high(depthmap_truth_np, lc_path)
    output[np.isnan(output[:, :, 0])] = 0
    intensity = output[pixel[1], pixel[0], 3]
    zval = output[pixel[1], pixel[0], 2]
    print((intensity, zval))
    #print(output[pixel[1], pixel[0], 3])
    cv2.imshow("int"+str(i), output[:,:,3]/255.)
    i+=1
    lc_outputs.append(output)

truth_depth = depthmap_truth_np[pixel[1], pixel[0]]
dpv_predicted = torch.exp(dpv_predicted)
dist_pred = dpv_predicted[0, :, pixel[1], pixel[0]]
dist_truth = dpv_truth[0, :, pixel[1], pixel[0]]
plt.plot(np.array(d_candi), dist_pred.numpy())
plt.plot(np.array(d_candi), dist_truth.numpy())
newline([truth_depth,0], [truth_depth,1])

# Create a function to generate the ground truth or some distribution but in a noisy way?
#

#cv2.imshow("field_visual", field_visual)
cv2.imshow("win", rgbimg.numpy())
cv2.imshow("depth", depthmap_truth_np/100.)
cv2.imshow("dpv_plane_predicted", normalize(dpv_plane_predicted.unsqueeze(0)).squeeze(0).cpu().numpy())
plt.ion()
plt.pause(0.1)
cv2.waitKey(0)