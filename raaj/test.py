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

def update(dpv1, dpv2):
    dpvmul = dpv1 * dpv2
    dpvnorm = dpvmul/torch.sum(dpvmul, dim=0)
    return dpvnorm

def mapping(x):
    # https://www.desmos.com/calculator/htpohhqx1a

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

def mixed_model(d_candi, z_img, unc_img, A, B):
    mixed_dist = util.gen_soft_label_torch(d_candi, z_img, unc_img, zero_invalid=True, pow=2.)*A + util.gen_uniform(d_candi, z_img)*B
    mixed_dist = torch.clamp(mixed_dist, 0, np.inf)
    mixed_dist = mixed_dist / torch.sum(mixed_dist, dim=0)
    return mixed_dist

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

def newline(p1, p2, c='g'):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], color=c)
    ax.add_line(l)
    return l

data = np.load("testing.npy", allow_pickle=True).item()
dpv_plane_predicted = torch.tensor(data["dpv_plane_predicted"])
dpv_predicted = torch.tensor(data["dpv_predicted"])
depthmap_truth_np = data["depthmap_truth_np"]
rgbimg = torch.tensor(data["rgbimg"])
d_candi = data["d_candi"]
intr_up = torch.tensor(data["intr_up"])

data["lc"]["rTc"][0,3] = 0.0

lightcurtain = LightCurtain()
lightcurtain.init(data["lc"])

# GT DPV
#mask = (torch.tensor(depthmap_truth_np) > 0).float()
#dpv_truth = util.gen_soft_label_torch(d_candi, torch.tensor(depthmap_truth_np).cuda(), 0.5, zero_invalid=True).unsqueeze(0).cpu()

# Field
#dpv_plane_predicted, debugmap = losses.gen_ufield(dpv_predicted.cuda(), d_candi, intr_up, None, img=None)
#dpv_plane_predicted = dpv_plane_predicted.squeeze(0)

# Plan
lc_paths, field_visual = lightcurtain.plan(dpv_plane_predicted)
lc_paths = data["lc_paths"]

#pixel = [107,140]
#pixel = [235,144]
pixel = [336,171]
#pixel = [100,100]

# Mask
lc_mask = depthmap_truth_np > 0

i=0
lc_outputs = []
DPVs = []
for lc_path in lc_paths:
    # output, thickimg, DPV = lightcurtain.sense_high(depthmap_truth_np, lc_path)
    # output[np.isnan(output[:, :, 0])] = 0
    # truthz = depthmap_truth_np[pixel[1], pixel[0]]
    # intensity = output[pixel[1], pixel[0], 3]
    # thickness = thickimg[pixel[1], pixel[0]]
    # zval = output[pixel[1], pixel[0], 2]
    # print((truthz, intensity, zval, thickness))
    # #print(output[pixel[1], pixel[0], 3])
    # #cv2.imshow("int"+str(i), output[:,:,3]/255.)
    # i+=1
    # # CRITICAL
    # #output[:,:,2] *= lc_mask
    # #thickimg *= lc_mask
    # lc_outputs.append([output, thickimg])

    DPV, output = lightcurtain.sense_high(depthmap_truth_np, lc_path)
    DPVs.append(DPV.cpu())

# Convert the LC stuff into a DPV
# Multiply the DPV's and plot the results?

# # I need to convert the output into a DPV
# DPVs = []
# for lc_output, thickimg in lc_outputs:
#
#     # I have the Z val and the intensity
#     z_img = lc_output[:,:,2]
#     int_img = lc_output[:,:,3]/255.
#     unc_img = (thickimg/6.)**2
#     #unc_img = unc_img*0 + 0.1
#
#     # Generate?
#     A = mapping(int_img)
#     # Try fucking with 1 in the 1-A value
#     DPV = mixed_model(d_candi, torch.tensor(z_img), torch.tensor(unc_img), torch.tensor(A), torch.tensor(1.-A))
#     DPVs.append(DPV)

# Exp
dpv_predicted = torch.exp(dpv_predicted)[0]

# Fuse
dpv_fused = torch.exp(torch.log(dpv_predicted) + torch.log(DPVs[0]) + torch.log(DPVs[1]) + torch.log(DPVs[2]))
dpv_fused = dpv_fused/torch.sum(dpv_fused, dim=0)

#dpv_fused = dpv_predicted
#dpv_fused = update(dpv_fused, DPVs[0])
#dpv_fused = update(dpv_fused, DPVs[1])
#dpv_fused = update(dpv_fused, DPVs[2])

dpv_pred_depth = util.dpv_to_depthmap(dpv_predicted.unsqueeze(0), d_candi, BV_log=False).squeeze(0)
dpv_fused_depth = util.dpv_to_depthmap(dpv_fused.unsqueeze(0), d_candi, BV_log=False).squeeze(0)
# print(AA[pixel[1], pixel[0]])
# print(AA.shape)
# stop

# Full Error
full_error = torch.sum(np.abs(dpv_pred_depth*lc_mask - depthmap_truth_np))/torch.sum(torch.tensor(lc_mask))
print(full_error)
full_error = torch.sum(np.abs(dpv_fused_depth*lc_mask - depthmap_truth_np))/torch.sum(torch.tensor(lc_mask))
print(full_error)

# Visualize
axes = plt.axes()
axes.set_ylim([0, 1.])
truth_depth = depthmap_truth_np[pixel[1], pixel[0]]
pred_depth = dpv_pred_depth[pixel[1], pixel[0]]
fused_depth = dpv_fused_depth[pixel[1], pixel[0]]
pred_dist = dpv_predicted[:, pixel[1], pixel[0]]

newline([truth_depth,0], [truth_depth,1], 'g')
newline([fused_depth,0], [fused_depth,1], 'b')
newline([pred_depth,0], [pred_depth,1], 'r')
print("Pred Error: " + str(abs(truth_depth - pred_depth)))
print("Fused Error: " + str(abs(truth_depth - fused_depth)))

plt.plot(np.array(d_candi), pred_dist.numpy())
plt.plot(np.array(d_candi), DPVs[0][:, pixel[1], pixel[0]].numpy())
plt.plot(np.array(d_candi), DPVs[1][:, pixel[1], pixel[0]].numpy())
plt.plot(np.array(d_candi), DPVs[2][:, pixel[1], pixel[0]].numpy())
plt.plot(np.array(d_candi), dpv_fused[:, pixel[1], pixel[0]].numpy())

plt.show()

# # Make EXP
# dpv_predicted = torch.exp(dpv_predicted)
#
#
# # Dist Mult Test
# dpv_updated = update(dpv_predicted,dpv_truth)
#
# truth_depth = depthmap_truth_np[pixel[1], pixel[0]]
# #dpv_predicted = torch.exp(dpv_predicted)
# dist_pred = dpv_predicted[0, :, pixel[1], pixel[0]]
# dist_truth = dpv_truth[0, :, pixel[1], pixel[0]]
# dist_updated = dpv_updated[0, :, pixel[1], pixel[0]]
# plt.plot(np.array(d_candi), dist_pred.numpy())
# plt.plot(np.array(d_candi), dist_truth.numpy())
# plt.plot(np.array(d_candi), dist_updated.numpy())
# newline([truth_depth,0], [truth_depth,1])
# #plt.ion()
# #plt.pause(0.1)
# plt.show()

# Create a function to generate the ground truth or some distribution but in a noisy way?
#

#cv2.imshow("field_visual", field_visual)
# cv2.imshow("win", rgbimg.numpy())
# cv2.imshow("depth", depthmap_truth_np/100.)
# cv2.imshow("dpv_plane_predicted", normalize(dpv_plane_predicted.unsqueeze(0)).squeeze(0).cpu().numpy())
# cv2.waitKey(0)