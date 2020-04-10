# Python
import numpy as np
import time
import sys

import torch
import torch.nn.functional as F

sys.path.append("/home/raaj/lcsim/python/")
from sim import LCDevice
from planner import PlannerRT
import pylc_lib as pylc

import util

class FieldWarp:
    def __init__(self, intr_input, dist_input, size_input, intr_output, dist_output, size_output, output2input):
        # Assign
        self.intr_input = intr_input
        self.dist_input = dist_input
        self.size_input = size_input
        self.intr_output = intr_output
        self.dist_output = dist_output
        self.size_output = size_output
        self.output2input = output2input

        # Compute Scaled
        self.intr_input_scaled = util.intr_scale(self.intr_input, self.size_input, self.size_output)
        self.dist_input_scaled = self.dist_input
        self.size_input_scaled = self.size_output

        # Compute angles
        self.angles_input = pylc.generateCameraAngles(self.intr_input, self.dist_input, self.size_input[0],
                                                      self.size_input[1])
        self.angles_input_scaled = pylc.generateCameraAngles(self.intr_input_scaled, self.dist_input_scaled,
                                                             self.size_input_scaled[0], self.size_input_scaled[1])
        self.angles_output = pylc.generateCameraAngles(self.intr_output, self.dist_output, self.size_output[0],
                                                       self.size_output[1])

        # Previous Fields
        self.flowfields = dict()

    def warp(self, input, flowfield):
        gridfield = torch.zeros(flowfield.shape).to(input.device)
        yv, xv = torch.meshgrid([torch.arange(0, input.shape[0]).float().cuda(), torch.arange(0, input.shape[1]).float().cuda()])
        ystep = 2. / float(input.shape[0] - 1)
        xstep = 2. / float(input.shape[1] - 1)
        gridfield[0, :, :, 0] = -1 + xv * xstep - flowfield[0, :, :, 0] * xstep
        gridfield[0, :, :, 1] = -1 + yv * ystep - flowfield[0, :, :, 1] * ystep
        input = input.unsqueeze(0).unsqueeze(0)
        output = F.grid_sample(input, gridfield, mode='bilinear').squeeze(0).squeeze(0)
        return output

    def digitize_soft(self, input, array):
        position = np.digitize([input], array)[0] - 1
        lp = array[position]
        # if position == len(array) - 1 or position == -1:
        #     return position

        if position == len(array) - 1:
            return 100000000
        elif position == -1:
            return -100000000

        rp = array[position + 1]
        soft_position = position + 1. * (float(input - lp) / float(rp - lp))
        return soft_position

    def preprocess(self, field, candi_input, candi_output):
        assert field.shape[0] == len(candi_input)
        assert field.shape[1] == self.size_input[0]
        field = field.unsqueeze(0).unsqueeze(0)
        field = F.upsample(field, size=[len(candi_output), self.size_input_scaled[0]], scale_factor=None,
                           mode='bilinear').squeeze(0).squeeze(0)
        return field

    def _ztheta2zrange(self, field, angles, d_candi, r_candi):
        r_field = torch.zeros(field.shape).to(field.device)
        flowfield = torch.zeros((1, field.shape[0], field.shape[1], 2)).to(field.device)
        assert r_field.shape[1] == len(angles)
        assert r_field.shape[0] == len(d_candi)
        assert r_field.shape[0] == len(r_candi)

        for r in range(0, r_field.shape[0]):
            for c in range(0, r_field.shape[1]):
                # Extract Values
                rng = r_candi[r]
                theta = angles[c]

                # Compute XYZ
                yval = 0.
                xval = rng * np.sin(np.radians(theta))
                zval = rng * np.cos(np.radians(theta))
                pt = np.array([xval, yval, zval, 1.]).reshape((4, 1))

                zbin = self.digitize_soft(zval, d_candi)
                thetabin = self.digitize_soft(theta, angles)

                # Set
                # r_field[r,c] = field[zbin, thetabin]
                flowfield[0, r, c, 0] = c - thetabin
                flowfield[0, r, c, 1] = r - zbin

        r_field = self.warp(field, flowfield)

        return r_field, flowfield

    def _transformZTheta(self, field, angles_input, d_candi_input, angles_output, d_candi_output, output2input):
        assert field.shape[1] == len(angles_input)
        assert field.shape[1] == len(angles_output)
        assert field.shape[0] == len(d_candi_input)
        assert field.shape[0] == len(d_candi_output)
        flowfield = torch.zeros((1, field.shape[0], field.shape[1], 2)).to(field.device)

        for r in range(0, field.shape[0]):
            for c in range(0, field.shape[1]):
                # r controls the d_candi_lc
                # c controls the angle
                zval = d_candi_output[r]
                theta = angles_output[c]

                # Compute XYZ
                rng = np.sqrt((np.power(zval, 2)) / (1 - np.power(np.sin(np.radians(theta)), 2)))
                xval = rng * np.sin(np.radians(theta))
                yval = 0
                pt = np.array([xval, yval, zval, 1.]).reshape((4, 1))

                # Transform
                tpt = np.matmul(output2input, pt)[:, 0]

                # Compute RTheta
                rng = np.sqrt(tpt[0] * tpt[0] + tpt[1] * tpt[1] + tpt[2] * tpt[2])
                zval = tpt[2]
                xval = tpt[0]
                theta = np.degrees(np.arcsin(xval / rng))

                zbin = self.digitize_soft(zval, d_candi_input)
                thetabin = self.digitize_soft(theta, angles_input)

                flowfield[0, r, c, 0] = c - thetabin
                flowfield[0, r, c, 1] = r - zbin

        n_field = self.warp(field, flowfield)

        return n_field, flowfield

    def ztheta2zrange_input(self, field, d_candi, r_candi, name):
        if name in self.flowfields.keys():
            output = self.warp(field, self.flowfields[name])
            return output
        else:
            output, flowfield = self._ztheta2zrange(field, self.angles_input_scaled, d_candi, r_candi)
            self.flowfields[name] = flowfield
            return output

    def ztheta2zrange_output(self, field, d_candi, r_candi, name):
        if name in self.flowfields.keys():
            output = self.warp(field, self.flowfields[name])
            return output
        else:
            output, flowfield = self._ztheta2zrange(field, self.angles_output, d_candi, r_candi)
            self.flowfields[name] = flowfield
            return output

    def transformZTheta(self, field, d_candi_input, d_candi_output, name):
        if name in self.flowfields.keys():
            print("load " + name)
            output = self.warp(field, self.flowfields[name])
            return output
        else:
            output, flowfield = self._transformZTheta(field, self.angles_input_scaled, d_candi_input,
                                                      self.angles_output, d_candi_output, self.output2input)
            self.flowfields[name] = flowfield
            print("stored " + name)
            return output

import cv2

def normalize(field):
    minv, _ = field.min(1) # [1,384]
    maxv, _ = field.max(1)  # [1,384]
    return (field - minv)/(maxv-minv)

def create_mean_kernel(N):
    kernel = torch.Tensor(np.zeros((N,N)).astype(np.float32)).cuda()
    kernel[:,N/2] = 1/float(N)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    params = {'weight': kernel, 'padding': N//2}
    return params

def invert(field):
    efield = -((1/np.sqrt(0.5)*(field-0.5))**2) + 0.5
    return efield

class LightCurtain:
    def __init__(self):
        self.lightcurtain = None
        self.initialized = False

    # intr_rgb, dist_rgb, size_rgb, intr_lc, dist_lc, size_lc, lc2rgb, lc2laser, laser_fov

    def get_flat(self, z):
        points = []
        for x in np.arange(-10., 10., 0.01):
            points.append([x, z])
        return np.array(points).astype(np.float32)

    def init(self, PARAMS):
        self.PARAMS = PARAMS
        CAMERA_PARAMS_LARGE = {
            'width': PARAMS["size_lc"][0],
            'height': PARAMS["size_lc"][1],
            'matrix': PARAMS["intr_lc"],
            'distortion': PARAMS["dist_lc"],
            'hit_mode': 1,
            'hit_noise': 0.01
        }
        CAMERA_PARAMS_SMALL = {
            'width': PARAMS["size_lc"][0] / 4,
            'height': PARAMS["size_lc"][1] / 4,
            'matrix': util.intr_scale_unit(PARAMS["intr_lc"], 1 / 4.),
            'distortion': PARAMS["dist_lc"],
            'hit_mode': 1,
            'hit_noise': 0.01
        }
        LASER_PARAMS = {
           'lTc': PARAMS["lTc"],
           'fov': PARAMS["laser_fov"]
        }
        self.lightcurtain_large = LCDevice(CAMERA_PARAMS=CAMERA_PARAMS_LARGE, LASER_PARAMS=LASER_PARAMS)
        self.lightcurtain_small = LCDevice(CAMERA_PARAMS=CAMERA_PARAMS_SMALL, LASER_PARAMS=LASER_PARAMS)
        self.planner = PlannerRT(self.lightcurtain_large, PARAMS["r_candi_up"], PARAMS["size_lc"][0], debug=False)
        dist_rgb = np.array(PARAMS["dist_rgb"]).astype(np.float32).reshape((1, 5))
        dist_lc = np.array(PARAMS["dist_lc"]).astype(np.float32).reshape((1, 5))
        self.fw = FieldWarp(PARAMS["intr_rgb"], dist_rgb, PARAMS["size_rgb"],
                            PARAMS["intr_lc"], dist_lc, PARAMS["size_lc"],
                            PARAMS["rTc"])
        self.d_candi = PARAMS["d_candi"]
        self.r_candi = PARAMS["r_candi"]
        self.d_candi_up = PARAMS["d_candi_up"]
        self.r_candi_up = PARAMS["r_candi_up"]
        self.PARAMS['cTr'] = np.linalg.inv(PARAMS["rTc"])
        self.initialized = True

    def plan(self, field):
        #cv2.imshow("field", field.cpu().numpy())

        start = time.time()

        # Fix Weird Side bug
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

        # Preprocess to the right size
        field_preprocessed = self.fw.preprocess(field, self.d_candi, self.d_candi_up)

        # Apply smoothing Kernel
        mean_kernel = create_mean_kernel(5)
        field_preprocessed = F.conv2d(field_preprocessed.unsqueeze(0).unsqueeze(0), **mean_kernel).squeeze(0).squeeze(0)

        # Transform RGB to LC
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
            field_preprocessed = self.fw.transformZTheta(field_preprocessed, self.d_candi_up, self.d_candi_up, "transform")

        # Normalize 0 to 1
        field_preprocessed = normalize(field_preprocessed.unsqueeze(0)).squeeze(0)

        # Warp from Z to theta
        field_preprocessed_range = self.fw.ztheta2zrange_input(field_preprocessed, self.d_candi_up, self.r_candi_up, "z2rwarp")

        # Generate Peak Fields
        left_field = field_preprocessed_range.clone()
        right_field = field_preprocessed_range.clone()
        values, indices = torch.max(field_preprocessed_range, 0)
        # Extremely slow needs to be in CUDA
        for i in range(0, indices.shape[0]):
            maxind = indices[i]
            left_field[0:maxind, i] = 1.
            right_field[maxind:len(self.r_candi_up), i] = 1.

        # Invert the Fields
        left_field = invert(left_field)
        right_field = invert(right_field)

        # Plan
        pts_main = self.planner.get_design_points(field_preprocessed_range.cpu().numpy())
        pts_up = self.planner.get_design_points(left_field.cpu().numpy())
        pts_down = self.planner.get_design_points(right_field.cpu().numpy())

        print("Forward: " + str(time.time() - start))

        # Visual
        field_visual = np.repeat(field_preprocessed.cpu().numpy()[:, :, np.newaxis], 3, axis=2)
        pixels = np.array([np.digitize(pts_main[:,1], self.d_candi_up) - 1, range(0, pts_main.shape[0])]).T
        field_visual[pixels[:,0], pixels[:,1], :] = [1, 0, 0]
        pixels = np.array([np.digitize(pts_up[:, 1], self.d_candi_up) - 1, range(0, pts_up.shape[0])]).T
        field_visual[pixels[:, 0], pixels[:, 1], :] = [0, 1, 0]
        pixels = np.array([np.digitize(pts_down[:, 1], self.d_candi_up) - 1, range(0, pts_down.shape[0])]).T
        field_visual[pixels[:, 0], pixels[:, 1], :] = [0, 1, 0]

        return [pts_main, pts_up, pts_down], field_visual

    # def sense_low(self, np_depth_image, np_design_pts):
    #     output = self.lightcurtain_small.get_return(np_depth_image, np_design_pts)
    #
    #     print(output.shape) # 64,96,4

    #
    #     return output

    def sense_high(self, depth_rgb, design_pts_lc):
        start = time.time()

        # Warp depthmap to LC frame
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
        #if True:
            pts_rgb = util.depth_to_pts(torch.Tensor(depth_rgb).unsqueeze(0), self.PARAMS['intr_rgb'])
            pts_rgb = pts_rgb.reshape((pts_rgb.shape[0], pts_rgb.shape[1]*pts_rgb.shape[2]))
            pts_rgb = torch.cat([pts_rgb, torch.ones(1, pts_rgb.shape[1])])
            pts_rgb = pts_rgb.numpy().T
            depth_lc, _ = pylc.transformPoints(pts_rgb, self.PARAMS['intr_lc'], self.PARAMS['cTr'], self.PARAMS['size_lc'][0], self.PARAMS['size_lc'][1], {"filtering": 2})
        else:
            depth_lc = depth_rgb

        # Sense
        output_lc = self.lightcurtain_large.get_return(depth_lc, design_pts_lc)
        output_lc[np.isnan(output_lc[:,:,0])] = 0

        # Warp output to RGB frame
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
        #if True:
            pts = output_lc.reshape((output_lc.shape[0]*output_lc.shape[1], 4))
            depth_sensed, int_sensed = pylc.transformPoints(pts, self.PARAMS['intr_rgb'], self.PARAMS['rTc'], self.PARAMS['size_rgb'][0], self.PARAMS['size_rgb'][1], {"filtering": 0})
        else:
            int_sensed = output_lc[:,:,3]
            depth_sensed = output_lc[:, :,2]

        # Generate XYZ version for viz
        pts_sensed = util.depth_to_pts(torch.Tensor(depth_sensed).unsqueeze(0), self.PARAMS['intr_rgb'])
        output_rgb = np.zeros(output_lc.shape).astype(np.float32)
        output_rgb[:, :, 0] = pts_sensed[0, :, :]
        output_rgb[:, :, 1] = pts_sensed[1, :, :]
        output_rgb[:, :, 2] = pts_sensed[2, :, :]
        output_rgb[:, :, 3] = int_sensed

        print("Sense: " + str(time.time() - start))

        # Return
        return output_rgb

        # cv2.imshow("depth_rgb", depth_rgb / 100.)
        # cv2.imshow("depth_lc", depth_lc / 100.)
        # cv2.imshow("depth_sensed_orig", output_lc[:,:,2] / 100.)
        # cv2.imshow("int_sensed_orig", output_lc[:, :, 3] / 255.)
        # cv2.imshow("depth_sensed", depth_sensed/100.)
        # cv2.imshow("int_sensed", int_sensed/255.)
        # cv2.waitKey(0)


    # In sensing, we need to transform depth from RGB to LC
    # Then we sense
    # Then we [XYZI] in fact we have Z as depth and intensity map
    # Test: Convert ZI into XYZI, see if matches