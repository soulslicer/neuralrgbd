import numpy as np
import math
import PIL.Image as image
import warping.View as View
import warping.homography as warp_homo

import torch
import torch.nn.functional as F
import torchvision
import kitti
import cv2
import util
import inverse_warp as iv

def rgb_stereo_consistency_loss(src_rgb_img, target_rgb_img, target_depth_map, pose_target2src, intr, viz=False):
    # Warp
    target_warped_rgb_img, valid_points = iv.inverse_warp(src_rgb_img, target_depth_map, pose_target2src, intr)

    # Mask (Should just be doing top half and valid_points)
    tophalf = torch.ones(valid_points.shape).bool().cuda();
    tophalf[:, 0:tophalf.shape[1] / 3, :] = False
    full_mask = valid_points & tophalf
    full_mask = full_mask.float()
    target_rgb_img = target_rgb_img * full_mask
    target_warped_rgb_img = target_warped_rgb_img * full_mask

    # Integrate loss here too and visualize for changing pose
    diff_img = (target_rgb_img - target_warped_rgb_img).abs()  # [1, 3, 256, 384]
    ssim_map = (0.5 * (1 - ssim(target_rgb_img, target_warped_rgb_img))).clamp(0, 1)
    # diff_img = (0.15 * diff_img + 0.85 * ssim_map)
    photo_error = mean_on_mask(diff_img, full_mask)

    # Visualize
    if viz:
        img_color_target = util.torchrgb_to_cv2(target_rgb_img.squeeze(0))
        img_color_diff = util.torchrgb_to_cv2(diff_img.squeeze(0), False)
        img_depth_target = cv2.cvtColor(target_depth_map[0, :, :].detach().cpu().numpy() / 100., cv2.COLOR_GRAY2BGR)
        img_color_warped_target = util.torchrgb_to_cv2(target_warped_rgb_img.squeeze(0))
        diff = np.abs(img_color_warped_target - img_color_target)
        combined = np.hstack([img_color_target, img_color_warped_target, diff, img_depth_target])
        cv2.imshow("win", combined)
        cv2.waitKey(15)

    return photo_error

def depth_stereo_consistency_loss(src_depth_img, target_depth_img, src_depth_mask, target_depth_mask, pose_target2src,
                                  intr):
    # Transform (Below needed only if baseline z changes or big trans)
    src_depth_img_trans = iv.transform_dmap(src_depth_img[0, 0, :, :], torch.inverse(pose_target2src), intr[0, :, :])
    src_depth_img_trans = (src_depth_img_trans.unsqueeze(0) * src_depth_mask.float()).unsqueeze(0)
    target_warped_depth_img, valid_points = iv.inverse_warp(src_depth_img_trans, target_depth_img.squeeze(0),
                                                            pose_target2src, intr, 'nearest')

    # Mask (Should just be doing top half and valid_points)
    warp_mask = target_warped_depth_img > 0.
    tophalf = torch.ones(valid_points.shape).bool().cuda();
    tophalf[:, 0:tophalf.shape[1] / 3, :] = False
    full_mask = valid_points & tophalf & warp_mask  # We should not need target_depth_mask
    full_mask = full_mask.float()
    target_depth_img = target_depth_img * full_mask
    target_warped_depth_img = target_warped_depth_img * full_mask

    # Score
    target_depth_img = target_depth_img.clamp(min=1e-3)
    target_warped_depth_img = target_warped_depth_img.clamp(min=1e-3)
    diff_depth = ((target_depth_img - target_warped_depth_img).abs() /
                  (target_depth_img + target_warped_depth_img).abs()).clamp(0, 1)
    dc_loss = mean_on_mask(diff_depth, full_mask)
    # diff_depth = (target_depth_img - target_warped_depth_img).abs()
    # reconstruction_loss = mean_on_mask(diff_depth, full_mask)
    return dc_loss

def depth_consistency_loss(large_dm, small_dm):
    tophalf = torch.ones(small_dm.shape).bool().cuda();
    tophalf[:, 0:tophalf.shape[1] / 3, :] = False
    downscaled_dm = F.interpolate(large_dm.unsqueeze(0), size=[small_dm.shape[1], small_dm.shape[2]],
                                  mode='nearest').squeeze(0)
    small_dm = small_dm.clamp(min=1e-3)
    downscaled_dm = downscaled_dm.clamp(min=1e-3)
    diff_depth = ((downscaled_dm - small_dm).abs() /
                  (downscaled_dm + small_dm).abs()).clamp(0, 1)
    dc_loss = mean_on_mask(diff_depth, tophalf.float())
    return dc_loss

def edge_aware_smoothness_loss(pred_disp, img, max_scales):
    def gradient_x(img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy

    def get_edge_smoothness(img, pred):
        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x),
                                          1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y),
                                          1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    s = 0
    for scaled_disp in pred_disp:
        s += 1
        if s > max_scales:
            break

        b, _, h, w = scaled_disp.size()
        scaled_img = F.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp) * weight
        weight /= 4.0

    return loss

def create_gaussian_window(window_size, channel):
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = torch.matmul(_1D_window, _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(
        channel, 1, window_size, window_size).contiguous()
    return window

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
window_size = 5
gaussian_img_kernel = create_gaussian_window(window_size, 3).float().to(device)
def ssim(img1, img2):
    params = {'weight': gaussian_img_kernel,
              'groups': 3, 'padding': window_size//2}
    mu1 = F.conv2d(img1, **params)
    mu2 = F.conv2d(img2, **params)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, **params) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, **params) - mu2_sq
    sigma12 = F.conv2d(img1*img2, **params) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map

def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / mask.sum()
    return mean_value

def soft_cross_entropy_loss(soft_label, x, mask=None, BV_log=False, ):
    if BV_log:
        #x_softmax = torch.exp(x)
        log_x_softmax = x
    else:
        x_softmax = F.softmax(x, dim=1)
        log_x_softmax = torch.log(x_softmax)

    loss = -torch.sum(soft_label * log_x_softmax, 1)

    if mask is not None:
        loss = loss * mask
        nonzerocount = (mask == 1).sum()
        if nonzerocount == 0: return 0.
        loss = torch.sum(loss)/nonzerocount
    else:
        loss = torch.mean(loss)
    return loss

def convert_flowfield(flowfield):
    yv, xv = torch.meshgrid([torch.arange(0, flowfield.shape[1]).float().cuda(), torch.arange(0, flowfield.shape[2]).float().cuda()])
    ystep = 2. / float(flowfield.shape[1] - 1)
    xstep = 2. / float(flowfield.shape[2] - 1)
    flowfield[0, :, :, 0] = -1 + xv * xstep - flowfield[0, :, :, 0] * xstep
    flowfield[0, :, :, 1] = -1 + yv * ystep - flowfield[0, :, :, 1] * ystep
    return flowfield

def gen_ufield(dpv_predicted, d_candi, intr_up, visualizer, img):

    # Generate Shiftmap
    pshift = 5
    flowfield = torch.zeros((1, dpv_predicted.shape[2], dpv_predicted.shape[3], 2)).float().cuda()
    flowfield_inv = torch.zeros((1, dpv_predicted.shape[2], dpv_predicted.shape[3], 2)).float().cuda()
    flowfield[:, :, :, 1] = pshift
    flowfield_inv[:, :, :, 1] = -pshift
    convert_flowfield(flowfield)
    convert_flowfield(flowfield_inv)

    # Shift the DPV
    dpv_shifted = F.grid_sample(dpv_predicted, flowfield)
    depthmap_shifted = util.dpv_to_depthmap(dpv_shifted, d_candi, BV_log=True)
    depthmap_predicted = util.dpv_to_depthmap(dpv_predicted, d_candi, BV_log=True)

    # Get Mask for Pts within Y range
    maxd = 40. # Issue, currentnetwork puts maxrange wrong
    pts_shifted = util.depth_to_pts(depthmap_shifted, intr_up)
    #zero_mask = (~((pts_shifted[1,:,:] > 1.4) | (pts_shifted[1,:,:] < -1.0))).float()
    #zero_mask = (~((pts_shifted[1, :, :] > 1.3) | (pts_shifted[1, :, :] < 1.0))).float()
    zero_mask = (~((pts_shifted[1, :, :] > 1.0) | (pts_shifted[1, :, :] < 0.5) | (pts_shifted[2, :, :] > maxd-1))).float() # THEY ALL SEEM TO BE DIFF HEIGHT? (CHECK CALIB)
    depthmap_shifted = depthmap_shifted * zero_mask

    # Shift Mask
    zero_mask_predicted = F.grid_sample(zero_mask.unsqueeze(0).unsqueeze(0), flowfield_inv).squeeze(0).squeeze(0)
    depthmap_predicted = depthmap_predicted * zero_mask_predicted

    # DPV Zero out and collapse
    zero_mask_predicted = zero_mask_predicted.repeat([64, 1, 1], 0, 1).unsqueeze(0)
    dpv_plane = torch.sum(torch.exp(dpv_predicted) * zero_mask_predicted, axis = 2) # [1,64,384]
    dpv_plane = F.softmax(dpv_plane, dim=1)

    # Apply softmax here?

    # Make 0 to 1 for visualization
    minval, _ = dpv_plane.min(1) # [1,384]
    maxval, _ = dpv_plane.max(1)  # [1,384]
    #dpv_plane = (dpv_plane - minval) / (maxval - minval)
    viz = dpv_plane.squeeze(0).cpu().numpy()
    cv2.imshow("win", viz*20)

    # We should draw the ground truth profile on this field

    # We need to power unwarp it

    # Need to transform it


    #depthmap_predicted = util.dpv_to_depthmap(dpv_predicted, d_candi, BV_log=True)


    #

    # Cloud for distance
    dcloud = []
    for m in range(0, 40):
        dcloud.append([0, 2.5, m, 255, 0, 0, 0, 0, 0])
    dcloud = np.array(dcloud).astype(np.float32)
    #cv2.imshow("win", depthmap_predicted.squeeze(0).cpu().numpy()/100.)
    cloud = util.tocloud(depthmap_predicted.cpu(), img, intr_up, None, [255,255,255])
    visualizer.addCloud(cloud, 1)
    visualizer.addCloud(dcloud, 5)
    visualizer.swapBuffer()
    import time
    while 1:
        cv2.waitKey(15)
        time.sleep(0.1)

    #stop

    #cloud_orig = util.tocloud(depthmap_predicted_np, img, intr_up, None)


    #for d

    """
    1. Shift the DPV Pixels all up by some degree until it looks flat (Viz depthmap and ptcloud)
    2. Convert into an XYZ field, remove those that Y value is above a certain value (Road)
        Same for Sky to remove them
        Remove a good chunk of it, only keep the middle parts of car.
    3. You can then flatten it (Depends on 2.)
    4. At this point, see if those near edges have a bad distribution
        Try to get rid of those with extreme uncertainty?
    4. Power Rescale. We can take each distribution and rescale it
    """

    #stop

