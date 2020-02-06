# Torch
import open3d as o3d
import torch
torch.backends.cudnn.benchmark=True
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Python
import numpy as np
import math
import time
import os,sys
import cv2

# Custom
import util
from util import Logger
import kitti
import batch_loader
import inverse_warp as iv

# Other
from tensorboardX import SummaryWriter

# PCL
#from viewer import viewer
from viewer.viewer import Visualizer

def hack(cloud):
    fcloud = np.zeros(cloud.shape).astype(np.float32)
    for i in range(0, cloud.shape[0]):
        fcloud[i] = cloud[i]
    return fcloud

def viz_debug(local_info_valid, visualizer, d_candi, d_candi_up):
    ref_dats_in = local_info_valid['ref_dats']
    src_dats_in = local_info_valid['src_dats']
    left_cam_intrin_in = local_info_valid['left_cam_intrins']
    right_cam_intrin_in = local_info_valid['left_cam_intrins']
    left_src_cam_poses_in = torch.cat(local_info_valid['left_src_cam_poses'], dim=0)
    right_src_cam_poses_in = torch.cat(local_info_valid['right_src_cam_poses'], dim=0)
    T_left2right = local_info_valid["T_left2right"]

    # [2,5,4,4]

    """
    Stereo Warp (WE ASSUME LEFT AND RIGHT INTRINSICS ARE THE SAME)
    """

    batch_num = 1
    src_frame = 3
    target_frame = src_frame  # FIXED

    target_rgb_img = src_dats_in[batch_num][target_frame]["left_camera"]["img"][0, :, :, :]
    target_rgb_img[0, :, :] = target_rgb_img[0, :, :] * kitti.__imagenet_stats["std"][0] + \
                              kitti.__imagenet_stats["mean"][0]
    target_rgb_img[1, :, :] = target_rgb_img[1, :, :] * kitti.__imagenet_stats["std"][1] + \
                              kitti.__imagenet_stats["mean"][1]
    target_rgb_img[2, :, :] = target_rgb_img[2, :, :] * kitti.__imagenet_stats["std"][2] + \
                              kitti.__imagenet_stats["mean"][2]
    target_rgb_img = torch.unsqueeze(target_rgb_img, 0)

    src_rgb_img = src_dats_in[batch_num][src_frame]["right_camera"]["img"][0, :, :, :]
    src_rgb_img[0, :, :] = src_rgb_img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
    src_rgb_img[1, :, :] = src_rgb_img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
    src_rgb_img[2, :, :] = src_rgb_img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
    src_rgb_img = torch.unsqueeze(src_rgb_img, 0)

    target_depth_map = src_dats_in[batch_num][target_frame]["left_camera"]["dmap_imgsize"]
    depth_mask = target_depth_map > 0.;
    #depth_mask = depth_mask.float()

    pose_target2src = T_left2right
    # pose_target2src = torch.inverse(pose_target2src)
    pose_target2src = torch.unsqueeze(pose_target2src, 0)

    intr = left_cam_intrin_in[batch_num]["intrinsic_M"] * 4;
    #intr[0,0] *= 2;
    intr[2, 2] = 1;
    intr = intr[0:3, 0:3]
    intr = torch.tensor(intr.astype(np.float32))
    intr = torch.unsqueeze(intr, 0)

    target_warped_img, valid_points = iv.inverse_warp(src_rgb_img, target_depth_map, pose_target2src, intr)

    full_mask = depth_mask & valid_points
    full_mask = full_mask.float()

    target_rgb_img = target_rgb_img * full_mask.float()

    # Visualize RGB Image
    src_rgb_img = cv2.cvtColor(src_rgb_img[0, :, :, :].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    target_rgb_img = cv2.cvtColor(target_rgb_img[0, :, :, :].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    target_warped_img = cv2.cvtColor(target_warped_img[0, :, :, :].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    #comb = target_rgb_img * 0.5 + target_warped_img * 0.5
    comb = np.power(target_rgb_img - target_warped_img, 2)

    #fimage = np.vstack((target_rgb_img, src_rgb_img))
    fimage = np.vstack((target_rgb_img, target_warped_img, comb))

    print(target_rgb_img.shape)
    print(intr)

    cv2.namedWindow("fimage")
    cv2.moveWindow("fimage", 2500, 50)
    cv2.imshow("fimage", fimage)
    cv2.waitKey(0)
    stop

    """
    Left Cam Warp
    
    img: the source image (where to sample pixels) -- [B, 3, H, W]
    depth: depth map of the target image -- [B, H, W]
    pose: 6DoF pose parameters from target to source -- [B, 6] [B, 4, 4]
    intrinsics: camera intrinsic matrix -- [B, 3, 3]
    """

    # batch_num = 1
    # src_frame = 3
    # target_frame = 4 # FIXED
    #
    # target_rgb_img = src_dats_in[batch_num][target_frame]["left_camera"]["img"][0, :, :, :]
    # target_rgb_img[0, :, :] = target_rgb_img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
    # target_rgb_img[1, :, :] = target_rgb_img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
    # target_rgb_img[2, :, :] = target_rgb_img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
    # target_rgb_img = torch.unsqueeze(target_rgb_img, 0)
    #
    # src_rgb_img = src_dats_in[batch_num][src_frame]["left_camera"]["img"][0, :, :, :]
    # src_rgb_img[0, :, :] = src_rgb_img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
    # src_rgb_img[1, :, :] = src_rgb_img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
    # src_rgb_img[2, :, :] = src_rgb_img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
    # src_rgb_img = torch.unsqueeze(src_rgb_img, 0)
    #
    # target_depth_map = src_dats_in[batch_num][target_frame]["left_camera"]["dmap_imgsize"]
    # depth_mask = target_depth_map > 0.;
    # depth_mask = depth_mask.float()
    #
    # pose_target2src = left_src_cam_poses_in[batch_num, src_frame, :, :]
    # #pose_target2src = torch.inverse(pose_target2src)
    # pose_target2src = torch.unsqueeze(pose_target2src, 0)
    #
    # intr = left_cam_intrin_in[batch_num]["intrinsic_M"] * 4;
    # #intr[0,0] *= 2;
    # intr[2, 2] = 1;
    # intr = intr[0:3, 0:3]
    # intr = torch.tensor(intr.astype(np.float32))
    # intr = torch.unsqueeze(intr, 0)
    #
    # target_warped_img, valid_points = iv.inverse_warp(src_rgb_img, target_depth_map, pose_target2src, intr)
    #
    # #target_rgb_img = target_rgb_img * depth_mask
    #
    # # Visualize RGB Image
    # target_rgb_img = cv2.cvtColor(target_rgb_img[0,:,:,:].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    # target_warped_img = cv2.cvtColor(target_warped_img[0,:,:,:].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    # comb = target_rgb_img*0.25 + target_warped_img*0.75
    # #comb = target_rgb_img - target_warped_img
    #
    # fimage = np.vstack((target_rgb_img, target_warped_img, comb))
    #
    # cv2.imshow("fimage", fimage)
    # cv2.waitKey(0)
    # stop

    """
    Left Viz Pt Cloud
    """

    # # (Debug Visualize - Left)
    # batch_num = 1
    # for idx, datum in enumerate(src_dats_in[batch_num]):
    #     datum = datum["left_camera"]
    #     print(datum["img_path"])
    #
    #     # Images
    #     rgb_img = datum["img"][0, :, :, :].numpy()
    #     rgb_img[0, :, :] = rgb_img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
    #     rgb_img[1, :, :] = rgb_img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
    #     rgb_img[2, :, :] = rgb_img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
    #     gray_img = datum["img_gray"][0, 0, :, :].numpy()
    #     depth_imgsize = datum["dmap_imgsize"]
    #     depth_mask = depth_imgsize > 0.;
    #     depth_mask = depth_mask.float()
    #     depth_digit = datum["dmap_imgsize_digit"]
    #     depth_digit_up = datum["dmap_up4_imgsize_digit"]
    #     transform = left_src_cam_poses_in[batch_num, idx, :, :]
    #
    #     # Low Rez Quantized
    #     dpv = util.digitized_to_dpv(depth_digit, len(d_candi))
    #     depthmap_quantized = util.dpv_to_depthmap(dpv, d_candi) * depth_mask
    #
    #     # High Rez Quantized
    #     # dpv = util.digitized_to_dpv(depth_digit_up, len(d_candi_up))
    #     # depthmap_quantized = util.dpv_to_depthmap(dpv, d_candi_up) * depth_mask
    #
    #     # Original Depth Map
    #     # depthmap_quantized = depth_imgsize
    #
    #     # Convert to Pts
    #     intr = left_cam_intrin_in[batch_num]["intrinsic_M"] * 4;
    #     intr[2, 2] = 1;
    #     pts = util.depth_to_pts(depthmap_quantized, intr)
    #     pts = pts.reshape((3, pts.shape[1] * pts.shape[2]))
    #     # pts_numpy = pts.numpy()
    #
    #     # Attempt to transform
    #     transform = torch.inverse(transform)
    #     pts = torch.cat([pts, torch.ones((1, pts.shape[1]))])
    #     pts = torch.matmul(transform, pts)
    #     pts_numpy = pts[0:3, :].numpy()
    #
    #     # Convert Color
    #     pts_color = rgb_img.reshape((3, rgb_img.shape[1] * rgb_img.shape[2])) * 255
    #     pts_normal = np.zeros((3, rgb_img.shape[1] * rgb_img.shape[2]))
    #
    #     # Visualize
    #     all_together = np.concatenate([pts_numpy, pts_color, pts_normal], 0).astype(np.float32).T
    #     all_together = hack(all_together)
    #     visualizer.addCloud(all_together, 2)
    #     visualizer.swapBuffer()
    #
    #     # Visualize RGB Image
    #     rgb_img = rgb_img.transpose(1, 2, 0)
    #     rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    #     cv2.imshow("win", rgb_img)
    #     cv2.waitKey(0)
    #
    #     # Issue, the d_candi range is 0 to 1


# print(str(iepoch) + " " + str(batch_idx) + " " + str(frame_count))
# I need the name of the dataset
# I need to know why its size 2

# print(len(local_info["ref_dats"])) # seems to be a function of the batch size??

def main():
    import argparse
    print('Parsing the arguments...')
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('--exp_name', required =True, type=str, help='The name of the experiment. Used to naming the folders')
    parser.add_argument('--nepoch', required = True, type=int, help='# of epochs to run')
    parser.add_argument('--pre_trained', action='store_true', default=False, help='If use the pre-trained model; (False)')
    # Logging #
    parser.add_argument('--TB_add_img_interv', type=int, default = 50, help='The inerval for log one training image')
    parser.add_argument('--pre_trained_model_path', type=str, default='.', help='The pre-trained model path for KV-net')
    # Model Saving #
    parser.add_argument('--save_model_interv', type=int, default= 5000, help='The interval of iters to save the model; default: 5000')
    # TensorBoard #
    parser.add_argument('--TB_fldr', type=str, default='runs', help='The tensorboard logging root folder; default: runs')
    # Training #
    parser.add_argument('--RNet', action = 'store_true', help='if use refinement net to improve the depth resolution', default=True)
    parser.add_argument('--weight_var', default=.001, type=float, help='weight for the variance loss, if we use L1 loss')
    parser.add_argument('--pose_noise_level', default=0, type=float, help='Noise level for pose. Used for training with pose noise')
    parser.add_argument('--frame_interv', default=5, type=int, help='frame interval')
    parser.add_argument('--LR', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--t_win', type=int, default = 2, help='The radius of the temporal window; default=2')
    parser.add_argument('--d_min', type=float, default=1, help='The minimal depth value; default=0')
    parser.add_argument('--d_max', type=float, default=60, help='The maximal depth value; default=15')
    parser.add_argument('--ndepth', type=int, default= 64, help='The # of candidate depth values; default= 128')
    parser.add_argument('--grad_clip', action='store_true', help='if clip the gradient')
    parser.add_argument('--grad_clip_max', type=float, default=2, help='the maximal norm of the gradient')
    parser.add_argument('--sigma_soft_max', type=float, default=10., help='sigma_soft_max, default = 500.')
    parser.add_argument('--feature_dim', type=int, default=64, help='The feature dimension for the feature extractor; default=64')
    parser.add_argument('--batch_size', type=int, default = 0, help='The batch size for training; default=0, means batch_size=nGPU')
    # Dataset #
    parser.add_argument('--dataset', type=str, default='scanNet', help='Dataset name: {scanNet, kitti,}') 
    parser.add_argument('--dataset_path', type=str, default='.', help='Path to the dataset') 
    parser.add_argument('--change_aspect_ratio', action='store_true', default=False, help='If we want to change the aspect ratio. This option is only useful for KITTI')
    parser.add_argument('--viz', action='store_true', help='viz')

    hack_num = 326

    # Arguments Parsing
    args = parser.parse_args()
    exp_name = args.exp_name
    saved_model_path = './outputs/saved_models/%s'%(exp_name)
    dataset_name = args.dataset
    batch_size = args.batch_size
    n_epoch = args.nepoch
    TB_add_img_interv = args.TB_add_img_interv
    pre_trained = args.pre_trained
    t_win_r = args.t_win
    nDepth = args.ndepth
    d_candi = np.linspace(args.d_min, args.d_max, nDepth)
    d_candi_up = np.linspace(args.d_min, args.d_max, nDepth*4)
    LR = args.LR
    sigma_soft_max = args.sigma_soft_max #10.#500.
    dnet_feature_dim = args.feature_dim
    frame_interv = args.frame_interv # should be multiple of 5 for scanNet dataset
    if_clip_gradient = args.grad_clip
    grad_clip_max = args.grad_clip_max
    d_candi_dmap_ref = d_candi
    nDepth_dmap_ref = nDepth
    viz = args.viz

    # Save Folder #
    util.m_makedir(saved_model_path)
    savemodel_interv = args.save_model_interv

    # Writer #
    log_dir = 'outputs/%s/%s'%(args.TB_fldr, exp_name)
    writer = SummaryWriter(log_dir = log_dir, comment='%s'%(exp_name))
    util.save_args(args, '%s/tr_paras.txt'%(log_dir)) # save the training parameters #
    logfile = os.path.join(log_dir,'log_'+str(time.time())+'.txt')
    stdout = Logger(logfile)
    sys.stdout = stdout

    if viz:
        visualizer = Visualizer("V")
        visualizer.start()
    else:
        visualizer = None

    # Dataset #
    dataset_path = args.dataset_path
    if dataset_name == "kitti":
        dataset_init = kitti.KITTI_dataset
        fun_get_paths = lambda traj_indx: kitti.get_paths(traj_indx,split_txt= './kitti_split/training.txt', mode='train', database_path_base = dataset_path)

        # Cropping
        if not args.change_aspect_ratio: # we will keep the aspect ratio and do cropping
            img_size = [768, 256]
            crop_w = 384
        else: # we will change the aspect ratio and NOT do cropping
            img_size = [384, 256]
            crop_w = None

        # Load Dataset
        n_scenes , _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)
        fldr_path, img_paths, dmap_paths, poses, intrin_path = fun_get_paths(0)
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path = intrin_path, img_size= img_size, digitize= True,
                               d_candi= d_candi_dmap_ref, resize_dmap=.25, crop_w = crop_w)

        # https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

    # Train
    for iepoch in range(n_epoch):
        BatchScheduler = batch_loader.Batch_Loader(
                batch_size = batch_size, fun_get_paths = fun_get_paths,
                dataset_traj = dataset, nTraj=len(traj_Indx), dataset_name = dataset_name, hack_num = hack_num)

        # * Idea, we can do the entire feature extractor in parallel, then we split it?

        for batch_idx in range(len(BatchScheduler)):
            for frame_count, ref_indx in enumerate( range(BatchScheduler.traj_len)):
                local_info = BatchScheduler.local_info_full()
                n_valid_batch= local_info['is_valid'].sum()

                if n_valid_batch > 0:
                    local_info_valid = batch_loader.get_valid_items(local_info)

                    #if viz:
                    viz_debug(local_info_valid, visualizer, d_candi, d_candi_up)


if __name__ == '__main__':
    main()
