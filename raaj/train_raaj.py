# Torch
import torch
torch.backends.cudnn.benchmark=True
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

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
from model import KVNET

# Other
from tensorboardX import SummaryWriter

# PCL
#from viewer import viewer

# Kill
import sys
import signal
exit = 0
def signal_handler(sig, frame):
    global exit
    exit = 1
signal.signal(signal.SIGINT, signal_handler)

def powerf(d_min, d_max, nDepth, power):
    f = lambda x: d_min + (d_max - 1) * x
    x = np.linspace(start=0, stop=1, num=nDepth)
    x = np.power(x, power)
    candi = [f(v) for v in x]
    return np.array(candi)

def hack(cloud):
    fcloud = np.zeros(cloud.shape).astype(np.float32)
    for i in range(0, cloud.shape[0]):
        fcloud[i] = cloud[i]
    return fcloud

def tocloud(depth, rgb, intr, extr):
    pts = util.depth_to_pts(depth, intr)
    pts = pts.reshape((3, pts.shape[1] * pts.shape[2]))
    # pts_numpy = pts.numpy()

    # Attempt to transform
    transform = torch.inverse(extr)
    pts = torch.cat([pts, torch.ones((1, pts.shape[1]))])
    pts = torch.matmul(transform, pts)
    pts_numpy = pts[0:3, :].numpy()

    # Convert Color
    pts_color = rgb.reshape((3, rgb.shape[1] * rgb.shape[2])) * 255
    pts_normal = np.zeros((3, rgb.shape[1] * rgb.shape[2]))

    # Visualize
    all_together = np.concatenate([pts_numpy, pts_color, pts_normal], 0).astype(np.float32).T
    all_together = hack(all_together)
    return all_together

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

    # batch_num = 1
    # src_frame = 3
    # target_frame = src_frame  # FIXED
    #
    # target_rgb_img = src_dats_in[batch_num][target_frame]["left_camera"]["img"][0, :, :, :]
    # target_rgb_img[0, :, :] = target_rgb_img[0, :, :] * kitti.__imagenet_stats["std"][0] + \
    #                           kitti.__imagenet_stats["mean"][0]
    # target_rgb_img[1, :, :] = target_rgb_img[1, :, :] * kitti.__imagenet_stats["std"][1] + \
    #                           kitti.__imagenet_stats["mean"][1]
    # target_rgb_img[2, :, :] = target_rgb_img[2, :, :] * kitti.__imagenet_stats["std"][2] + \
    #                           kitti.__imagenet_stats["mean"][2]
    # target_rgb_img = torch.unsqueeze(target_rgb_img, 0)
    #
    # src_rgb_img = src_dats_in[batch_num][src_frame]["right_camera"]["img"][0, :, :, :]
    # src_rgb_img[0, :, :] = src_rgb_img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
    # src_rgb_img[1, :, :] = src_rgb_img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
    # src_rgb_img[2, :, :] = src_rgb_img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
    # src_rgb_img = torch.unsqueeze(src_rgb_img, 0)
    #
    # target_depth_map = src_dats_in[batch_num][target_frame]["left_camera"]["dmap_imgsize"]
    # depth_mask = target_depth_map > 0.;
    # #depth_mask = depth_mask.float()
    #
    # pose_target2src = T_left2right
    # # pose_target2src = torch.inverse(pose_target2src)
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
    # full_mask = depth_mask & valid_points
    # full_mask = full_mask.float()
    #
    # target_rgb_img = target_rgb_img * full_mask.float()
    #
    # # Visualize RGB Image
    # src_rgb_img = cv2.cvtColor(src_rgb_img[0, :, :, :].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    # target_rgb_img = cv2.cvtColor(target_rgb_img[0, :, :, :].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    # target_warped_img = cv2.cvtColor(target_warped_img[0, :, :, :].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    # #comb = target_rgb_img * 0.5 + target_warped_img * 0.5
    # comb = np.power(target_rgb_img - target_warped_img, 2)
    #
    # #fimage = np.vstack((target_rgb_img, src_rgb_img))
    # fimage = np.vstack((target_rgb_img, target_warped_img, comb))
    #
    # print(target_rgb_img.shape)
    # print(intr)
    #
    # cv2.namedWindow("fimage")
    # cv2.moveWindow("fimage", 2500, 50)
    # cv2.imshow("fimage", fimage)
    # cv2.waitKey(0)
    # stop

    """
    Left Cam Warp
    
    img: the source image (where to sample pixels) -- [B, 3, H, W]
    depth: depth map of the target image -- [B, H, W]
    pose: 6DoF pose parameters from target to source -- [B, 6] [B, 4, 4]
    intrinsics: camera intrinsic matrix -- [B, 3, 3]
    """

    # batch_num = 1
    # src_frame = 1
    # target_frame = len(src_dats_in[batch_num])/2  # FIXED MIDDLE ONE ALWAYS
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
    #
    # pose_target2src = left_src_cam_poses_in[batch_num, src_frame, :, :]
    # #pose_target2src = torch.inverse(pose_target2src)
    # pose_target2src = torch.unsqueeze(pose_target2src, 0)
    #
    # intr = left_cam_intrin_in[batch_num]["intrinsic_M"] * 4;
    # intr[2, 2] = 1;
    # intr = intr[0:3, 0:3]
    # intr = torch.tensor(intr.astype(np.float32))
    # intr = torch.unsqueeze(intr, 0)
    #
    # target_warped_img, valid_points = iv.inverse_warp(src_rgb_img, target_depth_map, pose_target2src, intr)
    #
    # full_mask = depth_mask & valid_points
    # full_mask = full_mask.float()
    #
    # target_rgb_img = target_rgb_img * full_mask.float()
    #
    # #target_rgb_img = target_rgb_img * depth_mask
    #
    # # Visualize RGB Image
    # target_rgb_img = cv2.cvtColor(target_rgb_img[0,:,:,:].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    # target_warped_img = cv2.cvtColor(target_warped_img[0,:,:,:].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    # #comb = target_rgb_img*0.25 + target_warped_img*0.75
    # comb = np.power(target_rgb_img - target_warped_img, 2)
    #
    # fimage = np.vstack((target_rgb_img, target_warped_img, comb))
    # error = np.sum(np.sum(fimage))
    # print(error)
    #
    # cv2.namedWindow("fimage")
    # cv2.moveWindow("fimage", 2500, 50)
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
    #     rgb_lowres_img = datum["img_dw"][0, :, :, :].numpy()
    #     rgb_img[0, :, :] = rgb_img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
    #     rgb_img[1, :, :] = rgb_img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
    #     rgb_img[2, :, :] = rgb_img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
    #     rgb_lowres_img[0, :, :] = rgb_lowres_img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
    #     rgb_lowres_img[1, :, :] = rgb_lowres_img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
    #     rgb_lowres_img[2, :, :] = rgb_lowres_img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
    #     gray_img = datum["img_gray"][0, 0, :, :].numpy()
    #     depth_imgsize = datum["dmap_imgsize"]
    #     depth_mask = depth_imgsize > 0.;
    #     depth_mask = depth_mask.float()
    #     depth_digit = datum["dmap_imgsize_digit"]
    #     depth_digit_up = datum["dmap_up4_imgsize_digit"]
    #     depth_digit_lowres = datum["dmap"]
    #     transform = left_src_cam_poses_in[batch_num, idx, :, :]
    #
    #     # Low Res Depth Quantized
    #     dpv = util.digitized_to_dpv(depth_digit_lowres, len(d_candi))
    #     depthmap_lowres_quantized = util.dpv_to_depthmap(dpv, d_candi)
    #     #depthmap_lowres_quantized = datum["dmap_raw"]
    #
    #     # Low Depth Quantized
    #     dpv = util.digitized_to_dpv(depth_digit, len(d_candi))
    #     depthmap_quantized = util.dpv_to_depthmap(dpv, d_candi) * depth_mask
    #
    #     # High Depth Quantized
    #     dpv = util.digitized_to_dpv(depth_digit_up, len(d_candi_up))
    #     depthmap_up_quantized = util.dpv_to_depthmap(dpv, d_candi_up) * depth_mask
    #
    #     # Original Depth Map
    #     depthmap_orig = depth_imgsize
    #
    #     # Intr change
    #     intr = left_cam_intrin_in[batch_num]["intrinsic_M"] * 4;
    #     intr[2, 2] = 1;
    #
    #     # Cloud
    #     cloud_orig = tocloud(depthmap_orig, rgb_img, intr, transform)
    #     cloud_quantized = tocloud(depthmap_quantized, rgb_img, intr, transform)
    #     cloud_up_quantized = tocloud(depthmap_up_quantized, rgb_img, intr, transform)
    #     cloud_lowres_quantized = tocloud(depthmap_lowres_quantized, rgb_lowres_img, left_cam_intrin_in[batch_num]["intrinsic_M"], transform)
    #
    #     cloud_quantized[:,1] += 2.;
    #     cloud_up_quantized[:, 1] += 2.;
    #
    #     # Cloud for distance
    #     dcloud = []
    #     for m in range(0, 30):
    #         dcloud.append([0,2,m, 255,255,255, 0,0,0])
    #     dcloud = np.array(dcloud).astype(np.float32)
    #
    #     # IMPLEMENT SOFT TARGET ALGO
    #     # DO DURING TRAINING TIME BUT HOW
    #
    #     #visualizer.addCloud(cloud_orig, 2)
    #     visualizer.addCloud(cloud_up_quantized, 2)
    #     #visualizer.addCloud(cloud_lowres_quantized, 3)
    #     visualizer.addCloud(dcloud, 4)
    #     visualizer.swapBuffer()
    #
    #     # Visualize RGB Image
    #     rgb_img = rgb_img.transpose(1, 2, 0)
    #     rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    #     cv2.imshow("win", rgb_img)
    #     cv2.waitKey(0)


# A Simple Python program to demonstrate working
# of yield

# A generator function that yields 1 for the first time,
# 2 second time and 3 third time
# def simpleGeneratorFun():
#     yield 1
#     yield 2
#     yield 3
#
#
# # Driver code to check above generator function
# for value in simpleGeneratorFun():
#     print(value)
#
# stop

# Data Loading Module
import torch.multiprocessing
from torch.multiprocessing import Process, Queue, Value, cpu_count

class BatchSchedulerMP:
    def __init__(self, inputs, mode):
        self.mode = mode
        self.inputs = inputs
        self.queue = Queue()
        self.control = Value('i', 1)
        if self.mode == 0:
            self.process = Process(target=self.worker, args=(self.inputs, self.queue, self.control))
            self.process.start()
        #self.worker(self.inputs, self.queue, self.control)

    def stop(self):
        self.control.value = 0

    def get_mp(self):
        if self.mode == 0:
            while 1:
                items = self.queue.get()
                if items is None: break
                yield items
        else:
            for items in self.single(self.inputs):
                if items is None: break
                yield items

    def load(self, inputs):
        dataset_path = inputs["dataset_path"]
        t_win_r = inputs["t_win_r"]
        img_size = inputs["img_size"]
        crop_w = inputs["crop_w"]
        d_candi = inputs["d_candi"]
        d_candi_up = inputs["d_candi_up"]
        dataset_name = inputs["dataset_name"]
        hack_num = inputs["hack_num"]
        batch_size = inputs["batch_size"]
        qmax = inputs["qmax"]
        n_epoch = inputs["n_epoch"]

        dataset_init = kitti.KITTI_dataset
        fun_get_paths = lambda traj_indx: kitti.get_paths(traj_indx, split_txt='./kitti_split/training.txt',
                                                          mode='train',
                                                          database_path_base=dataset_path, t_win=t_win_r)

        # Load Dataset
        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)
        fldr_path, img_paths, dmap_paths, poses, intrin_path = fun_get_paths(0)
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path=intrin_path, img_size=img_size, digitize=True,
                               d_candi=d_candi, d_candi_up=d_candi_up, resize_dmap=.25,
                               crop_w=crop_w)
        BatchScheduler = batch_loader.Batch_Loader(
            batch_size=batch_size, fun_get_paths=fun_get_paths,
            dataset_traj=dataset, nTraj=len(traj_Indx), dataset_name=dataset_name, t_win_r=t_win_r,
            hack_num=hack_num)
        return BatchScheduler

    def worker(self, inputs, queue, control):
        qmax = inputs["qmax"]
        n_epoch = inputs["n_epoch"]

        # Iterate batch
        broken = False
        for iepoch in range(n_epoch):
            BatchScheduler = self.load(inputs)
            for batch_idx in range(len(BatchScheduler)):
                start = time.time()
                for frame_count, ref_indx in enumerate(range(BatchScheduler.traj_len)):
                    local_info = BatchScheduler.local_info_full()

                    # Queue Max
                    while queue.qsize() >= qmax:
                        if control.value == 0:
                            broken = True
                            break
                        time.sleep(0.01)
                    if control.value == 0:
                        broken = True
                        break

                    # Put in Q
                    queue.put([local_info, len(BatchScheduler), batch_idx, frame_count, ref_indx, iepoch])

                    # Update dat_array
                    if frame_count < BatchScheduler.traj_len - 1:
                        BatchScheduler.proceed_frame()

                    #print(batch_idx, frame_count)
                    if broken: break

                if broken: break
                BatchScheduler.proceed_batch()

            if broken: break
        queue.put(None)

    def single(self, inputs):
        qmax = inputs["qmax"]
        n_epoch = inputs["n_epoch"]

        # Iterate batch
        broken = False
        for iepoch in range(n_epoch):
            BatchScheduler = self.load(inputs)
            for batch_idx in range(len(BatchScheduler)):
                start = time.time()
                for frame_count, ref_indx in enumerate(range(BatchScheduler.traj_len)):
                    local_info = BatchScheduler.local_info_full()

                    # Put in Q
                    yield [local_info, len(BatchScheduler), batch_idx, frame_count, ref_indx, iepoch]

                    # Update dat_array
                    if frame_count < BatchScheduler.traj_len - 1:
                        BatchScheduler.proceed_frame()

                    #print(batch_idx, frame_count)
                    if broken: break

                if broken: break
                BatchScheduler.proceed_batch()

            if broken: break
        yield None

visualizer = None

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
    parser.add_argument('--qpower', type=float, default=1., help='How much exp quantization wanted')
    parser.add_argument('--ngpu', type=int, default=1., help='How many GPU')

    #hack_num = 326
    hack_num = 0
    #print("HACK!!")

    # ==================================================================================== #

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
    qpower = args.qpower
    ngpu = args.ngpu

    # Linear
    #d_candi = np.linspace(args.d_min, args.d_max, nDepth)
    #d_candi_up = np.linspace(args.d_min, args.d_max, nDepth*4)

    # Quad
    d_candi = powerf(args.d_min, args.d_max, nDepth, qpower)
    d_candi_up = powerf(args.d_min, args.d_max, nDepth*4, qpower)

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
        from viewer.viewer import Visualizer
        global visualizer
        visualizer = Visualizer("V")
        visualizer.start()
    else:
        visualizer = None

    # ==================================================================================== #

    # Dataset #
    dataset_path = args.dataset_path
    if dataset_name == "kitti":
        # Cropping
        if not args.change_aspect_ratio: # we will keep the aspect ratio and do cropping
            img_size = [768, 256]
            crop_w = 384
        else: # we will change the aspect ratio and NOT do cropping
            img_size = [384, 256]
            crop_w = None

        # https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

        inputs = {
            "dataset_path": dataset_path,
            "t_win_r": t_win_r,
            "img_size": img_size,
            "crop_w": crop_w,
            "d_candi": d_candi,
            "d_candi_up": d_candi_up,
            "dataset_name": "kitti",
            "hack_num": hack_num,
            "batch_size": batch_size,
            "n_epoch": n_epoch,
            "qmax": 1
        }
        b = BatchSchedulerMP(inputs, 0)

    if b.mode == 0:
        print("Preloading..")
        while b.queue.qsize() < b.inputs["qmax"]:
            time.sleep(0.1)

    # ==================================================================================== #

    # Model
    print('Init Network - Assume left usage')
    model_KVnet = KVNET(feature_dim = dnet_feature_dim, cam_intrinsics = None,
                        d_candi = d_candi, d_candi_up = d_candi_up, sigma_soft_max = sigma_soft_max, KVNet_feature_dim = dnet_feature_dim,
                        d_upsample_ratio_KV_net = None)

    #model_KVnet = torch.nn.DataParallel(model_KVnet,  dim=0)
    model_KVnet.cuda()
    model_KVnet.train()

    optimizer_KV = optim.Adam(model_KVnet.parameters(), lr = LR , betas= (.9, .999 ))

    model_path_KV = saved_model_path + "/" + args.pre_trained_model_path
    if model_path_KV is not '.' and pre_trained:
        print('loading KV_net at %s'%(model_path_KV))
        util.load_pretrained_model(model_KVnet, model_path_KV, optimizer_KV)

    print('Done')
    global exit

    # ==================================================================================== #

    # Vars
    LOSS = []
    total_iter = -1

    # Keep Getting
    start = time.time()
    for items in b.get_mp():
        end = time.time()
        #print("Data: " + str(end - start))
        # Exit
        if exit:
            print("Exiting")
            b.stop()
            sys.exit()

        # Get data
        local_info, batch_length, batch_idx, frame_count, ref_indx, iepoch = items
        if local_info is None:
            print("Ending")
            b.stop()
            time.sleep(2)
            del b
            break

        #import copy
        #local_info = copy.deepcopy(local_info)

        # Process
        n_valid_batch = local_info['is_valid'].sum()
        if n_valid_batch > 0:
            local_info_valid = batch_loader.get_valid_items(local_info)
            local_info_valid["d_candi"] = d_candi

            # Train
            output = train(model_KVnet, optimizer_KV, local_info_valid, ngpu, total_iter)
            loss_v = output["loss"]
            #loss_v = 0
        else:
            loss_v = LOSS[-1]
            pass

        # Add Iterations
        total_iter += 1

        # logging #
        if frame_count > 0:
            LOSS.append(loss_v)
            print('video batch %d / %d, iter: %d, frame_count: %d; Epoch: %d / %d, loss = %.5f'\
                  %(batch_idx + 1, batch_length, total_iter, frame_count, iepoch + 1, n_epoch, loss_v))
            writer.add_scalar('data/train_error', float(loss_v), total_iter)

        # Save
        if total_iter % savemodel_interv == 0 and total_iter != 0:
            print("Saving..")
            # if training, save the model #
            savefilename = saved_model_path + '/kvnet_checkpoint_iter_' + str(total_iter) + '.tar'
            torch.save({'iter': total_iter,
                        'frame_count': frame_count,
                        'ref_indx': ref_indx,
                        'traj_idx': batch_idx,
                        'state_dict': model_KVnet.state_dict(),
                        'optimizer': optimizer_KV.state_dict(),
                        'loss': loss_v}, savefilename)

        # Note time
        start = time.time()

        # # Iterate batch
        # for batch_idx in range(len(BatchScheduler)):
        #     start = time.time()
        #     for frame_count, ref_indx in enumerate( range(BatchScheduler.traj_len)):
        #         local_info = BatchScheduler.local_info_full()
        #         n_valid_batch = local_info['is_valid'].sum()
        #
        #         if n_valid_batch > 0:
        #             local_info_valid = batch_loader.get_valid_items(local_info)
        #             local_info_valid["d_candi"] = d_candi

                        # refine_costV - False	costv negative	64 Depth (TEST THESE)

        #
        #             # Put the data loader seperate?
        #             # It runs its epochs and its enumerator as per normal
        #             # Dumps it into a queue, that this then reads out
        #             # Flag to indicate ending?
        #             # Class function to call start and stop to reboot it - or just kill and respawn
        #             # Pass in Test or Train
        #
        #             # Generate a map of laser angle - laser thickness uncertainity
        #
        #             # Test case
        #
        #             # Noise to Pose?
        #
        #             # The log_softmax input should not be negative! (I CHANGED IT)
        #
        #             if viz:
        #                 viz_debug(local_info_valid, visualizer, d_candi, d_candi_up)
        #
        #             # Test
        #             # We must do it so that it
        #
        #             # Add KNet
        #
        #             # Test with feedback
        #
        #             # NAN at
        #             # video batch 4 / 18, iter: 450, frame_count: 237; Epoch: 1 / 20, loss = 126.69191
        #             # video batch 4 / 18, iter: 451, frame_count: 238; Epoch: 1 / 20, loss = nan
        #
        #             end = time.time()
        #             print(end - start)
        #             start = time.time()
        #
        #
        #             # Train
        #             output = train(model_KVnet, optimizer_KV, local_info_valid, ngpu, total_iter)
        #             loss_v = output["loss"]
        #
        #         else:
        #             loss_v = LOSS[-1]
        #
        #
        #         # Update dat_array
        #         if frame_count < BatchScheduler.traj_len-1:
        #             BatchScheduler.proceed_frame()
        #
        #         # Add Iterations
        #         total_iter += 1
        #
        #         # logging #
        #         if frame_count > 0:
        #             LOSS.append(loss_v)
        #             print('video batch %d / %d, iter: %d, frame_count: %d; Epoch: %d / %d, loss = %.5f'\
        #                   %(batch_idx + 1, len(BatchScheduler), total_iter, frame_count, iepoch + 1, n_epoch, loss_v))
        #             writer.add_scalar('data/train_error', float(loss_v), total_iter)
        #
        #         # Save
        #         if total_iter % savemodel_interv == 0:
        #             # if training, save the model #
        #             savefilename = saved_model_path + '/kvnet_checkpoint_iter_' + str(total_iter) + '.tar'
        #             torch.save({'iter': total_iter,
        #                         'frame_count': frame_count,
        #                         'ref_indx': ref_indx,
        #                         'traj_idx': batch_idx,
        #                         'state_dict': model_KVnet.state_dict(),
        #                         'optimizer': optimizer_KV.state_dict(),
        #                         'loss': loss_v}, savefilename)
        #
        #     BatchScheduler.proceed_batch()

def train(model, optimizer_KV, local_info_valid, ngpu, total_iter):
    start = time.time()

    # Ensure same size
    valid = (len(local_info_valid["left_cam_intrins"]) == len(local_info_valid["left_src_cam_poses"]) == len(local_info_valid["src_dats"]))
    if not valid:
        raise Exception('Batch size invalid')

    # Keep to middle only
    midval = int(len(local_info_valid["src_dats"][0])/2)

    # Grab ground truth digitized map
    dmap_imgsize_digit_arr = []
    dmap_digit_arr = []
    for i in range(0, len(local_info_valid["src_dats"])):
        dmap_imgsize_digit = local_info_valid["src_dats"][i][midval]["left_camera"]["dmap_imgsize_digit"]
        dmap_imgsize_digit_arr.append(dmap_imgsize_digit)
        dmap_digit = local_info_valid["src_dats"][i][midval]["left_camera"]["dmap"]
        dmap_digit_arr.append(dmap_digit)
    dmap_imgsize_digits = torch.cat(dmap_imgsize_digit_arr).cuda() # [B,256,384] uint64
    dmap_digits = torch.cat(dmap_digit_arr).cuda() # [B,64,96] uint64

    intrinsics_arr = []
    intrinsics_up_arr = []
    unit_ray_arr = []
    for i in range(0, len(local_info_valid["left_cam_intrins"])):
        intr = local_info_valid["left_cam_intrins"][i]["intrinsic_M_cuda"]
        intr_up = intr*4; intr_up[2,2] = 1;
        intrinsics_arr.append(intr.unsqueeze(0))
        intrinsics_up_arr.append(intr_up.unsqueeze(0))
        unit_ray_arr.append(local_info_valid["left_cam_intrins"][i]["unit_ray_array_2D"].unsqueeze(0))
    intrinsics = torch.cat(intrinsics_arr)
    intrinsics_up = torch.cat(intrinsics_up_arr)
    unit_ray = torch.cat(unit_ray_arr)

    src_cam_poses_arr = []
    for i in range(0, len(local_info_valid["left_src_cam_poses"])):
        pose = local_info_valid["left_src_cam_poses"][i]
        src_cam_poses_arr.append(pose[:,0:midval+1,:,:]) # currently [1x3x4x4]
    src_cam_poses = torch.cat(src_cam_poses_arr)

    rgb_arr = []
    debug_path = []
    for i in range(0, len(local_info_valid["src_dats"])):
        rgb_set = []
        debug_path_int = []
        for j in range(0, len(local_info_valid["src_dats"][i])):
            rgb_set.append(local_info_valid["src_dats"][i][j]["left_camera"]["img"])
            debug_path_int.append(local_info_valid["src_dats"][i][j]["left_camera"]["img_path"])
            if j == midval: break
        rgb_arr.append(torch.cat(rgb_set).unsqueeze(0))
        debug_path.append(debug_path_int)
    rgb = torch.cat(rgb_arr)

    # 10ms


    model_input = {
        "intrinsics": intrinsics,
        "unit_ray": unit_ray,
        "src_cam_poses": src_cam_poses,
        "rgb": rgb,
        "bv_predict": None # Has to be [B, 64, H, W]
    }
    BV_cur, BV_cur_refined = torch.nn.parallel.data_parallel(model, model_input, range(ngpu))
    # [B,128,64,96] [B,128,256,384]

    # NLL Loss
    loss = 0
    for ibatch in range(BV_cur.shape[0]):
        #loss = loss + torch.sum(BV_cur[ibatch,:,:,:])
        loss = loss + F.nll_loss(BV_cur[ibatch,:,:,:].unsqueeze(0), dmap_digits[ibatch,:,:].unsqueeze(0), ignore_index=0)
        loss = loss + F.nll_loss(BV_cur_refined[ibatch,:,:,:].unsqueeze(0), dmap_imgsize_digits[ibatch,:,:].unsqueeze(0), ignore_index=0)

    # What if we convert the DPV to a depth map, and regress that too?

    # Apply the Binary CE Loss thing in Gengshan work that function

    # Demonstrate light curtain on KITTI dataset

    # Backward
    bsize = BV_cur.shape[0]
    optimizer_KV.zero_grad()
    loss = loss / torch.tensor(float(bsize)).cuda(loss.get_device()) # SHOULD BE DIVIDED BY BATCH SIZE!
    loss.backward()
    optimizer_KV.step()

    # Debug Viz (comment if needed)
    if total_iter % 10 == -1:
        bnum = 0
        img = rgb[bnum,-1,:,:,:] # [1,3,256,384]
        img[0, :, :] = img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
        img[1, :, :] = img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
        img[2, :, :] = img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
        img = cv2.cvtColor(img[:, :, :].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
        img = img[:,:,0]
        ###
        dmap_up_digit = dmap_imgsize_digits[bnum,:,:].unsqueeze(0) # [1,256,384] uint64
        d_candi = local_info_valid["d_candi"]
        dpv = util.digitized_to_dpv(dmap_up_digit, len(d_candi))
        depthmap_quantized = util.dpv_to_depthmap(dpv, d_candi).squeeze(0).numpy() # [1,256,384]
        depthmap_quantized = depthmap_quantized/100.
        ###
        dpv_predicted = BV_cur_refined[bnum,:,:,:].unsqueeze(0).detach().cpu()
        depthmap_quantized_predicted = util.dpv_to_depthmap(dpv_predicted, d_candi, BV_log=True).squeeze(0).numpy()  # [1,256,384]
        depthmap_quantized_predicted = depthmap_quantized_predicted / 100.
        ###
        combined = np.hstack([img, depthmap_quantized, depthmap_quantized_predicted])
        cv2.namedWindow("win")
        cv2.moveWindow("win", 2500, 50)
        cv2.imshow("win", combined)
        cv2.waitKey(15)
    else:
        cv2.waitKey(15)

    # Return
    return {"loss": loss.detach().cpu().numpy()}

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        #global visualizer
        if visualizer is not None: visualizer.kill_received = True
