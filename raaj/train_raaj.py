# Torch
import torch

torch.backends.cudnn.benchmark = True
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

# Other
from tensorboardX import SummaryWriter

# Kill
import sys
import signal

# Data Loading Module
import torch.multiprocessing
from torch.multiprocessing import Process, Queue, Value, cpu_count
from batch_scheduler import *

# Exit
exit = 0
def signal_handler(sig, frame):
    global exit
    exit = 1
signal.signal(signal.SIGINT, signal_handler)

# Global Vars
visualizer = None

def main():
    import argparse
    print('Parsing the arguments...')
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('--exp_name', required=True, type=str,
                        help='The name of the experiment. Used to naming the folders')
    parser.add_argument('--nepoch', type=int, help='# of epochs to run')
    parser.add_argument('--pre_trained', action='store_true', default=False,
                        help='If use the pre-trained model; (False)')
    # Logging #
    parser.add_argument('--TB_add_img_interv', type=int, default=50, help='The inerval for log one training image')
    parser.add_argument('--pre_trained_model_path', type=str, default='.', help='The pre-trained model path for KV-net')
    # Model Saving #
    parser.add_argument('--save_model_interv', type=int, default=5000,
                        help='The interval of iters to save the model; default: 5000')
    # TensorBoard #
    parser.add_argument('--TB_fldr', type=str, default='runs',
                        help='The tensorboard logging root folder; default: runs')
    # Training #
    parser.add_argument('--RNet', action='store_true', help='if use refinement net to improve the depth resolution',
                        default=True)
    parser.add_argument('--weight_var', default=.001, type=float,
                        help='weight for the variance loss, if we use L1 loss')
    parser.add_argument('--pose_noise_level', default=0, type=float,
                        help='Noise level for pose. Used for training with pose noise')
    parser.add_argument('--frame_interv', default=5, type=int, help='frame interval')
    parser.add_argument('--LR', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--t_win', type=int, default=2, help='The radius of the temporal window; default=2')
    parser.add_argument('--d_min', type=float, default=1, help='The minimal depth value; default=0')
    parser.add_argument('--d_max', type=float, default=60, help='The maximal depth value; default=15')
    parser.add_argument('--ndepth', type=int, default=64, help='The # of candidate depth values; default= 128')
    parser.add_argument('--grad_clip', action='store_true', help='if clip the gradient')
    parser.add_argument('--grad_clip_max', type=float, default=2, help='the maximal norm of the gradient')
    parser.add_argument('--sigma_soft_max', type=float, default=10., help='sigma_soft_max, default = 500.')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='The feature dimension for the feature extractor; default=64')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The batch size for training; default=1, means batch_size=nGPU')
    # Dataset #
    parser.add_argument('--dataset', type=str, default='scanNet', help='Dataset name: {scanNet, kitti,}')
    parser.add_argument('--dataset_path', type=str, default='.', help='Path to the dataset')
    parser.add_argument('--change_aspect_ratio', action='store_true', default=False,
                        help='If we want to change the aspect ratio. This option is only useful for KITTI')
    parser.add_argument('--viz', action='store_true', help='viz')
    parser.add_argument('--qpower', type=float, default=1., help='How much exp quantization wanted')
    parser.add_argument('--ngpu', type=int, default=1., help='How many GPU')
    parser.add_argument('--test_interval', type=int, default=1000, help='Test Interval')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Testing (False)')
    parser.add_argument('--pre_trained_folder', type=str, default='', help='The pre-trained folder to evaluate')
    parser.add_argument('--velodyne_depth', action='store_true', default=False,
                        help='velodyne_depth (False)')
    parser.add_argument('--lc', action='store_true', default=False,
                        help='velodyne_depth (False)')
    parser.add_argument('--drefine', type=str, default='', help='Additional flags for dnet')
    parser.add_argument('--pytorch_scaling', action='store_true', default=False,
                        help='pytorch scaling for data (False)')
    parser.add_argument('--softce', type=float, default=0., help='If soft cross entropy is wanted')
    parser.add_argument('--dsc_mul', type=float, default=0., help='Depth Stereo Consistency Loss Multiplier')
    parser.add_argument('--dc_mul', type=float, default=0., help='Depth Consistency Loss Multiplier')
    parser.add_argument('--rsc_mul', type=float, default=0., help='RGB Stereo Consistency Loss Multiplier')
    parser.add_argument('--rsc_low_mul', type=float, default=0., help='RGB Stereo Low Consistency Loss Multiplier')
    parser.add_argument('--smooth_mul', type=float, default=0., help='Smoothness Loss Multiplier')
    parser.add_argument('--pnoise', type=float, default=0., help='Noise added to pose')
    parser.add_argument('--nmode', type=str, default='default', help='Model Network Mode')
    parser.add_argument('--run_eval', action='store_true', default=False)
    parser.add_argument('--run_model', action='store_true', default=False)
    parser.add_argument('--load_args', action='store_true', default=False)
    parser.add_argument('--halflr', type=str, default="")
    parser.add_argument('--flow_rgb_mul', type=float, default=1., help='')
    parser.add_argument('--flow_depth_mul', type=float, default=1., help='')

    # hack_num = 326
    hack_num = 0

    # ==================================================================================== #

    # Arguments Parsing
    args = parser.parse_args()
    saved_model_path = "./outputs/saved_models/" + args.exp_name + "/"
    args_path = saved_model_path + "args.json"
    if args.load_args:
        args_old, leftover = parser.parse_known_args()
        args = util.load_argparse(args_path)
        # Add more variables later
        args.viz = args_old.viz
        args.ngpu = int(args_old.ngpu)
        args.batch_size = int(args_old.batch_size)
        args.run_eval = args_old.run_eval
        args.run_model = args_old.run_model
        args.lc = args_old.lc

    # Quick Modes
    if args.run_eval:
        print("Run Eval Mode")
        args.test = True
        args.pre_trained = True
        args.pre_trained_folder = saved_model_path
    elif args.run_model:
        print("Run Model Mode")
        args.test = True
        args.pre_trained = True
        args.pre_trained_model_path = saved_model_path

    # Readout
    nmode = args.nmode
    pnoise = args.pnoise
    smooth_mul = args.smooth_mul
    dsc_mul = args.dsc_mul
    dc_mul = args.dc_mul
    rsc_mul = args.rsc_mul
    rsc_low_mul = args.rsc_low_mul
    softce = args.softce
    lc = args.lc
    pytorch_scaling = args.pytorch_scaling
    test_interval = args.test_interval
    velodyne_depth = args.velodyne_depth
    pre_trained_folder = args.pre_trained_folder
    test = args.test
    exp_name = args.exp_name
    dataset_name = args.dataset
    batch_size = args.batch_size
    n_epoch = args.nepoch
    TB_add_img_interv = args.TB_add_img_interv
    pre_trained = args.pre_trained
    t_win_r = args.t_win
    nDepth = args.ndepth
    qpower = args.qpower
    ngpu = args.ngpu
    drefine = args.drefine
    halflr = [float(x) for x in args.halflr.split()]
    flow_rgb_mul = args.flow_rgb_mul
    flow_depth_mul = args.flow_depth_mul

    # Checks
    # if softce and not pytorch_scaling: raise('Soft CE needs Pytorch scaling to be turned on?')

    # Linear
    # d_candi = np.linspace(args.d_min, args.d_max, nDepth)
    # d_candi_up = np.linspace(args.d_min, args.d_max, nDepth*4)

    # Quad
    d_candi = util.powerf(args.d_min, args.d_max, nDepth, qpower)
    d_candi_up = util.powerf(args.d_min, args.d_max, nDepth * 2, qpower)

    LR = args.LR
    sigma_soft_max = args.sigma_soft_max  # 10.#500.
    dnet_feature_dim = args.feature_dim
    frame_interv = args.frame_interv  # should be multiple of 5 for scanNet dataset
    if_clip_gradient = args.grad_clip
    grad_clip_max = args.grad_clip_max
    d_candi_dmap_ref = d_candi
    nDepth_dmap_ref = nDepth
    viz = args.viz

    # Save Folder #
    util.m_makedir(saved_model_path)
    savemodel_interv = args.save_model_interv
    if not test: util.save_argparse(args, saved_model_path + "/args.json")

    # Writer #
    log_dir = 'outputs/%s/%s' % (args.TB_fldr, exp_name)
    writer = SummaryWriter(log_dir=log_dir, comment='%s' % (exp_name))
    util.save_args(args, '%s/tr_paras.txt' % (log_dir))  # save the training parameters #
    logfile = os.path.join(log_dir, 'log_' + str(time.time()) + '.txt')
    stdout = Logger(logfile)
    sys.stdout = stdout

    if lc:
        # CO Stuff
        from light_curtain import LightCurtain
        lightcurtain = LightCurtain()
    else:
        lightcurtain = None

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
        if not args.change_aspect_ratio:  # we will keep the aspect ratio and do cropping
            img_size = [768, 256]
            crop_w = 384
        else:  # we will change the aspect ratio and NOT do cropping
            img_size = [768, 256]
            crop_w = None

        # Testing Data Loader
        testing_inputs = {
            "pytorch_scaling": pytorch_scaling,
            "velodyne_depth": velodyne_depth,
            "dataset_path": dataset_path,
            "t_win_r": t_win_r,
            "img_size": img_size,
            "crop_w": crop_w,
            "d_candi": d_candi,
            "d_candi_up": d_candi_up,
            "dataset_name": "kitti",
            "hack_num": hack_num,
            "batch_size": batch_size,
            "n_epoch": 1,
            "qmax": 1,
            "mode": "val"
        }
        btest = BatchSchedulerMP(testing_inputs, 1)

        # Training Data Loader
        training_inputs = {
            "pytorch_scaling": pytorch_scaling,
            "velodyne_depth": velodyne_depth,
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
            "qmax": 1,
            "mode": "train"
        }
        if not test: b = BatchSchedulerMP(training_inputs, 0)

    if not test:
        if b.mode == 0:
            print("Preloading..")
            while b.queue.qsize() < b.inputs["qmax"]:
                time.sleep(0.1)

    # ==================================================================================== #

    # Model
    print('Init Network - Assume left usage')
    model_KVnet = KVNET(feature_dim=dnet_feature_dim, cam_intrinsics=None,
                        d_candi=d_candi, d_candi_up=d_candi_up, sigma_soft_max=sigma_soft_max,
                        KVNet_feature_dim=dnet_feature_dim,
                        d_upsample_ratio_KV_net=None, drefine=drefine, nmode=nmode)

    # model_KVnet = torch.nn.DataParallel(model_KVnet,  dim=0)
    model_KVnet.cuda()
    model_KVnet.train()

    optimizer_KV = optim.Adam(model_KVnet.parameters(), lr=LR, betas=(.9, .999))

    model_path_KV = args.pre_trained_model_path
    if model_path_KV is not '.' and pre_trained:
        model_path_KV = util.load_filenames_from_folder(model_path_KV)[-1]
        print('loading KV_net at %s' % (model_path_KV))
        lparams = util.load_pretrained_model(model_KVnet, model_path_KV, optimizer_KV)
    else:
        lparams = None

    print('Done')
    global exit

    # Additional Params
    addparams = {
        "softce": softce,
        "dsc_mul": dsc_mul,
        "dc_mul": dc_mul,
        "rsc_mul": rsc_mul,
        "rsc_low_mul": rsc_low_mul,
        "smooth_mul": smooth_mul,
        "pnoise": pnoise,
        "flow_rgb_mul": flow_rgb_mul,
        "flow_depth_mul": flow_depth_mul,
    }

    # Evaluate Model Graph
    if len(pre_trained_folder):
        all_files = util.load_filenames_from_folder(pre_trained_folder)
        model_KVnet.eval()
        foutput = {
            "name": exp_name,
            "command": ' '.join(sys.argv[1:]),
            "test_interval": 5000,
            "rmse": [],
            "rmse_low": [],
            "sil": [],
            "sil_low": [],
            "rsc": [],
            "smooth": [],
            "dc": []
        }
        for file in all_files:
            print(file)
            util.load_pretrained_model(model_KVnet, file, None)
            results, results_low, rsc, smooth, dc = testing(model_KVnet, btest, d_candi, d_candi_up, ngpu, addparams,
                                                            None, None)
            foutput["rmse"].append(results["rmse"][0]);
            foutput["rmse_low"].append(results_low["rmse"][0]);
            foutput["sil"].append(results["scale invariant log"][0]);
            foutput["sil_low"].append(results_low["scale invariant log"][0])
            foutput["rsc"].append(float(rsc))
            foutput["smooth"].append(float(smooth))
            foutput["dc"].append(float(dc))
        print("RMSES")
        print(foutput["rmse"])
        print("SILS")
        print(foutput["sil"])
        print("RMSES_low")
        print(foutput["rmse_low"])
        print("SILS_low")
        print(foutput["sil_low"])
        import json
        with open('outputs/saved_models/' + exp_name + '.json', 'w') as f:
            json.dump(foutput, f)
        sys.exit()

    # Test
    if test:
        model_KVnet.eval()
        results = testing(model_KVnet, btest, d_candi, d_candi_up, ngpu, addparams, visualizer, lightcurtain)
        print(results)
        sys.exit()

    # ==================================================================================== #

    # Vars
    if lparams is not None:
        LOSS = []
        total_iter = lparams["iter"]
    else:
        LOSS = []
        total_iter = -1

    # Keep Getting
    start = time.time()
    prev_output = None
    for items in b.get_mp():
        end = time.time()
        print("Data: " + str(end - start))
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

        # LR Control
        completion_percentage = float(iepoch)/float(n_epoch)
        if completion_percentage in halflr:
            print("Halving LR...")
            halflr.remove(completion_percentage)
            util.half_lr(optimizer_KV)

        # import copy
        # local_info = copy.deepcopy(local_info)

        # Process
        n_valid_batch = local_info['is_valid'].sum()
        if n_valid_batch > 0:
            local_info_valid = batch_loader.get_valid_items(local_info)
            local_info_valid["d_candi"] = d_candi

            # Reset Video
            if frame_count == 1:
                print('Resetting..')
                prev_output = None

            # Train
            prev_output = train(model_KVnet, optimizer_KV, local_info_valid, ngpu, addparams, total_iter, prev_output,
                                visualizer, lightcurtain)
            loss_v = prev_output["loss"]
            # loss_v = 0
        else:
            loss_v = LOSS[-1]
            pass

        # Add Iterations
        total_iter += 1

        # logging #
        if frame_count > 0:
            LOSS.append(loss_v)
            print('video batch %d / %d, iter: %d, frame_count: %d; Epoch: %d / %d, loss = %.5f' \
                  % (batch_idx + 1, batch_length, total_iter, frame_count, iepoch + 1, n_epoch, loss_v))
            writer.add_scalar('data/train_error', float(loss_v), total_iter)

        # Save
        if total_iter % savemodel_interv == 0:
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


def generate_model_input(local_info_valid, camside="left", softce=0, pnoise=0):
    # Ensure same size
    valid = (len(local_info_valid[camside + "_cam_intrins"]) == len(
        local_info_valid[camside + "_src_cam_poses"]) == len(local_info_valid["src_dats"]))
    if not valid:
        raise Exception('Batch size invalid')

    # Keep to middle only
    midval = int(len(local_info_valid["src_dats"][0]) / 2)
    preval = 0

    # Grab ground truth digitized map
    dmap_imgsize_digit_arr = []
    dmap_digit_arr = []
    dmap_imgsize_arr = []
    dmap_arr = []
    dmap_imgsize_prev_arr = []
    dmap_prev_arr = []
    for i in range(0, len(local_info_valid["src_dats"])):
        dmap_imgsize_digit = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_imgsize_digit"]
        dmap_imgsize_digit_arr.append(dmap_imgsize_digit)
        dmap_digit = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap"]
        dmap_imgsize = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_imgsize"]
        dmap = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_raw"]
        dmap_digit_arr.append(dmap_digit)
        dmap_imgsize_arr.append(dmap_imgsize)
        dmap_arr.append(dmap)
        dmap_imgsize_prev = local_info_valid["src_dats"][i][preval][camside + "_camera"]["dmap_imgsize"]
        dmap_prev = local_info_valid["src_dats"][i][preval][camside + "_camera"]["dmap_raw"]
        dmap_imgsize_prev_arr.append(dmap_imgsize_prev)
        dmap_prev_arr.append(dmap_prev)
    dmap_imgsize_digits = torch.cat(dmap_imgsize_digit_arr)  # [B,256,384] uint64
    dmap_digits = torch.cat(dmap_digit_arr)  # [B,64,96] uint64
    dmap_imgsizes = torch.cat(dmap_imgsize_arr)
    dmaps = torch.cat(dmap_arr)
    dmap_imgsizes_prev = torch.cat(dmap_imgsize_prev_arr)
    dmaps_prev = torch.cat(dmap_prev_arr)

    intrinsics_arr = []
    intrinsics_up_arr = []
    unit_ray_arr = []
    for i in range(0, len(local_info_valid[camside + "_cam_intrins"])):
        intr = local_info_valid[camside + "_cam_intrins"][i]["intrinsic_M_cuda"]
        intr_up = intr * 4;
        intr_up[2, 2] = 1;
        intrinsics_arr.append(intr.unsqueeze(0))
        intrinsics_up_arr.append(intr_up.unsqueeze(0))
        unit_ray_arr.append(local_info_valid[camside + "_cam_intrins"][i]["unit_ray_array_2D"].unsqueeze(0))
    intrinsics = torch.cat(intrinsics_arr)
    intrinsics_up = torch.cat(intrinsics_up_arr)
    unit_ray = torch.cat(unit_ray_arr)

    src_cam_poses_arr = []
    for i in range(0, len(local_info_valid[camside + "_src_cam_poses"])):
        pose = local_info_valid[camside + "_src_cam_poses"][i]
        src_cam_poses_arr.append(pose[:, 0:midval + 1, :, :])  # currently [1x3x4x4]
    src_cam_poses = torch.cat(src_cam_poses_arr)
    if pnoise: src_cam_poses = util.add_noise2pose(src_cam_poses, pnoise)

    mask_imgsize_arr = []
    mask_arr = []
    rgb_arr = []
    debug_path = []
    for i in range(0, len(local_info_valid["src_dats"])):
        rgb_set = []
        debug_path_int = []
        for j in range(0, len(local_info_valid["src_dats"][i])):
            rgb_set.append(local_info_valid["src_dats"][i][j][camside + "_camera"]["img"])
            debug_path_int.append(local_info_valid["src_dats"][i][j][camside + "_camera"]["img_path"])
            if j == midval: break
        rgb_arr.append(torch.cat(rgb_set).unsqueeze(0))
        debug_path.append(debug_path_int)
        mask_imgsize_arr.append(local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_mask_imgsize"])
        mask_arr.append(local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_mask"])
    rgb = torch.cat(rgb_arr)
    masks_imgsize = torch.cat(mask_imgsize_arr).float()
    masks = torch.cat(mask_arr).float()

    # Create Soft Label
    d_candi = local_info_valid["d_candi"]
    if softce:
        soft_labels_imgsize = []
        soft_labels = []
        variance = torch.tensor(softce)
        for i in range(0, dmap_imgsizes.shape[0]):
            # Clamping
            dmap_imgsize = dmap_imgsizes[i, :, :].clamp(d_candi[0], d_candi[-1]) * masks_imgsize[i, 0, :, :]
            dmap = dmaps[i, :, :].clamp(d_candi[0], d_candi[-1]) * masks[i, 0, :, :]
            soft_labels_imgsize.append(
                util.gen_soft_label_torch(d_candi, dmap_imgsize.cuda(), variance, zero_invalid=True))
            soft_labels.append(util.gen_soft_label_torch(d_candi, dmap.cuda(), variance, zero_invalid=True))
            # Generate Fake one with all 1
            # soft_labels.append(util.digitized_to_dpv(dmap_digits[i,:,:].unsqueeze(0), len(d_candi)).squeeze(0).cuda())
            # soft_labels_imgsize.append(util.digitized_to_dpv(dmap_imgsize_digits[i,:,:].unsqueeze(0), len(d_candi)).squeeze(0).cuda())
    else:
        soft_labels_imgsize = []
        soft_labels = []

    model_input = {
        "intrinsics": intrinsics.cuda(),
        "intrinsics_up": intrinsics_up.cuda(),
        "unit_ray": unit_ray.cuda(),
        "src_cam_poses": src_cam_poses.cuda(),
        "rgb": rgb.cuda(),
        "bv_predict": None,  # Has to be [B, 64, H, W],
        "dmaps": dmaps.cuda(),
        "masks": masks.cuda(),
        "d_candi": d_candi,
    }

    # dmap_imgsizes_prev = torch.cat(dmap_imgsize_prev_arr)
    # dmaps_prev = torch.cat(dmap_prev_arr)

    gt_input = {
        "masks_imgsizes": masks_imgsize.cuda(),
        "masks": masks.cuda(),
        "dmap_imgsize_digits": dmap_imgsize_digits.cuda(),
        "dmap_digits": dmap_digits.cuda(),
        "dmap_imgsizes": dmap_imgsizes.cuda(),
        "dmaps": dmaps.cuda(),
        "dmap_imgsizes_prev": dmap_imgsizes_prev.cuda(),
        "dmaps_prev": dmaps_prev.cuda(),
        "soft_labels_imgsize": soft_labels_imgsize,
        "soft_labels": soft_labels
    }

    return model_input, gt_input


def testing(model, btest, d_candi, d_candi_up, ngpu, addparams, visualizer, lightcurtain):
    all_errors = []
    all_errors_low = []
    all_lc_before_errors = []
    all_lc_after_errors = []
    rsc_sum = 0.
    dc_sum = 0.
    smooth_sum = 0.
    counter = 0.
    start = time.time()
    softce = addparams["softce"]

    # Cloud for distance
    dcloud = []
    for m in range(0, 40):
        dcloud.append([0, 0, m, 255, 255, 255, 0, 0, 0])
    dcloud = np.array(dcloud).astype(np.float32)

    prev_output = None
    for items in btest.get_mp():
        if exit:
            print("Exiting")
            if visualizer is not None: visualizer.kill_received = True
            btest.stop()
            sys.exit()

        # Get data
        local_info, batch_length, batch_idx, frame_count, ref_indx, iepoch = items
        if frame_count == 1: prev_output = None

        # Need a way to see that video batch ended etc.

        # Process
        n_valid_batch = local_info['is_valid'].sum()
        if n_valid_batch > 0:
            local_info_valid = batch_loader.get_valid_items(local_info)
            local_info_valid["d_candi"] = d_candi

            # Create input
            model_input, gt_input = generate_model_input(local_info_valid, "left", softce)
            model_input_right, gt_input_right = generate_model_input(local_info_valid, "right")

            # Create LC output
            # Right now assume r_candi = d_candi
            if lightcurtain is not None:
                if not lightcurtain.initialized:
                    PARAMS={
                        "intr_rgb": model_input["intrinsics_up"][0, :, :].cpu().numpy(),
                        "dist_rgb": [0., 0., 0., 0., 0.],
                        "size_rgb": [model_input["rgb"].shape[4], model_input["rgb"].shape[3]],
                        "intr_lc": model_input["intrinsics_up"][0, :, :].cpu().numpy(),
                        "dist_lc": [0., 0., 0., 0., 0.],
                        "size_lc": [model_input["rgb"].shape[4], model_input["rgb"].shape[3]],
                        "rTc": np.array([
                                [1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.],
                            ]).astype(np.float32),
                        "lTc": np.array([
                                [1., 0., 0., 0.2],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.],
                            ]).astype(np.float32),
                        "laser_fov": 80.,
                        "d_candi": d_candi,
                        "r_candi": d_candi,
                        "d_candi_up": d_candi_up,
                        "r_candi_up": d_candi_up
                    }
                    lightcurtain.init(PARAMS)

            # Stuff
            start = time.time()
            model_input["prev_output"] = prev_output
            BV_cur_all_arr, BV_cur_refined_all = torch.nn.parallel.data_parallel(model, model_input, range(ngpu))
            BV_cur_all = BV_cur_all_arr[-1]
            prev_output = BV_cur_all.detach()
            # print("Forward: " + str(time.time() - start))

            # Truth
            depthmap_truth_all = gt_input["dmap_imgsizes"]  # [1,256,384]
            depthmap_truth_low_all = gt_input["dmaps"]  # [1,256,384]

            # Masks
            depth_mask_all = gt_input["masks_imgsizes"][:, :, :, :].float()
            depth_mask_low_all = gt_input["masks"][:, :, :, :].float()

            # Batch
            bsize = BV_cur_refined_all.shape[0]
            pose_target2src = local_info_valid["T_left2right"].unsqueeze(0).cuda()
            for b in range(bsize):
                start = time.time()
                BV_cur = BV_cur_all[b, :, :, :].unsqueeze(0)
                BV_cur_refined = BV_cur_refined_all[b, :, :, :].unsqueeze(0)
                depthmap_truth = depthmap_truth_all[b, :, :].unsqueeze(0)
                depthmap_truth_low = depthmap_truth_low_all[b, :, :].unsqueeze(0)
                depth_mask = depth_mask_all[b, :, :, :]
                depth_mask_low = depth_mask_low_all[b, :, :, :]
                intr_up = model_input["intrinsics_up"][b, :, :].unsqueeze(0)
                intr = model_input["intrinsics"][b, :, :].unsqueeze(0)
                left_rgb = model_input["rgb"][b, -1, :, :, :].unsqueeze(0)
                right_rgb = model_input_right["rgb"][b, -1, :, :, :].unsqueeze(0)
                soft_label_refined = gt_input["soft_labels_imgsize"][b].unsqueeze(0)
                #hard_label_refined = util.digitized_to_dpv(gt_input["dmap_imgsize_digits"][b,:,:].unsqueeze(0), len(d_candi))

                # Predicted
                dpv_low_predicted = BV_cur[0, :, :, :].unsqueeze(0).detach()
                dpv_predicted = BV_cur_refined[0, :, :, :].unsqueeze(0).detach()
                depthmap_predicted = util.dpv_to_depthmap(dpv_predicted, d_candi, BV_log=True)  # [1,256,384]
                depthmap_low_predicted = util.dpv_to_depthmap(dpv_low_predicted, d_candi, BV_log=True)  # [1,256,384]

                # Losses
                rsc_sum += losses.rgb_stereo_consistency_loss(right_rgb, left_rgb, depthmap_predicted,
                                                              pose_target2src, intr_up)
                smooth_sum += losses.edge_aware_smoothness_loss([depthmap_predicted.unsqueeze(0)], left_rgb, 1)
                dc_sum += losses.depth_consistency_loss(depthmap_predicted, depthmap_low_predicted)
                counter += 1

                # Generate Numpy
                depthmap_predicted_np = (depthmap_predicted * depth_mask).squeeze(0).cpu().numpy()
                depthmap_predicted_low_np = (depthmap_low_predicted * depth_mask_low).squeeze(0).cpu().numpy()
                depthmap_truth_np = (depthmap_truth * depth_mask.cpu()).squeeze(0).cpu().numpy()
                depthmap_truth_low_np = (depthmap_truth_low * depth_mask_low.cpu()).squeeze(0).cpu().numpy()

                # Clamp the truth max depth
                depthmap_truth_np[depthmap_truth_np >= d_candi[-1]] = d_candi[-1]
                depthmap_truth_low_np[depthmap_truth_low_np >= d_candi[-1]] = d_candi[-1]

                # Error
                errors = util.depthError(depthmap_predicted_np, depthmap_truth_np)
                errors_low = util.depthError(depthmap_predicted_low_np, depthmap_truth_low_np)
                all_errors.append(errors)
                all_errors_low.append(errors_low)

                # # Test Transform
                # transform = iv.pose_vec2mat_full(torch.Tensor([0., 0., 0., 0.0, 0.0, 0.]).unsqueeze(0).repeat(2,1)).cuda()
                # print(transform)
                # depthmaps = depthmap_truth_low.repeat(2,1,1).cuda()
                # intrinsics = intr.squeeze(0).cuda()
                # depthmaps_transformed = util.transform_depth(depthmaps, intrinsics, transform)

                # Light Curtain
                if lightcurtain is not None:
                    # Field
                    dpv_plane_predicted, debugmap = losses.gen_ufield(dpv_predicted, d_candi, intr_up.squeeze(0), visualizer, img=None)
                    dpv_plane_predicted = dpv_plane_predicted.squeeze(0)
                    plane_mask = depth_mask * debugmap  # [1x256x384]
                    plane_mask = depth_mask

                    # Plan
                    lc_paths, field_visual = lightcurtain.plan(dpv_plane_predicted)

                    # Sense
                    lc_outputs = []
                    lc_DPVs = []
                    for lc_path in lc_paths:
                        lc_DPV, output = lightcurtain.sense_high(depthmap_truth_np, lc_path, visualizer)
                        lc_outputs.append(output)
                        lc_DPVs.append(lc_DPV)

                    # Fuse
                    dpv_pred = torch.exp(dpv_predicted)[0]
                    dpv_fused = torch.exp(
                        torch.log(dpv_pred) + torch.log(lc_DPVs[0]) + torch.log(lc_DPVs[1]) + torch.log(lc_DPVs[2]))
                    dpv_fused = dpv_fused / torch.sum(dpv_fused, dim=0)
                    dpv_fused_depth = util.dpv_to_depthmap(dpv_fused.unsqueeze(0), d_candi, BV_log=False)
                    dpv_pred_depth = util.dpv_to_depthmap(dpv_pred.unsqueeze(0), d_candi, BV_log=False)
                    depthmap_truth_capped = torch.tensor(depthmap_truth_np).cuda().unsqueeze(0)

                    # Store
                    depthmap_truth_lc_np = (depthmap_truth_capped * plane_mask).squeeze(0).cpu().numpy()
                    depthmap_pred_lc_np = (dpv_pred_depth * plane_mask).squeeze(0).cpu().numpy()
                    depthmap_fuse_lc_np = (dpv_fused_depth * plane_mask).squeeze(0).cpu().numpy()
                    lc_before_error = util.depthError(depthmap_pred_lc_np, depthmap_truth_lc_np)
                    lc_after_error = util.depthError(depthmap_fuse_lc_np, depthmap_truth_lc_np)
                    all_lc_before_errors.append(lc_before_error)
                    all_lc_after_errors.append(lc_after_error)

                    # Visualization
                    if visualizer is not None:
                        dpv_plane_truth, _ = losses.gen_ufield(soft_label_refined, d_candi, intr_up.squeeze(0), visualizer, None, False, True)
                        dpv_plane_fused, _ = losses.gen_ufield(dpv_fused.unsqueeze(0), d_candi, intr_up.squeeze(0), visualizer, img=None, BV_log=False)

                        rgbimg = util.torchrgb_to_cv2(left_rgb.squeeze(0))
                        rgbimg[:,:,0] += debugmap.squeeze(0).cpu().numpy()
                        cv2.imshow("rgbimg", rgbimg)
                        cv2.imshow("field_visual", field_visual)
                        #cv2.imshow("dpv_plane_truth", util.dpvplane_normalize(dpv_plane_truth).squeeze(0).cpu().numpy())

                        # dpv_plane_fused_norm = util.dpvplane_normalize(dpv_plane_fused)
                        # dpv_plane_pred_norm = util.dpvplane_normalize(dpv_plane_predicted.unsqueeze(0))
                        # cv2.imshow("dpv_plane_pred_norm", util.dpvplane_draw(dpv_plane_truth.squeeze(0), dpv_plane_pred_norm.squeeze(0)))
                        # cv2.imshow("dpv_plane_fused_norm", util.dpvplane_draw(dpv_plane_truth.squeeze(0), dpv_plane_fused_norm.squeeze(0)))

                        visualizer.addCloud(util.lcpath_to_cloud(lc_paths[0]), 3)
                        visualizer.addCloud(util.lcpath_to_cloud(lc_paths[1], [255,0,0]), 3)
                        visualizer.addCloud(util.lcpath_to_cloud(lc_paths[2], [255, 0, 0]), 3)
                        visualizer.addCloud(util.lcoutput_to_cloud(lc_outputs[0]), 3)
                        visualizer.addCloud(util.lcoutput_to_cloud(lc_outputs[1]), 3)
                        visualizer.addCloud(util.lcoutput_to_cloud(lc_outputs[2]), 3)
                        b = 0
                        cloud_truth = util.tocloud(depthmap_truth, util.demean(left_rgb[b,:,:,:]), intr_up[b,:,:], None)
                        #cloud_pred = util.tocloud(torch.tensor(depthmap_fuse_lc_np).unsqueeze(0), util.demean(left_rgb[b,:,:,:]), intr_up[b,:,:], None)
                        visualizer.addCloud(cloud_truth)
                        visualizer.swapBuffer()
                        cv2.waitKey(0)

                # Normal
                else:
                    if visualizer is not None:
                        intr_up = model_input["intrinsics_up"][b, :, :].cpu()
                        intr = model_input["intrinsics"][b, :, :].cpu()
                        depthmap_predicted_np = (depthmap_predicted * 1).cpu()
                        depthmap_low_predicted_np = (depthmap_low_predicted * 1).cpu()
                        depthmap_predicted_np[:, 0:depthmap_predicted_np.shape[1] / 2, :] = 0
                        depthmap_low_predicted_np[:, 0:depthmap_low_predicted_np.shape[1] / 2, :] = 0

                        # Get Image
                        img = model_input["rgb"][b, -1, :, :, :].cpu()  # [1,3,256,384]
                        img[0, :, :] = img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
                        img[1, :, :] = img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
                        img[2, :, :] = img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
                        img_low = F.avg_pool2d(img, 4)
                        img_color = cv2.cvtColor(img[:, :, :].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
                        img_depth = cv2.cvtColor(depthmap_predicted_np[0, :, :].numpy() / 100., cv2.COLOR_GRAY2BGR)
                        combined = np.hstack([img_color, img_depth])
                        print(combined.shape)

                        # Cloud
                        cloud_low_orig = util.tocloud(depthmap_low_predicted_np, img_low, intr, None)
                        cloud_orig = util.tocloud(depthmap_predicted_np, img, intr_up, None)
                        cloud_truth = util.tocloud(torch.tensor(depthmap_truth_np[np.newaxis, :]), img, intr_up, None)
                        cloud_low_truth = util.tocloud(torch.tensor(depthmap_truth_low_np[np.newaxis, :]), img_low, intr,
                                                       None, [0,0,255])
                        cv2.imshow("win", combined)
                        print(cloud_orig.shape)
                        #visualizer.addCloud(cloud_low_truth, 3)
                        visualizer.addCloud(cloud_orig,1)
                        #visualizer.addCloud(cloud_low_truth, 3)
                        #visualizer.addCloud(cloud_orig, 1)
                        #visualizer.addCloud(slicecloud, 2)
                        visualizer.addCloud(dcloud, 4)
                        visualizer.swapBuffer()
                        key = cv2.waitKey(0)
                        print("--")

                continue

                # Viz
                if visualizer is not None and b == 0:
                    intr_up = model_input["intrinsics_up"][b, :, :].cpu()
                    intr = model_input["intrinsics"][b, :, :].cpu()
                    depthmap_predicted_np = (depthmap_predicted * 1).cpu()
                    depthmap_low_predicted_np = (depthmap_low_predicted * 1).cpu()
                    depthmap_predicted_np[:, 0:depthmap_predicted_np.shape[1] / 2, :] = 0
                    depthmap_low_predicted_np[:, 0:depthmap_low_predicted_np.shape[1] / 2, :] = 0

                    # Get Image
                    img = model_input["rgb"][b, -1, :, :, :].cpu()  # [1,3,256,384]
                    img[0, :, :] = img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
                    img[1, :, :] = img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
                    img[2, :, :] = img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
                    img_low = F.avg_pool2d(img, 4)
                    img_color = cv2.cvtColor(img[:, :, :].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
                    img_depth = cv2.cvtColor(depthmap_predicted_np[0, :, :].numpy() / 100., cv2.COLOR_GRAY2BGR)
                    combined = np.hstack([img_color, img_depth])
                    print(combined.shape)

                    # Light Curtain Path Raw
                    if lightcurtain is not None:
                        # We must draw the full path here
                        planned_path = lc_paths[0]
                        pathcloud = np.zeros((planned_path.shape[0], 9))
                        pathcloud[:,0] = planned_path[:,0]
                        pathcloud[:, 1] = -1
                        pathcloud[:, 2] = planned_path[:, 1]
                        pathcloud[:, 3:5] = 255
                        pathcloud = util.hack(pathcloud)
                        visualizer.addCloud(pathcloud, 3)

                        #Looks bad. Do an experiemnt and how it would be with more discretization? smooth it

                    # # Light Curtain
                    # if lightcurtain is not None:
                    #     #path = lightcurtain.get_flat(22)
                    #     #output = lightcurtain.sense_low(depthmap_truth_low_np, lc_paths[0])
                    #     output = lightcurtain.sense_high(depthmap_truth_np, lc_paths[0])
                    #     output[np.isnan(output[:,:,0])] = 0
                    #     output = output.reshape((output.shape[0]*output.shape[1], 4))
                    #     lccloud = np.append(output, np.zeros((output.shape[0], 5)), axis=1)
                    #     lccloud[:, 4:6] = 50
                    #     lccloud = util.hack(lccloud)
                    #     visualizer.addCloud(lccloud, 3)

                    # # Field
                    # soft_label_refined = soft_label_refined * depth_mask.float()
                    # dpv_plane_predicted, debugmap = losses.gen_ufield(dpv_predicted, d_candi, intr_up, visualizer, img, True, False)
                    # dpv_plane_truth, _ = losses.gen_ufield(soft_label_refined, d_candi, intr_up, visualizer, img, False, True)

                    # import matplotlib.pyplot as plt
                    # def plotfig(index):
                    #     dist_pred = dpv_plane_predicted[0,:,index].cpu().numpy()
                    #     dist_truth = dpv_plane_truth[0, :, index].cpu().numpy()
                    #     plt.figure()
                    #     plt.plot(np.array(d_candi), dist_pred)
                    #     plt.plot(np.array(d_candi), dist_truth)
                    #     plt.ion()
                    #     plt.pause(0.005)
                    #     plt.show()
                    # plotfig(283)
                    # plotfig(345)
                    # # plotfig(24)
                    # # plotfig(187)

                    # cloud = util.tocloud(debugmap.cpu(), img, intr_up, None, [255, 255, 255])
                    # #visualizer.addCloud(cloud, 3)
                    # viz1 = dpv_plane_truth.squeeze(0).cpu().numpy()
                    # viz2 = dpv_plane_predicted.squeeze(0).cpu().numpy()
                    # truth = ((viz1 * 1)*255).astype(np.uint8)
                    # pred = np.clip((viz2 * 5)*255, 0, 255).astype(np.uint8)
                    # mask = (pred > 1).astype(np.uint8)
                    # pred_col = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
                    # for r in range(0, pred.shape[0]):
                    #     for c in range(0, pred.shape[1]):
                    #         if truth[r,c] > 1:
                    #             pred_col[r,c,:] = [0,0,255]
                    # cv2.imshow("final", pred_col)

                    # Cloud
                    cloud_low_orig = util.tocloud(depthmap_low_predicted_np, img_low, intr, None)
                    cloud_orig = util.tocloud(depthmap_predicted_np, img, intr_up, None)
                    cloud_truth = util.tocloud(torch.tensor(depthmap_truth_np[np.newaxis, :]), img, intr_up, None)
                    cloud_low_truth = util.tocloud(torch.tensor(depthmap_truth_low_np[np.newaxis, :]), img_low, intr,
                                                   None)
                    cv2.imshow("win", combined)
                    print(cloud_orig.shape)
                    #visualizer.addCloud(cloud_low_truth, 3)
                    visualizer.addCloud(cloud_truth,1)
                    #visualizer.addCloud(cloud_low_orig, 3)
                    #visualizer.addCloud(cloud_orig, 1)
                    #visualizer.addCloud(slicecloud, 2)
                    visualizer.addCloud(dcloud, 4)
                    visualizer.swapBuffer()
                    key = cv2.waitKey(0)
                    print("--")

                    # # Is this true? Such a low resolution is an incorrect representation of the uncertainty etc.
                    # """
                    # need a function where i can test a pixel. i have intensity, groundtruth and z unc
                    # i can generate a graph from this similar to above (the gt is just to visualize)
                    # """
                    # for dist in np.arange(10, 30, 0.01):
                    #     if lightcurtain is not None:
                    #         path = lightcurtain.get_flat(dist)
                    #         output = lightcurtain.sense(depthmap_truth_low_np, path)
                    #         output[np.isnan(output[:,:,0])] = 0
                    #         output = output.reshape((output.shape[0]*output.shape[1], 4))
                    #         lccloud = np.append(output, np.zeros((output.shape[0], 5)), axis=1)
                    #         lccloud[:, 4:6] = 50
                    #         lccloud = util.hack(lccloud)
                    #         visualizer.addCloud(lccloud, 3)
                    #         visualizer.addCloud(cloud_low_truth, 3)
                    #         visualizer.swapBuffer()

        else:
            pass

    if lightcurtain is not None:
        results_before = util.evaluateErrors(all_lc_before_errors)
        results_after = util.evaluateErrors(all_lc_after_errors)
        print(results_before["rmse"][0], results_after["rmse"][0])
        print("--")
        print(results_before["scale invariant log"][0], results_after["scale invariant log"][0])
        stop

    results = util.evaluateErrors(all_errors)
    results_low = util.evaluateErrors(all_errors_low)
    rsc_sum /= counter
    smooth_sum /= counter
    dc_sum /= counter
    return results, results_low, rsc_sum, smooth_sum, dc_sum


def train(model, optimizer_KV, local_info_valid, ngpu, addparams, total_iter, prev_output, visualizer, lightcurtain):
    # start = time.time()
    # print(time.time() - start)
    # return {"loss": 0, "BV_cur_left": 0, "BV_cur_right": 0}

    # Readout AddParams
    d_candi = local_info_valid["d_candi"]
    softce = addparams["softce"]
    dsc_mul = addparams["dsc_mul"]
    dc_mul = addparams["dc_mul"]
    rsc_mul = addparams["rsc_mul"]
    rsc_low_mul = addparams["rsc_low_mul"]
    smooth_mul = addparams["smooth_mul"]
    pnoise = addparams["pnoise"]
    flow_rgb_mul = addparams["flow_rgb_mul"]
    flow_depth_mul = addparams["flow_depth_mul"]

    # Create inputs
    model_input_left, gt_input_left = generate_model_input(local_info_valid, "left", softce, pnoise)
    model_input_right, gt_input_right = generate_model_input(local_info_valid, "right", softce, pnoise)

    # Previous
    if prev_output is not None:
        model_input_left["prev_output"] = prev_output["BV_cur_left"]
        model_input_right["prev_output"] = prev_output["BV_cur_right"]
    else:
        model_input_left["prev_output"] = None
        model_input_right["prev_output"] = None

    # Run Forward
    BV_cur_left_array, BV_cur_refined_left_array, flow_left, flow_left_refined = torch.nn.parallel.data_parallel(model, model_input_left, range(ngpu))
    BV_cur_right_array, BV_cur_refined_right_array, flow_right, flow_right_refined = torch.nn.parallel.data_parallel(model, model_input_right, range(ngpu))

    # RGB Flow
    def flow_rgb_comp(ibatch, flow, prev, curr):
        flow_curr = flow[ibatch, 0:2, :, :].permute(1, 2, 0).unsqueeze(0)
        pred = util.flowarp(prev, flow_curr)
        return losses.rgb_loss(pred, curr)

    # Depth Flow
    def flow_depth_comp(ibatch, flow, prev, curr):
        # Issue, if the depth goes to zero, it becomes not even counted. can we stop this?
        flow_2d = flow[ibatch, 0:2, :, :].permute(1, 2, 0).unsqueeze(0)
        flow_z = flow[ibatch, 2, :, :]
        pred = util.flowarp(prev, flow_2d) + flow_z
        mask = (prev > 0) & (curr > 0)
        pred = pred * mask.float()
        return losses.depth_loss(pred, curr)

    flow_rgb_loss = 0.
    flow_rgb_count = 0.
    flow_depth_loss = 0.
    flow_depth_count = 0.
    if flow_left is not None:
        for ibatch in range(flow_left.shape[0]):
            # RGB
            flow_rgb_count += 1.
            rgb_left_refined_prev = model_input_left["rgb"][ibatch, 0, :, :, :].unsqueeze(0)
            rgb_left_refined_curr = model_input_left["rgb"][ibatch, -1, :, :, :].unsqueeze(0)
            rgb_right_refined_prev = model_input_right["rgb"][ibatch, 0, :, :, :].unsqueeze(0)
            rgb_right_refined_curr = model_input_right["rgb"][ibatch, -1, :, :, :].unsqueeze(0)
            rgb_left_prev = F.avg_pool2d(rgb_left_refined_prev, 4)
            rgb_left_curr = F.avg_pool2d(rgb_left_refined_curr, 4)
            rgb_right_prev = F.avg_pool2d(rgb_right_refined_prev, 4)
            rgb_right_curr = F.avg_pool2d(rgb_right_refined_curr, 4)

            # RGB Losses
            flow_rgb_loss += flow_rgb_comp(ibatch, flow_left, rgb_left_prev, rgb_left_curr)
            flow_rgb_loss += flow_rgb_comp(ibatch, flow_right, rgb_right_prev, rgb_right_curr)
            flow_rgb_loss += flow_rgb_comp(ibatch, flow_left_refined, rgb_left_refined_prev, rgb_left_refined_curr)
            flow_rgb_loss += flow_rgb_comp(ibatch, flow_right_refined, rgb_right_refined_prev, rgb_right_refined_curr)

            # Depth
            flow_depth_count += 1.
            depth_left_refined_prev = gt_input_left["dmap_imgsizes_prev"][ibatch, :, :].unsqueeze(0).unsqueeze(0)
            depth_left_refined_curr = gt_input_left["dmap_imgsizes"][ibatch, :, :].unsqueeze(0).unsqueeze(0)
            depth_right_refined_prev = gt_input_right["dmap_imgsizes_prev"][ibatch, :, :].unsqueeze(0).unsqueeze(0)
            depth_right_refined_curr = gt_input_right["dmap_imgsizes"][ibatch, :, :].unsqueeze(0).unsqueeze(0)
            depth_left_prev = gt_input_left["dmaps_prev"][ibatch, :, :].unsqueeze(0).unsqueeze(0)
            depth_left_curr = gt_input_left["dmaps"][ibatch, :, :].unsqueeze(0).unsqueeze(0)
            depth_right_prev = gt_input_right["dmaps_prev"][ibatch, :, :].unsqueeze(0).unsqueeze(0)
            depth_right_curr = gt_input_right["dmaps"][ibatch, :, :].unsqueeze(0).unsqueeze(0)

            # Depth Losses
            flow_depth_loss += flow_depth_comp(ibatch, flow_left, depth_left_prev, depth_left_curr)
            flow_depth_loss += flow_depth_comp(ibatch, flow_right, depth_right_prev, depth_right_curr)
            flow_depth_loss += flow_depth_comp(ibatch, flow_left_refined, depth_left_refined_prev, depth_left_refined_curr)
            flow_depth_loss += flow_depth_comp(ibatch, flow_right_refined, depth_right_refined_prev, depth_right_refined_curr)

    # NLL Loss for Low Res
    ce_loss = 0
    ce_count = 0
    for ind in range(len(BV_cur_left_array)):
        BV_cur_left = BV_cur_left_array[ind]
        BV_cur_right = BV_cur_right_array[ind]
        for ibatch in range(BV_cur_left.shape[0]):
            ce_count += 1
            if not softce:
                # Left Losses
                ce_loss = ce_loss + F.nll_loss(BV_cur_left[ibatch, :, :, :].unsqueeze(0),
                                               gt_input_left["dmap_digits"][ibatch, :, :].unsqueeze(0), ignore_index=0)
                # Right Losses
                ce_loss = ce_loss + F.nll_loss(BV_cur_right[ibatch, :, :, :].unsqueeze(0),
                                               gt_input_right["dmap_digits"][ibatch, :, :].unsqueeze(0), ignore_index=0)
            else:
                # Left Losses
                ce_loss = ce_loss + losses.soft_cross_entropy_loss(gt_input_left["soft_labels"][ibatch].unsqueeze(0),
                                                                   BV_cur_left[ibatch, :, :, :].unsqueeze(0),
                                                                   mask=gt_input_left["masks"][ibatch, :, :, :],
                                                                   BV_log=True)
                # Right Losses
                ce_loss = ce_loss + losses.soft_cross_entropy_loss(gt_input_right["soft_labels"][ibatch].unsqueeze(0),
                                                                   BV_cur_right[ibatch, :, :, :].unsqueeze(0),
                                                                   mask=gt_input_right["masks"][ibatch, :, :, :],
                                                                   BV_log=True)

    # NLL Loss for High Res
    for ind in range(len(BV_cur_refined_left_array)):
        BV_cur_refined_left = BV_cur_refined_left_array[ind]
        BV_cur_refined_right = BV_cur_refined_right_array[ind]
        for ibatch in range(BV_cur_refined_left.shape[0]):
            ce_count += 1
            if not softce:
                # Left Losses
                ce_loss = ce_loss + F.nll_loss(BV_cur_refined_left[ibatch, :, :, :].unsqueeze(0),
                                               gt_input_left["dmap_imgsize_digits"][ibatch, :, :].unsqueeze(0),
                                               ignore_index=0)
                # Right Losses
                ce_loss = ce_loss + F.nll_loss(BV_cur_refined_right[ibatch, :, :, :].unsqueeze(0),
                                               gt_input_right["dmap_imgsize_digits"][ibatch, :, :].unsqueeze(0),
                                               ignore_index=0)
            else:
                # Left Losses
                ce_loss = ce_loss + losses.soft_cross_entropy_loss(
                    gt_input_left["soft_labels_imgsize"][ibatch].unsqueeze(0),
                    BV_cur_refined_left[ibatch, :, :, :].unsqueeze(0),
                    mask=gt_input_left["masks_imgsizes"][ibatch, :, :, :],
                    BV_log=True)
                # Right Losses
                ce_loss = ce_loss + losses.soft_cross_entropy_loss(
                    gt_input_right["soft_labels_imgsize"][ibatch].unsqueeze(0),
                    BV_cur_refined_right[ibatch, :, :, :].unsqueeze(0),
                    mask=gt_input_right["masks_imgsizes"][ibatch, :, :, :],
                    BV_log=True)

    # Get Last BV_cur
    BV_cur_left = BV_cur_left_array[-1]
    BV_cur_right = BV_cur_right_array[-1]
    BV_cur_refined_left = BV_cur_refined_left_array[-1]
    BV_cur_refined_right = BV_cur_refined_right_array[-1]

    # Regress all depthmaps once here
    small_dm_left_arr = []
    large_dm_left_arr = []
    small_dm_right_arr = []
    large_dm_right_arr = []
    for ibatch in range(BV_cur_left.shape[0]):
        small_dm_left_arr.append(util.dpv_to_depthmap(BV_cur_left[ibatch, :, :, :].unsqueeze(0), d_candi, BV_log=True))
        large_dm_left_arr.append(
            util.dpv_to_depthmap(BV_cur_refined_left[ibatch, :, :, :].unsqueeze(0), d_candi, BV_log=True))
        small_dm_right_arr.append(
            util.dpv_to_depthmap(BV_cur_right[ibatch, :, :, :].unsqueeze(0), d_candi, BV_log=True))
        large_dm_right_arr.append(
            util.dpv_to_depthmap(BV_cur_refined_right[ibatch, :, :, :].unsqueeze(0), d_candi, BV_log=True))

    # Downsample Consistency Loss (Should we even have a mask here?)
    dc_loss = 0
    for ibatch in range(BV_cur_left.shape[0]):
        if dc_mul == 0: break
        # Left
        mask_left = gt_input_left["masks"][ibatch, :, :, :]
        small_dm_left = small_dm_left_arr[ibatch]
        large_dm_left = large_dm_left_arr[ibatch]
        dc_loss = dc_loss + losses.depth_consistency_loss(large_dm_left, small_dm_left)
        # Right
        mask_right = gt_input_right["masks"][ibatch, :, :, :]
        small_dm_right = small_dm_right_arr[ibatch]
        large_dm_right = large_dm_right_arr[ibatch]
        dc_loss = dc_loss + losses.depth_consistency_loss(large_dm_right, small_dm_right)

    # Could we make the mask cut even more..so objects too far are not counted

    # Depth Stereo Consistency Loss (Very slow? cos of the backward - Ignore low res?)
    T_left2right = local_info_valid["T_left2right"]
    pose_target2src = T_left2right
    pose_target2src = torch.unsqueeze(pose_target2src, 0).cuda()
    pose_src2target = torch.inverse(T_left2right)
    pose_src2target = torch.unsqueeze(pose_src2target, 0).cuda()
    dsc_loss = 0
    for ibatch in range(BV_cur_left.shape[0]):
        if dsc_mul == 0: break
        # Get all Data
        intr_up_left = model_input_left["intrinsics_up"][ibatch, :, :].unsqueeze(0)
        intr_left = model_input_left["intrinsics"][ibatch, :, :].unsqueeze(0)
        intr_up_right = model_input_right["intrinsics_up"][ibatch, :, :].unsqueeze(0)
        intr_right = model_input_right["intrinsics"][ibatch, :, :].unsqueeze(0)
        depth_up_left = large_dm_left_arr[ibatch].unsqueeze(0)
        depth_left = small_dm_left_arr[ibatch].unsqueeze(0)
        depth_up_right = large_dm_right_arr[ibatch].unsqueeze(0)
        depth_right = small_dm_right_arr[ibatch].unsqueeze(0)
        mask_up_left = gt_input_left["masks_imgsizes"][ibatch, :, :, :]
        mask_left = gt_input_left["masks"][ibatch, :, :, :]
        mask_up_right = gt_input_right["masks_imgsizes"][ibatch, :, :, :]
        mask_right = gt_input_right["masks"][ibatch, :, :, :]
        # Right to Left
        dsc_loss = dsc_loss + losses.depth_stereo_consistency_loss(depth_up_right, depth_up_left, mask_up_right,
                                                                   mask_up_left,
                                                                   pose_target2src, intr_up_left)
        dsc_loss = dsc_loss + losses.depth_stereo_consistency_loss(depth_right, depth_left, mask_right, mask_left,
                                                                   pose_target2src, intr_left)
        # Left to Right
        dsc_loss = dsc_loss + losses.depth_stereo_consistency_loss(depth_up_left, depth_up_right, mask_up_left,
                                                                   mask_up_right,
                                                                   pose_src2target, intr_up_right)
        dsc_loss = dsc_loss + losses.depth_stereo_consistency_loss(depth_left, depth_right, mask_left, mask_right,
                                                                   pose_src2target, intr_right)

    # RGB Stereo Consistency Loss (Just on high res)
    rsc_loss = 0
    for ibatch in range(BV_cur_left.shape[0]):
        if ibatch == 0:
            viz = visualizer
        else:
            viz = None
        if rsc_mul == 0: break
        intr_up_left = model_input_left["intrinsics_up"][ibatch, :, :].unsqueeze(0)
        intr_up_right = model_input_right["intrinsics_up"][ibatch, :, :].unsqueeze(0)
        depth_up_left = large_dm_left_arr[ibatch]
        depth_up_right = large_dm_right_arr[ibatch]
        rgb_up_left = model_input_left["rgb"][ibatch, -1, :, :, :].unsqueeze(0)
        rgb_up_right = model_input_right["rgb"][ibatch, -1, :, :, :].unsqueeze(0)
        mask_up_left = gt_input_left["masks_imgsizes"][ibatch, :, :, :]
        mask_up_right = gt_input_right["masks_imgsizes"][ibatch, :, :, :]
        # Right to Left
        # src_rgb_img, target_rgb_img, target_depth_map, pose_target2src, intr
        rsc_loss = rsc_loss + losses.rgb_stereo_consistency_loss(rgb_up_right, rgb_up_left, depth_up_left,
                                                                 pose_target2src,
                                                                 intr_up_left, viz)
        # Left to Right
        rsc_loss = rsc_loss + losses.rgb_stereo_consistency_loss(rgb_up_left, rgb_up_right, depth_up_right,
                                                                 pose_src2target,
                                                                 intr_up_right)

    # RGB Stereo Consistency Loss (Low res)
    rsc_low_loss = 0
    for ibatch in range(BV_cur_left.shape[0]):
        if ibatch == 0:
            viz = visualizer
        else:
            viz = None
        if rsc_low_mul == 0: break
        intr_left = model_input_left["intrinsics"][ibatch, :, :].unsqueeze(0)
        intr_right = model_input_right["intrinsics"][ibatch, :, :].unsqueeze(0)
        depth_left = small_dm_left_arr[ibatch]
        depth_right = small_dm_right_arr[ibatch]
        rgb_left = F.avg_pool2d(model_input_left["rgb"][ibatch, -1, :, :, :].unsqueeze(0), 4)
        rgb_right = F.avg_pool2d(model_input_right["rgb"][ibatch, -1, :, :, :].unsqueeze(0), 4)
        # Right to Left
        # src_rgb_img, target_rgb_img, target_depth_map, pose_target2src, intr
        rsc_low_loss = rsc_low_loss + losses.rgb_stereo_consistency_loss(rgb_right, rgb_left, depth_left,
                                                                 pose_target2src,
                                                                 intr_left, viz)
        # Left to Right
        rsc_low_loss = rsc_low_loss + losses.rgb_stereo_consistency_loss(rgb_left, rgb_right, depth_right,
                                                                 pose_src2target,
                                                                 intr_right)

    # Smoothness loss (Just on high res)
    smooth_loss = 0
    for ibatch in range(BV_cur_left.shape[0]):
        if smooth_mul == 0: break
        depth_up_left = large_dm_left_arr[ibatch].unsqueeze(0)
        depth_up_right = large_dm_right_arr[ibatch].unsqueeze(0)
        rgb_up_left = model_input_left["rgb"][ibatch, -1, :, :, :].unsqueeze(0)
        rgb_up_right = model_input_right["rgb"][ibatch, -1, :, :, :].unsqueeze(0)
        # Left
        smooth_loss = smooth_loss + losses.edge_aware_smoothness_loss([depth_up_left], rgb_up_left, 1)
        # Right
        smooth_loss = smooth_loss + losses.edge_aware_smoothness_loss([depth_up_right], rgb_up_right, 1)

    # All Loss
    loss = torch.tensor(0.).cuda()

    # Depth Losses
    bsize = torch.tensor(float(BV_cur_left.shape[0] * 2)).cuda()
    if bsize != 0:
        ce_loss = (ce_loss / ce_count) * 1.
        dsc_loss = (dsc_loss / bsize) * dsc_mul
        dc_loss = (dc_loss / bsize) * dc_mul
        rsc_loss = (rsc_loss / bsize) * rsc_mul
        rsc_low_loss = (rsc_low_loss / bsize) * rsc_low_mul
        smooth_loss = (smooth_loss / bsize) * smooth_mul
        loss += (ce_loss + dsc_loss + dc_loss + rsc_loss + rsc_low_loss + smooth_loss)

    # Flow Losses
    if flow_left is not None:
        flow_rgb_loss = (flow_rgb_loss / flow_rgb_count) * flow_rgb_mul
        flow_depth_loss = (flow_depth_loss / flow_depth_count) * flow_depth_mul
        loss += (flow_rgb_loss + flow_depth_loss)

    # Step
    optimizer_KV.zero_grad()
    loss.backward()
    optimizer_KV.step()

    # Return
    return {"loss": loss.detach().cpu().numpy(), "BV_cur_left": BV_cur_left, "BV_cur_right": BV_cur_right}


    # Debug Viz (comment if needed)
    if total_iter % 10 == -1:
        bnum = 0
        img = model_input_left["rgb"][bnum, -1, :, :, :]  # [1,3,256,384]
        img[0, :, :] = img[0, :, :] * kitti.__imagenet_stats["std"][0] + kitti.__imagenet_stats["mean"][0]
        img[1, :, :] = img[1, :, :] * kitti.__imagenet_stats["std"][1] + kitti.__imagenet_stats["mean"][1]
        img[2, :, :] = img[2, :, :] * kitti.__imagenet_stats["std"][2] + kitti.__imagenet_stats["mean"][2]
        img = cv2.cvtColor(img[:, :, :].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
        img = img[:, :, 0]
        ###
        dmap_up_digit = gt_input_left["dmap_imgsize_digits"][bnum, :, :].unsqueeze(0)  # [1,256,384] uint64
        d_candi = local_info_valid["d_candi"]
        dpv = util.digitized_to_dpv(dmap_up_digit, len(d_candi))
        depthmap_quantized = util.dpv_to_depthmap(dpv, d_candi).squeeze(0).numpy()  # [1,256,384]
        depthmap_quantized = depthmap_quantized / 100.
        ###
        dpv_predicted = BV_cur_refined_left[bnum, :, :, :].unsqueeze(0).detach().cpu()
        depthmap_quantized_predicted = util.dpv_to_depthmap(dpv_predicted, d_candi, BV_log=True).squeeze(
            0).numpy()  # [1,256,384]
        depthmap_quantized_predicted = depthmap_quantized_predicted / 100.
        ###
        combined = np.hstack([img, depthmap_quantized, depthmap_quantized_predicted])
        cv2.namedWindow("win")
        cv2.moveWindow("win", 2500, 50)
        cv2.imshow("win", combined)
        cv2.waitKey(15)
    else:
        cv2.waitKey(15)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # global visualizer
        if visualizer is not None: visualizer.kill_received = True
