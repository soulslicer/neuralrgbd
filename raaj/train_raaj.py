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

# PCL
# from viewer import viewer

# Kill
import sys
import signal

exit = 0


def signal_handler(sig, frame):
    global exit
    exit = 1


signal.signal(signal.SIGINT, signal_handler)

# Data Loading Module
import torch.multiprocessing
from torch.multiprocessing import Process, Queue, Value, cpu_count

visualizer = None


def main():
    import argparse
    print('Parsing the arguments...')
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('--exp_name', required=True, type=str,
                        help='The name of the experiment. Used to naming the folders')
    parser.add_argument('--nepoch', required=True, type=int, help='# of epochs to run')
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
    parser.add_argument('--batch_size', type=int, default=0,
                        help='The batch size for training; default=0, means batch_size=nGPU')
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
    parser.add_argument('--smooth_mul', type=float, default=0., help='Smoothness Loss Multiplier')
    parser.add_argument('--nmode', type=str, default='default', help='Model Network Mode')

    # hack_num = 326
    hack_num = 0
    # print("HACK!!")

    # ==================================================================================== #

    # Arguments Parsing
    args = parser.parse_args()
    nmode = args.nmode
    smooth_mul = args.smooth_mul
    dsc_mul = args.dsc_mul
    dc_mul = args.dc_mul
    rsc_mul = args.rsc_mul
    softce = args.softce
    lc = args.lc
    pytorch_scaling = args.pytorch_scaling
    test_interval = args.test_interval
    velodyne_depth = args.velodyne_depth
    pre_trained_folder = args.pre_trained_folder
    test = args.test
    exp_name = args.exp_name
    saved_model_path = './outputs/saved_models/%s' % (exp_name)
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

    # Checks
    # if softce and not pytorch_scaling: raise('Soft CE needs Pytorch scaling to be turned on?')

    # Linear
    # d_candi = np.linspace(args.d_min, args.d_max, nDepth)
    # d_candi_up = np.linspace(args.d_min, args.d_max, nDepth*4)

    # Quad
    d_candi = util.powerf(args.d_min, args.d_max, nDepth, qpower)
    d_candi_up = util.powerf(args.d_min, args.d_max, nDepth * 4, qpower)

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

    # Writer #
    log_dir = 'outputs/%s/%s' % (args.TB_fldr, exp_name)
    writer = SummaryWriter(log_dir=log_dir, comment='%s' % (exp_name))
    util.save_args(args, '%s/tr_paras.txt' % (log_dir))  # save the training parameters #
    logfile = os.path.join(log_dir, 'log_' + str(time.time()) + '.txt')
    stdout = Logger(logfile)
    sys.stdout = stdout

    if lc:
        # CO Stuff
        sys.path.append("/home/raaj/cmu/lc_ws/src/light_curtain_ros/carla_lc/src/carla_lc/")
        from opt_curtain import CurtainOpt
        lightcurtain = CurtainOpt(False, True)
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

        # https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

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

        # # HACK
        # for items in btest.get_mp():
        #     pass
        # stop

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

    model_path_KV = saved_model_path + "/../" + args.pre_trained_model_path
    if model_path_KV is not '.' and pre_trained:
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
        "smooth_mul": smooth_mul
    }

    # Evaluate Model Graph
    if len(pre_trained_folder):
        import re
        def natural_sort(l):
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)

        all_files = []
        for subdir, dirs, files in os.walk(pre_trained_folder):
            for file in files:
                all_files.append(os.path.join(subdir, file))
        all_files = natural_sort(all_files)
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
        rmses = [];
        rmses_low = [];
        sils = [];
        sils_low = [];
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

    # Load total iter from pretrained

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


def generate_model_input(local_info_valid, camside="left", softce=0):
    # Ensure same size
    valid = (len(local_info_valid[camside + "_cam_intrins"]) == len(
        local_info_valid[camside + "_src_cam_poses"]) == len(local_info_valid["src_dats"]))
    if not valid:
        raise Exception('Batch size invalid')

    # Keep to middle only
    midval = int(len(local_info_valid["src_dats"][0]) / 2)

    # Grab ground truth digitized map
    dmap_imgsize_digit_arr = []
    dmap_digit_arr = []
    dmap_imgsize_arr = []
    dmap_arr = []
    for i in range(0, len(local_info_valid["src_dats"])):
        dmap_imgsize_digit = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_imgsize_digit"]
        dmap_imgsize_digit_arr.append(dmap_imgsize_digit)
        dmap_digit = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap"]
        dmap_imgsize = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_imgsize"]
        dmap = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_raw"]
        dmap_digit_arr.append(dmap_digit)
        dmap_imgsize_arr.append(dmap_imgsize)
        dmap_arr.append(dmap)
    dmap_imgsize_digits = torch.cat(dmap_imgsize_digit_arr)  # [B,256,384] uint64
    dmap_digits = torch.cat(dmap_digit_arr)  # [B,64,96] uint64
    dmap_imgsizes = torch.cat(dmap_imgsize_arr)
    dmaps = torch.cat(dmap_arr)

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
    if softce:
        d_candi = local_info_valid["d_candi"]
        soft_labels_imgsize = []
        soft_labels = []
        variance = softce
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
        "bv_predict": None  # Has to be [B, 64, H, W]
    }

    gt_input = {
        "masks_imgsizes": masks_imgsize.cuda(),
        "masks": masks.cuda(),
        "dmap_imgsize_digits": dmap_imgsize_digits.cuda(),
        "dmap_digits": dmap_digits.cuda(),
        "dmap_imgsizes": dmap_imgsizes,
        "dmaps": dmaps,
        "soft_labels_imgsize": soft_labels_imgsize,
        "soft_labels": soft_labels
    }

    return model_input, gt_input


def testing(model, btest, d_candi, d_candi_up, ngpu, addparams, visualizer, lightcurtain):
    import deval.pyevaluatedepth_lib as dlib
    epsilon = sys.float_info.epsilon
    all_errors = []
    all_errors_low = []
    rsc_sum = 0.
    dc_sum = 0.
    smooth_sum = 0.
    counter = 0.
    start = time.time()

    # Cloud for distance
    dcloud = []
    for m in range(0, 20):
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

            # viz_debug(local_info_valid, visualizer, d_candi, d_candi_up)
            # print("---")
            # continue

            # Create input
            model_input, gt_input = generate_model_input(local_info_valid)
            model_input_right, gt_input_right = generate_model_input(local_info_valid, "right")

            # Create LC output
            # def get_basic(self, baseline=0.2, laser_fov=80, intrinsics=[400., 0., 256, 0., 400., 256., 0., 0., 1.],
            #              width=512, height=512, distortion=[0.000000, 0.000000, 0.000000, 0.000000, 0.000000]):
            if lightcurtain is not None:
                if not lightcurtain.pset:
                    params, sensor_setup = lightcurtain.get_basic(baseline=0.2, laser_fov=80,
                                                                  intrinsics=model_input["intrinsics_up"][0, :,
                                                                             :].cpu().numpy(),
                                                                  width=model_input["rgb"].shape[4],
                                                                  height=model_input["rgb"].shape[3])
                    lightcurtain.load_data(params, sensor_setup)

            # Stuff
            start = time.time()
            model_input["prev_output"] = prev_output
            BV_cur_all_arr, BV_cur_refined_all = torch.nn.parallel.data_parallel(model, model_input, range(ngpu))
            BV_cur_all = BV_cur_all_arr[-1]
            prev_output = BV_cur_all
            # print("Forward: " + str(time.time() - start))

            # Truth
            depthmap_truth_all = gt_input["dmap_imgsizes"]  # [1,256,384]
            depthmap_truth_low_all = gt_input["dmaps"]  # [1,256,384]

            # Masks
            depth_mask_all = gt_input["masks_imgsizes"][:, :, :, :].float()
            depth_mask_low_all = gt_input["masks"][:, :, :, :].float()

            # Batch
            start = time.time()
            bsize = BV_cur_refined_all.shape[0]
            pose_target2src = local_info_valid["T_left2right"].unsqueeze(0).cuda()
            for b in range(bsize):
                BV_cur = BV_cur_all[b, :, :, :].unsqueeze(0)
                BV_cur_refined = BV_cur_refined_all[b, :, :, :].unsqueeze(0)
                depthmap_truth = depthmap_truth_all[b, :, :].unsqueeze(0)
                depthmap_truth_low = depthmap_truth_low_all[b, :, :].unsqueeze(0)
                depth_mask = depth_mask_all[b, :, :, :]
                depth_mask_low = depth_mask_low_all[b, :, :, :]
                intr = model_input["intrinsics_up"][b, :, :].unsqueeze(0)
                left_rgb = model_input["rgb"][b, -1, :, :, :].unsqueeze(0)
                right_rgb = model_input_right["rgb"][b, -1, :, :, :].unsqueeze(0)

                # Predicted
                dpv_low_predicted = BV_cur[0, :, :, :].unsqueeze(0).detach()
                dpv_predicted = BV_cur_refined[0, :, :, :].unsqueeze(0).detach()
                depthmap_predicted = util.dpv_to_depthmap(dpv_predicted, d_candi, BV_log=True)  # [1,256,384]
                depthmap_low_predicted = util.dpv_to_depthmap(dpv_low_predicted, d_candi, BV_log=True)  # [1,256,384]

                # Generate UField

                # Losses
                rsc_sum += losses.rgb_stereo_consistency_loss(right_rgb, left_rgb, depthmap_predicted,
                                                              pose_target2src, intr)
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
                errors = dlib.depthError(depthmap_predicted_np + epsilon, depthmap_truth_np + epsilon)
                errors_low = dlib.depthError(depthmap_predicted_low_np + epsilon, depthmap_truth_low_np + epsilon)
                all_errors.append(errors)
                all_errors_low.append(errors_low)

                # Viz
                if visualizer is not None and b == 0:
                    intr_up = model_input["intrinsics_up"][b, :, :].cpu()
                    intr = model_input["intrinsics"][b, :, :].cpu()
                    depthmap_predicted_np = (depthmap_predicted * 1).cpu()
                    depthmap_low_predicted_np = (depthmap_low_predicted * 1).cpu()
                    depthmap_predicted_np[:, 0:depthmap_predicted_np.shape[1] / 2, :] = 0
                    depthmap_low_predicted_np[:, 0:depthmap_low_predicted_np.shape[1] / 2, :] = 0

                    # Visualize side Cloud
                    # subslice = util.dpv_to_xyz(torch.exp(dpv_low_predicted), d_candi, intr, 3, 1)  # torch.Size([24576, 4])
                    subslice = util.dpv_to_xyz(torch.exp(dpv_predicted), d_candi, intr_up, 10,
                                               4)  # torch.Size([24576, 4])
                    slicecloud = np.zeros((subslice.shape[0], 9)).astype(np.float32)
                    slicecloud[:, 0:3] = subslice[:, 0:3]
                    subslice[:, 3] = (subslice[:, 3] - torch.min(subslice[:, 3])) / (
                            torch.max(subslice[:, 3]) - torch.min(subslice[:, 3]))
                    slicecloud[:, 3] = subslice[:, 3] * 255
                    slicecloud[:, 4] = 0
                    slicecloud[:, 5] = 50

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

                    # Light Curtain
                    if lightcurtain is not None:
                        arc = lightcurtain.get_arc(22)
                        lccloud, npimgs = lightcurtain.compute([arc], [depthmap_truth_np])
                        lccloud = np.append(lccloud, np.zeros((lccloud.shape[0], 5)), axis=1)
                        lccloud[:, 4:6] = 50
                        lccloud = util.hack(lccloud)
                        visualizer.addCloud(lccloud, 2)

                    # Cloud
                    cloud_low_orig = util.tocloud(depthmap_low_predicted_np, img_low, intr, None)
                    cloud_orig = util.tocloud(depthmap_predicted_np, img, intr_up, None)
                    cloud_truth = util.tocloud(torch.tensor(depthmap_truth_np[np.newaxis, :]), img, intr_up, None)
                    cloud_low_truth = util.tocloud(torch.tensor(depthmap_truth_low_np[np.newaxis, :]), img_low, intr,
                                                   None)
                    cv2.imshow("win", combined)
                    print(cloud_orig.shape)
                    print(slicecloud.shape)
                    # visualizer.addCloud(cloud_low_truth, 3)
                    # visualizer.addCloud(cloud_truth,1)
                    # visualizer.addCloud(cloud_low_orig, 3)
                    visualizer.addCloud(cloud_orig, 1)
                    #visualizer.addCloud(slicecloud, 2)
                    visualizer.addCloud(dcloud, 4)
                    visualizer.swapBuffer()
                    key = cv2.waitKey(0)
                    print("--")

        else:
            pass

    results = dlib.evaluateErrors(all_errors)
    results_low = dlib.evaluateErrors(all_errors_low)
    rsc_sum /= counter
    smooth_sum /= counter
    dc_sum /= counter
    return results, results_low, rsc_sum, smooth_sum, dc_sum


def train(model, optimizer_KV, local_info_valid, ngpu, addparams, total_iter, prev_output, visualizer, lightcurtain):
    # start = time.time()
    # print(time.time() - start)
    # return {"loss": 0}

    # Readout AddParams
    d_candi = local_info_valid["d_candi"]
    softce = addparams["softce"]
    dsc_mul = addparams["dsc_mul"]
    dc_mul = addparams["dc_mul"]
    rsc_mul = addparams["rsc_mul"]
    smooth_mul = addparams["smooth_mul"]

    # Create inputs
    model_input_left, gt_input_left = generate_model_input(local_info_valid, "left", softce)
    model_input_right, gt_input_right = generate_model_input(local_info_valid, "right", softce)

    # Previous
    if prev_output is not None:
        model_input_left["prev_output"] = prev_output["BV_cur_left"]
        model_input_right["prev_output"] = prev_output["BV_cur_right"]
    else:
        model_input_left["prev_output"] = None
        model_input_right["prev_output"] = None

    # Run Forward
    BV_cur_left_array, BV_cur_refined_left = torch.nn.parallel.data_parallel(model, model_input_left, range(ngpu))
    BV_cur_right_array, BV_cur_refined_right = torch.nn.parallel.data_parallel(model, model_input_right, range(ngpu))

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

    # Backward
    bsize = torch.tensor(float(BV_cur_left.shape[0] * 2)).cuda()
    optimizer_KV.zero_grad()
    ce_loss = (ce_loss / ce_count) * 1.
    dsc_loss = (dsc_loss / bsize) * dsc_mul
    dc_loss = (dc_loss / bsize) * dc_mul
    rsc_loss = (rsc_loss / bsize) * rsc_mul
    smooth_loss = (smooth_loss / bsize) * smooth_mul
    loss = ce_loss + dsc_loss + dc_loss + rsc_loss + smooth_loss
    loss.backward()
    optimizer_KV.step()

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

    # Return
    return {"loss": ce_loss.detach().cpu().numpy(), "BV_cur_left": BV_cur_left.detach(), "BV_cur_right": BV_cur_right.detach()}


class BatchSchedulerMP:
    def __init__(self, inputs, mode):
        self.mode = mode
        self.inputs = inputs
        self.queue = Queue()
        self.control = Value('i', 1)
        if self.mode == 0:
            self.process = Process(target=self.worker, args=(self.inputs, self.queue, self.control))
            self.process.start()
        # self.worker(self.inputs, self.queue, self.control)

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
        mode = inputs["mode"]
        pytorch_scaling = inputs["pytorch_scaling"]
        velodyne_depth = inputs["velodyne_depth"]
        if mode == "train":
            split_txt = './kitti_split/training.txt'
        elif mode == "val":
            split_txt = './kitti_split/testing.txt'

        dataset_init = kitti.KITTI_dataset
        fun_get_paths = lambda traj_indx: kitti.get_paths(traj_indx, split_txt=split_txt,
                                                          mode=mode,
                                                          database_path_base=dataset_path, t_win=t_win_r)

        # Load Dataset
        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)
        fldr_path, img_paths, dmap_paths, poses, intrin_path = fun_get_paths(0)
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path=intrin_path, img_size=img_size, digitize=True,
                               d_candi=d_candi, d_candi_up=d_candi_up, resize_dmap=.25,
                               crop_w=crop_w, velodyne_depth=velodyne_depth, pytorch_scaling=pytorch_scaling)
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

                    # print(batch_idx, frame_count)
                    if broken: break

                if broken: break
                BatchScheduler.proceed_batch()

            if broken: break
        queue.put(None)
        time.sleep(1)

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

                    # if frame_count == 0:
                    #   print(local_info["src_dats"][0][0]["left_camera"]["img_path"])

                    # Put in Q
                    yield [local_info, len(BatchScheduler), batch_idx, frame_count, ref_indx, iepoch]

                    # Update dat_array
                    if frame_count < BatchScheduler.traj_len - 1:
                        BatchScheduler.proceed_frame()

                    # print(batch_idx, frame_count)
                    if broken: break

                if broken: break
                BatchScheduler.proceed_batch()

            if broken: break
        yield None


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # global visualizer
        if visualizer is not None: visualizer.kill_received = True
