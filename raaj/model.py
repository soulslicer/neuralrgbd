'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

'''
The full KV-net framework,
support (still in progress) multiple-gpu training
'''

import torch

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from psm_submodule import *
import warping.homography as warp_homo
import math
import time

import submodels
import util

class KVNET(nn.Module):
    r'''
    Inside this module, we will do the full KV-Net pipeline:
    * D-Net (feature extractiion + BV_cur estimation )
    * KV-Net 
    '''
    def __init__(self, feature_dim, cam_intrinsics, d_candi, d_candi_up, sigma_soft_max,
                 KVNet_feature_dim, d_upsample_ratio_KV_net, 
                 if_refined = True, refine_channel = 3, if_upsample_d = False, drefine = "",
                 nmode = "default"):

        super(KVNET, self).__init__()
        self.feature_dim = feature_dim
        self.KVNet_feature_dim = KVNet_feature_dim
        self.sigma_soft_max = sigma_soft_max
        self.d_upsample_ratio_KV_net = d_upsample_ratio_KV_net
        self.d_candi = d_candi
        self.if_refined = if_refined
        self.if_upsample_d = if_upsample_d
        self.nmode = nmode

        # D Net for Feature Extraction and Cost Volume Generation
        self.feature_extractor = submodels.feature_extractor(feature_dim = feature_dim, multi_scale=True)
        self.d_net = submodels.D_NET_BASIC(
                self.feature_extractor, cam_intrinsics,
                d_candi, sigma_soft_max, use_img_intensity=True,
                BV_log = True, output_features = True, drefine = drefine)

        # R Net for Upsampling
        self.r_net = submodels.RefineNet_DPV_upsample(
                int(self.feature_dim), int(self.feature_dim/2), 3,
                D = len(self.d_candi), upsample_D=self.if_upsample_d )

        # Refine DPV further
        self.kv_net = None

        # print #
        print('KV-Net initialization:')

    def forward(self, model_input):
        # gpuID = torch.zeros((1)).cuda().get_device()

        # "intrinsics": intrinsics, # [B, 3, 3]
        # "unit_ray": unit_ray, [B, 3, 6144]
        # "src_cam_poses": src_cam_poses, [B, 2, 4, 4]
        # "rgb": rgb [4, 2, 3,256,384]

        if self.nmode == "default":

            # Compute the cost volume and get features
            BV_cur, cost_volumes, d_net_features = self.d_net(model_input)
            d_net_features.append(model_input["rgb"][:,-1,:,:,:])
            # 64 in feature Dim depends on the command line arguments
            # [B, 128, 64, 96] - has log on it [[B,64,64,96] [B,32,128,192] [B,3,256,384]]

            # Should we just add the knet for some 3D Conv? or some version of it
            BV_cur_array = [BV_cur]

            # Make sure size is still correct here!
            BV_cur_refined = self.r_net(torch.exp(BV_cur_array[-1]), img_features=d_net_features)
            # [B,128,256,384]

            return BV_cur_array, BV_cur_refined

        elif self.nmode == "irefine":
            # Variables
            down_sample_rate = 4.
            bsize = model_input["rgb"].shape[0]
            kvnet_size = model_input["rgb"].shape[1]*3 + 1

            # Setup KV Net
            if self.kv_net == None:
                self.kv_net = submodels.KV_NET_BASIC(kvnet_size, dres_count = 4, feature_dim = 32).cuda()

            # Compute the cost volume and get features
            BV_cur_base, cost_volumes, d_net_features = self.d_net(model_input)
            d_net_features.append(model_input["rgb"][:,-1,:,:,:])
            # [B, 128, 64, 96] - has log on it [[B,64,64,96] [B,32,128,192] [B,3,256,384]]

            # Construct Input for network (Combine cost volume and image)
            kvnet_volumes = []
            for i in range(0, bsize):
                Rs_src = [pose[:3, :3] for pose in model_input["src_cam_poses"][i,:,:,:]]
                ts_src = [pose[:3, 3] for pose in model_input["src_cam_poses"][i,:,:,:]]
                ref_frame_dw = F.avg_pool2d(model_input["rgb"][i,-1,:,:,:].unsqueeze(0), int(down_sample_rate))
                src_frames_dw = [F.avg_pool2d(src_frame_.unsqueeze(0), int(down_sample_rate))
                                 for src_frame_ in model_input["rgb"][i,0:-1,:,:,:]]
                WAPRED_src_frames = warp_homo.warp_img_feats_mgpu(src_frames_dw, self.d_candi, Rs_src, ts_src,
                                                                  model_input["intrinsics"][i,:,:].unsqueeze(0),
                                                                  model_input["unit_ray"][i,:,:].unsqueeze(0))
                WAPRED_src_frames = torch.cat(tuple(WAPRED_src_frames), dim=0)
                ref_frame_dw_rep = torch.transpose(ref_frame_dw.repeat([len(self.d_candi), 1, 1, 1]), 0, 1)  # [3,64,64,96]
                kvnet_in_vol = torch.cat((WAPRED_src_frames, ref_frame_dw_rep,
                                          cost_volumes[i,:,:,:].unsqueeze(0)),dim=0).unsqueeze(0) #[1,16,64,64,96]
                kvnet_volumes.append(kvnet_in_vol)
            kvnet_in_vol = torch.cat(kvnet_volumes) # torch.Size([B, 9, 64, 64, 96])

            # K Net
            BV_cur = self.kv_net(kvnet_in_vol, True).squeeze(1)

            # Should we just add the knet for some 3D Conv? or some version of it
            BV_cur_array = [BV_cur_base, BV_cur]

            # Make sure size is still correct here!
            BV_cur_refined = self.r_net(torch.exp(BV_cur_array[-1]), img_features=d_net_features)
            # [B,128,256,384]

            return BV_cur_array, BV_cur_refined

        elif self.nmode == "irefine_feedback":
            # Variables
            down_sample_rate = 4.
            bsize = model_input["rgb"].shape[0]
            kvnet_size = model_input["rgb"].shape[1]*3 + 2

            # Setup KV Net
            if self.kv_net == None:
                self.kv_net = submodels.KV_NET_BASIC(kvnet_size, dres_count = 2, feature_dim = 32).cuda()

            # Compute the cost volume and get features
            BV_cur_base, cost_volumes, d_net_features = self.d_net(model_input)
            d_net_features.append(model_input["rgb"][:,-1,:,:,:])
            # [B, 128, 64, 96] - has log on it [[B,64,64,96] [B,32,128,192] [B,3,256,384]]

            # CAN WE MAKE THE FEEDBACK MODE KICK IN LATER?
            # Yes simply set BV_prev to None until actual iteration wanted is reached (this is bad)

            # Prev Output is None
            BV_prev = model_input["prev_output"]
            if BV_prev is None:

                # Should we just add the knet for some 3D Conv? or some version of it
                BV_cur_array = [BV_cur_base]

                # Make sure size is still correct here!
                BV_cur_refined = self.r_net(torch.exp(BV_cur_array[-1]), img_features=d_net_features)
                # [B,128,256,384]

                return BV_cur_array, BV_cur_refined

            # Prev Output is nont None
            else:

                # Construct Input for network (Combine cost volume and image)
                kvnet_volumes = []
                for i in range(0, bsize):
                    Rs_src = [pose[:3, :3] for pose in model_input["src_cam_poses"][i, :, :, :]]
                    ts_src = [pose[:3, 3] for pose in model_input["src_cam_poses"][i, :, :, :]]
                    ref_frame_dw = F.avg_pool2d(model_input["rgb"][i, -1, :, :, :].unsqueeze(0), int(down_sample_rate))
                    src_frames_dw = [F.avg_pool2d(src_frame_.unsqueeze(0), int(down_sample_rate))
                                     for src_frame_ in model_input["rgb"][i, 0:-1, :, :, :]]
                    WAPRED_src_frames = warp_homo.warp_img_feats_mgpu(src_frames_dw, self.d_candi, Rs_src, ts_src,
                                                                      model_input["intrinsics"][i, :, :].unsqueeze(0),
                                                                      model_input["unit_ray"][i, :, :].unsqueeze(0))
                    WAPRED_src_frames = torch.cat(tuple(WAPRED_src_frames), dim=0)
                    ref_frame_dw_rep = torch.transpose(ref_frame_dw.repeat([len(self.d_candi), 1, 1, 1]), 0,
                                                       1)  # [3,64,64,96]
                    kvnet_in_vol = torch.cat((WAPRED_src_frames, ref_frame_dw_rep,
                                              cost_volumes[i, :, :, :].unsqueeze(0),
                                              BV_prev[i,:,:,:].unsqueeze(0)), dim=0).unsqueeze(
                        0)  # [1,16,64,64,96]
                    kvnet_volumes.append(kvnet_in_vol)
                kvnet_in_vol = torch.cat(kvnet_volumes)  # torch.Size([B, 9, 64, 64, 96])

                # K Net
                BV_cur = self.kv_net(kvnet_in_vol, True).squeeze(1)

                # Should we just add the knet for some 3D Conv? or some version of it
                BV_cur_array = [BV_cur_base, BV_cur]

                # Make sure size is still correct here!
                BV_cur_refined = self.r_net(torch.exp(BV_cur_array[-1]), img_features=d_net_features)
                # [B,128,256,384]

                return BV_cur_array, BV_cur_refined