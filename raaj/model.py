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
        self.modA = None
        self.modB = None

        # Flow
        self.flowA = ABlock3x3(int(self.feature_dim)*2, 3, 64, 64*4, C=4, BN=True).cuda()
        self.flownet_upsample = submodels.FlowNet_DPV_upsample(
                int(self.feature_dim), int(self.feature_dim/2), 3,
                D = 3, upsample_D=self.if_upsample_d )

        # Other
        self.simpleA = SimpleBlockA(In_D=len(self.d_candi), Out_D=len(self.d_candi), Depth=64, C=2, mode="default")

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
            BV_cur, cost_volumes, d_net_features, _ = self.d_net(model_input)
            d_net_features.append(model_input["rgb"][:,-1,:,:,:])
            # 64 in feature Dim depends on the command line arguments
            # [B, 128, 64, 96] - has log on it [[B,64,64,96] [B,32,128,192] [B,3,256,384]]

            # Make sure size is still correct here!
            BV_cur_refined = self.r_net(torch.exp(BV_cur), img_features=d_net_features)
            # [B,128,256,384]

            return [BV_cur], [BV_cur_refined], None, None

        elif self.nmode == "defaultrefine":

            # Compute the cost volume and get features
            BV_cur, cost_volumes, d_net_features, _ = self.d_net(model_input)
            d_net_features.append(model_input["rgb"][:,-1,:,:,:])
            # 64 in feature Dim depends on the command line arguments
            # [B, 128, 64, 96] - has log on it [[B,64,64,96] [B,32,128,192] [B,3,256,384]]

            # Add Block
            sa_output = self.simpleA(BV_cur)
            BV_cur_mod = F.log_softmax(sa_output, dim=1)

            # Make sure size is still correct here!
            BV_cur_refined = self.r_net(torch.exp(BV_cur_mod), img_features=d_net_features)
            # [B,128,256,384]

            return [BV_cur, BV_cur_mod], [BV_cur_refined], None, None

        elif self.nmode == "reupsample":

            # Compute the cost volume and get features
            BV_cur, cost_volumes, d_net_features, _ = self.d_net(model_input)
            d_net_features.append(model_input["rgb"][:,-1,:,:,:])
            # 64 in feature Dim depends on the command line arguments
            # [B, 128, 64, 96] - has log on it [[B,64,64,96] [B,32,128,192] [B,3,256,384]]

            # Make sure size is still correct here!
            BV_cur_refined = self.r_net(torch.exp(BV_cur), img_features=d_net_features)
            # [B,128,256,384]

            # Downsample
            dsize = [BV_cur_refined.shape[2]/4, BV_cur_refined.shape[3]/4]
            BV_cur_downsampled = F.interpolate(BV_cur_refined, size=dsize, mode='nearest')

            # Fuse?
            BV_cur = torch.log(torch.clamp(torch.exp(BV_cur), util.epsilon, 1.))
            BV_cur_downsampled = torch.log(torch.clamp(torch.exp(BV_cur_downsampled), util.epsilon, 1.))
            fused_dpv = torch.exp(BV_cur + BV_cur_downsampled)
            fused_dpv = fused_dpv / torch.sum(fused_dpv, dim=1).unsqueeze(1)
            fused_dpv = torch.clamp(fused_dpv, util.epsilon, 1.)
            BV_fused = torch.log(fused_dpv)

            # Reupsample
            BV_fused_refined = self.r_net(torch.exp(BV_fused), img_features=d_net_features)

            return [BV_cur, BV_fused], [BV_cur_refined, BV_fused_refined], None, None

        elif self.nmode == "lhack":

            # Compute the cost volume and get features
            BV_cur, cost_volumes, d_net_features, _ = self.d_net(model_input)
            d_net_features.append(model_input["rgb"][:,-1,:,:,:])
            # 64 in feature Dim depends on the command line arguments
            # [B, 128, 64, 96] - has log on it [[B,64,64,96] [B,32,128,192] [B,3,256,384]]

            tofuse_dpv = []
            truth_var = torch.tensor(0.3)
            for b in range(0, model_input["dmaps"].shape[0]):
                dmap = model_input["dmaps"][b,:,:]
                mask = model_input["masks"][b,0,:,:].unsqueeze(0)
                mask_inv = 1. - mask
                truth_dpv = util.gen_soft_label_torch(model_input["d_candi"], dmap, truth_var, zero_invalid=True)
                uni_dpv = util.gen_uniform(model_input["d_candi"], dmap)
                modified_dpv = truth_dpv*mask + uni_dpv*mask_inv
                tofuse_dpv.append(modified_dpv.unsqueeze(0))
            tofuse_dpv = torch.cat(tofuse_dpv)
            tofuse_dpv = torch.clamp(tofuse_dpv, util.epsilon, 1.)

            fused_dpv = torch.exp(BV_cur + torch.log(tofuse_dpv))
            fused_dpv = fused_dpv / torch.sum(fused_dpv, dim=1).unsqueeze(1)
            fused_dpv = torch.clamp(fused_dpv, util.epsilon, 1.)
            BV_cur = torch.log(fused_dpv)

            # Make sure size is still correct here!
            BV_cur_refined = self.r_net(fused_dpv, img_features=d_net_features)
            # [B,128,256,384]

            return [BV_cur], [BV_cur_refined], None, None

        elif self.nmode == "irefine":
            # Variables
            down_sample_rate = 4.
            bsize = model_input["rgb"].shape[0]
            kvnet_size = model_input["rgb"].shape[1]*3 + 1

            # Setup KV Net
            if self.kv_net == None:
                self.kv_net = submodels.KV_NET_BASIC(kvnet_size, dres_count = 2, feature_dim = 32).cuda()

            # Compute the cost volume and get features
            BV_cur_base, cost_volumes, d_net_features, _ = self.d_net(model_input)
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

            return BV_cur_array, BV_cur_refined, None, None

        elif self.nmode == "irefine_feedback":
            # Variables
            down_sample_rate = 4.
            bsize = model_input["rgb"].shape[0]
            kvnet_size = model_input["rgb"].shape[1]*3 + 2

            # Setup KV Net
            if self.kv_net == None:
                self.kv_net = submodels.KV_NET_BASIC(kvnet_size, dres_count = 2, feature_dim = 32).cuda()

            # Compute the cost volume and get features
            BV_cur_base, cost_volumes, d_net_features, _ = self.d_net(model_input)
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

                return BV_cur_array, BV_cur_refined, None, None

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

                return BV_cur_array, BV_cur_refined, None, None

        # Flow
        elif self.nmode == "flow_b":
            # Compute the cost volume and get features
            BV_cur, cost_volumes, last_features, first_features = self.d_net(model_input)
            last_features.append(model_input["rgb"][:,-1,:,:,:])
            first_features.append(model_input["rgb"][:,0,:,:,:])

            # Flow
            flow_input = torch.cat([first_features[0], last_features[0]], dim=1)
            flow_a = self.flowA(flow_input)

            # Upsample Flow
            flow_upsampled = self.flownet_upsample(flow_a, last_features)

            # Other
            BV_cur_refined = self.r_net(torch.exp(BV_cur), img_features=last_features)

            # Return
            return [BV_cur], [BV_cur_refined], flow_a, flow_upsampled

            #return [torch.zeros((0))], [torch.zeros((0))], flow_a, flow_upsampled


            pass


        # I need a mechanism to make a reasoanble prediction