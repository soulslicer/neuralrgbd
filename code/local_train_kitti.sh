#'''
#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license
#
#author: Chao Liu <chaoliu1@cs.cmu.edu>
#'''

#!/bin/bash

python train_KVNet.py \
    --batch_size 2 \
		--exp_name kitti/ \
		--nepoch 20 \
		--RNet \
		--sigma_soft_max 10\
		--LR 1e-5 \
		--t_win 2 \
		--d_min 1 \
		--d_max 60 \
		--feature_dim 64 \
		--ndepth 64 \
		--grad_clip \
		--grad_clip_max 2. \
		--dataset kitti \
		--ngpu 1 \
		--dataset_path ../data/datasets/kitti/ \
