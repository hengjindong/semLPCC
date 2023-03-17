import os
import pdb
import sys
import math
import time
import argparse
import importlib
import collections

import open3d
import torch
import numpngw
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dataset_MM, depth_xyz_to_pc
from utils.pc_error_wrapper import pc_error
from utils.tools import depth_to_xyz
from utils.pc_save_utils import save_point_cloud_to_file
from utils.ground_simulation import get_ground_depth

def parse_args():
    parser = argparse.ArgumentParser('instance_based_pcc')
    parser.add_argument('--dataset_path', type=str,default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    parser.add_argument('--seq_name', type=str, default='00')
    parser.add_argument('--depth_width', type=int, default=2048)
    parser.add_argument('--depth_hight', type=int, default=64)
    return parser.parse_args()

def depth_save_as_pc(ply_path, depth):
    depth_xyz = depth_to_xyz(depth)
    valid_idx = np.where(np.sum(depth_xyz, -1) != 0)
    point_cloud = depth_xyz[valid_idx]
    save_point_cloud_to_file(ply_path, point_cloud)

def depth_xyz_save_as_pc(ply_path, depth_xyz):
    valid_idx = np.where(np.sum(depth_xyz, -1) != 0)
    point_cloud = depth_xyz[valid_idx]
    save_point_cloud_to_file(ply_path, point_cloud)

def distincting_label_object_scene(depth_sem):
    object_labels = [10, 11, 13, 15, 16, 18, 20, 30, 31, 32,
                     252, 253, 254, 255, 256, 257, 258, 259]
    ground_labels = [1, 40, 44, 48 , 49]
    H = len(depth_sem)
    W = len(depth_sem[0])
    object_idx = np.zeros((H, W))
    scene_idx = np.zeros((H, W))
    for row in range(H):
        for col in range(W):
            if np.isin(depth_sem[row][col], object_labels):
                object_idx[row][col] = 1
            else:
                if not np.isin(depth_sem[row][col], ground_labels):
                    scene_idx[row][col] = 1
    return object_idx, scene_idx

if __name__ == '__main__':
    args = parse_args()
    # save 100
    TEST_DATASET = Dataset_MM(root = args.dataset_path,
                             seq_name = args.seq_name,
                             depth_width = args.depth_width,
                             depth_hight = args.depth_hight)
    test_loader = DataLoader(TEST_DATASET,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)

    ground_depth = get_ground_depth()

    for step, data in tqdm(enumerate(test_loader),
                           total=(len(test_loader)),
                           smoothing=0.9):
        data_depth = data[0].detach().numpy()
        depth_save_as_pc('./bs_file/seq_file/'+args.seq_name+'/ply/'+str(step)+'.ply', data_depth[0])

        depth_16bit = (data_depth[0]*255).astype(np.uint16)
        depth_high = (depth_16bit>>8).astype(np.uint8)
        depth_low = depth_16bit.astype(np.uint8)


        # numpngw.write_png('./bs_file/seq_file/'+args.seq_name+'/png_16bit/'+str(step)+'.png', depth_16bit)
        numpngw.write_png('./bs_file/Compare_R/'+args.seq_name+'/png_high/'+str(step)+'.png', depth_high)
        numpngw.write_png('./bs_file/Compare_R/'+args.seq_name+'/png_low/'+str(step)+'.png', depth_low)

