import os
import pdb
import sys
import math
import time
import argparse
import importlib
import collections

import torch
import open3d
import numpngw
import numpy as np
import open3d as o3d
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import Dataset_pc
from utils.pc_error_wrapper import pc_error
from Chamfer3D.loss_utils import chamfer_rmse
from utils.tools import depth_to_xyz, pc_to_depth, depth_to_pc, mpeg_pcc_dmetric
from utils.pc_save_utils import save_point_cloud_to_file, depth_save_as_pc, depth_xyz_save_as_pc

def parse_args():
    parser = argparse.ArgumentParser('instance_based_pcc')
    parser.add_argument('--dataset_path', type=str,
                        default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    parser.add_argument('--seq_name', type=str, default='00')
    parser.add_argument('--save_seq_name', type=str, default='51')
    parser.add_argument('--depth_width', type=int, default=2048)
    parser.add_argument('--depth_hight', type=int, default=64)
    parser.add_argument('--loss_bit_range', type=int, default=8)
    return parser.parse_args()

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
                scene_idx[row][col] = 1
    return object_idx, scene_idx

def list_avg(in_list):
    return (sum(in_list) / len(in_list))

if __name__ == '__main__':
    args = parse_args()

    TEST_DATASET = Dataset_pc(root = args.dataset_path,
                             seq_name = args.seq_name,
                             depth_width = args.depth_width,
                             depth_hight = args.depth_hight)

    test_loader = DataLoader(TEST_DATASET,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)

    seq_save_path = args.dataset_path + '/' + args.save_seq_name
    for i_bit in range(args.loss_bit_range):
        temp_save_path = Path(seq_save_path)
        temp_save_path.mkdir(exist_ok=True)
        save_file = 'velodyne-' + str(i_bit)
        temp_save_path = temp_save_path.joinpath(save_file)
        temp_save_path.mkdir(exist_ok=True)

    for step, data in tqdm(enumerate(test_loader),
                           total=(len(test_loader)),
                           smoothing=0.9):
        depth = data[0].detach().numpy()
        depth_rem = data[1].detach().numpy()
        depth_sem = data[2].detach().numpy()

        object_idx, scene_idx = distincting_label_object_scene(depth_sem[0])

        object_depth = depth[0] * object_idx
        scene_depth = depth[0] * scene_idx

        depth_rem = depth_rem[0]
        for i in range(64):
            for j in range(2048):
                if depth_rem[i][j] > 0.2:
                    depth_rem[i][j] = depth_rem[i][j] - np.random.randint(10) * 0.005
        depth_rem = np.expand_dims(depth_rem, axis=2)
        object_depth_xyz = depth_to_xyz(object_depth.astype(np.float32))
        scene_depth_int16 = (scene_depth * 255).astype(np.uint16)

        bin_name = str(step)
        while len(bin_name) < 6:
            bin_name = '0' + bin_name

        for i_bit in range(args.loss_bit_range):
            scene_depth_quant = (scene_depth_int16>>i_bit)<<i_bit
            scene_depth_xyz_quant = depth_to_xyz(scene_depth_quant.astype(np.float32)) / 255
            scene_depth_xyz_quant = scene_depth_xyz_quant.astype(np.float32)

            depth_pc = scene_depth_xyz_quant + object_depth_xyz
            depth_pc = depth_pc.astype(np.float32)
            depth_pc_rem = np.concatenate((depth_pc, depth_rem), axis=2)
            pc = depth_pc_rem.flatten()

            seq_save_path_loss_bit = seq_save_path + '/velodyne-' + str(i_bit) + '/'
            bin_save_path = seq_save_path_loss_bit + bin_name + '.bin'
            pc.tofile(bin_save_path)

