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

from utils.pc_error_wrapper import pc_error
from Chamfer3D.loss_utils import chamfer_rmse
from dataset import Dataset_MM, depth_xyz_to_pc
from utils.ground_simulation import get_ground_depth
from utils.pc_save_utils import save_point_cloud_to_file
from utils.tools import depth_to_pc, depth_to_xyz, remove_balck_line_and_remote_points

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

def eval_pc_rmse(pc_rec, pc_ori):
    torch_pc_rec = torch.from_numpy(pc_rec).float().cuda()
    torch_pc_ori = torch.from_numpy(pc_ori).float().cuda()
    _, rmse = chamfer_rmse(torch_pc_rec.unsqueeze(0),
                           torch_pc_ori.unsqueeze(0))
    return rmse

def distincting_ground(depth_sem):
    ground_labels = [1, 40, 44, 48 , 49]
    H = len(depth_sem)
    W = len(depth_sem[0])
    ground_idx = np.zeros((H, W))
    for row in range(H):
        for col in range(W):
            if np.isin(depth_sem[row][col], ground_labels):
                ground_idx[row][col] = 1
    return ground_idx

def pc_total_range(pc, range_K):
    pc_cut = np.zeros((100000, 3))
    pc_cut_idx = 0
    range_max = range_K**2
    for i in range(len(pc)):
        if ((pc[i][0]*pc[i][0]+pc[i][1]*pc[i][1]<range_max) and pc_cut_idx < 100000):
            pc_cut[pc_cut_idx] = pc[i]
            pc_cut_idx += 1
    pc_cut = pc_cut.astype(np.float32)
    return pc_cut

def list_avg(in_list):
    return (sum(in_list) / len(in_list))

if __name__ == '__main__':
    args = parse_args()

    TEST_DATASET = Dataset_MM(root = args.dataset_path,
                             seq_name = args.seq_name,
                             depth_width = args.depth_width,
                             depth_hight = args.depth_hight)
    test_loader = DataLoader(TEST_DATASET,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)

    list_range = [5, 10, 15, 25, 35, 50, 100]
    all_lossly_rmse = np.zeros((len(list_range), len(test_loader)))  # cm

    sg_depth = get_ground_depth()

    for step, data in tqdm(enumerate(test_loader),
                           total=(len(test_loader)),
                           smoothing=0.9):
        data_depth = data[0].detach().numpy()
        data_xyz = data[1].detach().numpy()
        data_sem = data[2].detach().numpy()

        ground_idx = distincting_ground(data_sem[0])

        ground_depth = data_depth[0] * ground_idx
        gs_depth = sg_depth * ground_idx

        depth_g_int16 = (ground_depth*255).astype(np.uint16)
        depth_gs_int16 = (gs_depth*255).astype(np.uint16)

        pc_g = depth_to_pc(depth_g_int16.astype(np.float32)) / 255
        pc_gs = depth_to_pc(depth_gs_int16.astype(np.float32)) / 255

        idx = 0
        for i_range in list_range:
            pc_g_range = pc_total_range(pc_g, i_range)
            pc_gs_range = pc_total_range(pc_gs, i_range)
            rmse = eval_pc_rmse(pc_g_range, pc_gs_range)
            all_lossly_rmse[idx][step] = rmse
            idx = idx+1

    for idx in range(len(list_range)):
        print(list_avg(all_lossly_rmse[idx]))

