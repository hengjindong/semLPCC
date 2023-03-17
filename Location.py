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
from torch.utils.data import DataLoader

from dataset import Dataset_ME
from utils.pc_error_wrapper import pc_error
from Chamfer3D.loss_utils import chamfer_rmse
from utils.tools import depth_to_xyz, pc_to_depth, depth_to_pc, mpeg_pcc_dmetric
from utils.pc_save_utils import save_point_cloud_to_file, depth_save_as_pc, depth_xyz_save_as_pc

def parse_args():
    parser = argparse.ArgumentParser('instance_based_pcc')
    parser.add_argument('--dataset_path', type=str,
                        default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    parser.add_argument('--seq_name', type=str, default='00')
    parser.add_argument('--seq_len', type=int, default=3)
    parser.add_argument('--depth_width', type=int, default=2048)
    parser.add_argument('--depth_hight', type=int, default=64)
    parser.add_argument('--scene_interval', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=1)
    parser.add_argument('--border_num', type=int, default=64)
    parser.add_argument('--loss_bit_range', type=int, default=12)
    return parser.parse_args()

def list_avg(in_list):
    return (sum(in_list) / len(in_list))

def np_depth_to_o3d_pc(depth):
    depth_xyz = depth_to_xyz(depth)
    valid_idx = np.where(np.sum(depth_xyz, -1) != 0)
    point_cloud = depth_xyz[valid_idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    return pcd

def o3d_pc_save_as_pc(save_path, pcd):
    point_cloud = np.asarray(pcd.points)
    save_point_cloud_to_file(save_path, point_cloud)

def pc_border_res(pc_source, pc_target, border_num, loss_bit):
    if type(pc_source) == type(o3d.geometry.PointCloud()):
        pc_source = np.asarray(pc_source.points)
    if type(pc_target) == type(o3d.geometry.PointCloud()):
        pc_target = np.asarray(pc_target.points)
    depth_source = pc_to_depth(pc_source)
    depth_target = pc_to_depth(pc_target)

    depth_source = (depth_source * 255).astype(np.uint16)
    depth_target = (depth_target * 255).astype(np.uint16)
    depth_border_res = depth_target - depth_source

    H = len(depth_source)
    W = len(depth_source[0])
    for col in range(H):
        for raw in range(W):
            if col > (border_num - 1):
                depth_border_res[col][raw] = 0
    depth_border_res = (depth_border_res>>loss_bit)<<loss_bit

    depth_source_add_res = depth_source + depth_border_res

    depth_source_add_res = depth_source_add_res.astype(np.float32)
    depth_target = depth_target.astype(np.float32)
    pc_source_add_res = depth_to_pc(depth_source_add_res) / 255
    pc_target = depth_to_pc(depth_target) / 255
    return depth_border_res, pc_source_add_res, pc_target

def eval_depth_rate(depth):
    save_png_path = './temp/temp_Inter_ME_' + \
        str(args.border_num) + \
        str(np.random.randint(10)) + '.png'
    numpngw.write_png(save_png_path, depth)
    fsize_B = os.path.getsize(save_png_path)
    # pc_num = len(depth) * len(depth[0])
    pc_num = 120000

    fsize_bit = fsize_B * 8
    bpp = fsize_bit / pc_num
    return bpp, fsize_bit

def eval_pc_rmse(pc_rec, pc_ori):
    torch_pc_rec = torch.from_numpy(pc_rec).float().cuda()
    torch_pc_ori = torch.from_numpy(pc_ori).float().cuda()
    _, rmse = chamfer_rmse(torch_pc_rec.unsqueeze(0),
                           torch_pc_ori.unsqueeze(0))
    return rmse

if __name__ == '__main__':
    args = parse_args()

    TEST_DATASET = Dataset_ME(root = args.dataset_path,
                             seq_name = args.seq_name,
                             seq_len = args.seq_len,
                             depth_width = args.depth_width,
                             depth_hight = args.depth_hight,
                             scene_interval = args.scene_interval)
    test_loader = DataLoader(TEST_DATASET,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)

    all_time = np.zeros((len(test_loader)))

    for step, data in tqdm(enumerate(test_loader),
                           total=(len(test_loader)),
                           smoothing=0.9):
        depth_target = data[0].detach().numpy()
        depth_source = data[1].detach().numpy()

        pc_target = depth_to_pc(depth_target[0])
        pc_source = depth_to_pc(depth_source[0])

        ''' move estimation '''
        o3d_target = np_depth_to_o3d_pc(depth_target[0])
        o3d_source = np_depth_to_o3d_pc(depth_source[0])

        start_time = time.time()
        current_transformation = np.identity(4)
        result_icp = o3d.pipelines.registration.registration_icp(
            o3d_source,
            o3d_target,
            args.threshold,
            current_transformation)
        end_time = time.time()

        all_time[step] = end_time - start_time

    print('avg icp time: %f (s)' % (list_avg(all_time)))
