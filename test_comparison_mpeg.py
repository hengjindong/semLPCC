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

from dataset import Dataset_MM
from utils.pc_error_wrapper import pc_error
from Chamfer3D.loss_utils import chamfer_rmse
from utils.tools import depth_to_xyz, pc_to_depth, depth_to_pc, mpeg_pcc_dmetric
from utils.pc_save_utils import save_point_cloud_to_file, depth_save_as_pc, depth_xyz_save_as_pc

def parse_args():
    parser = argparse.ArgumentParser('instance_based_pcc')
    parser.add_argument('--dataset_path', type=str,
                        default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    parser.add_argument('--seq_name', type=str, default='00')
    parser.add_argument('--depth_width', type=int, default=2048)
    parser.add_argument('--depth_hight', type=int, default=64)
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

def eval_depth_rate(depth):
    save_png_path = './exp_me_icp/temp/temp_depth_' + \
        str(np.random.randint(10)) + '.png'
    numpngw.write_png(save_png_path, depth)
    fsize_B = os.path.getsize(save_png_path)
    pc_num = len(depth) * len(depth[0])

    fsize_bit = fsize_B * 8
    bpp = fsize_bit / pc_num
    return bpp, fsize_bit

def eval_bitstream_rate(path_bitstream, pc_num):
    fsize_B = os.path.getsize(path_bitstream)
    fsize_bit = fsize_B * 8
    bpp = fsize_bit / pc_num
    return bpp, fsize_bit

def compression_mpeg(quant, pc_ori, cur_name):
    path_tmc3 = '/home/install/exp_pcc/mpeg-pcc-tmc13/build/tmc3/tmc3'
    cfg_encoder = './test_comparison/mpeg/config/encoder_1.cfg'
    cfg_decoder = './test_comparison/mpeg/config/decoder.cfg'
    path_in_pc = './test_comparison/temp/mpeg_in_' + cur_name + '.ply'
    path_bin = './test_comparison/temp/mpeg_' + cur_name + '.bin'
    path_out_pc = './test_comparison/temp/mpeg_out_' + cur_name + '.ply'

    ''' encode  '''
    pc_quant = (pc_ori * quant).astype(np.int32)
    save_point_cloud_to_file(path_in_pc, pc_quant)

    command_encoder = path_tmc3 + ' -c ' + cfg_encoder + \
        ' --uncompressedDataPath=' + path_in_pc + \
        ' --compressedStreamPath=' + path_bin
    encoder = os.popen(command_encoder)
    time.sleep(2)

    ''' decode  '''
    command_decoder = path_tmc3 + ' -c ' + cfg_decoder + \
        ' --compressedStreamPath=' + path_bin + \
        ' --reconstructedDataPath=' + path_out_pc
    decoder = os.popen(command_decoder)
    time.sleep(2)

    o3d_pc_rec = o3d.io.read_point_cloud(path_out_pc)
    pc_rec = np.asarray(o3d_pc_rec.points)
    pc_rec = pc_rec.astype(np.float64) / quant
    return pc_rec, path_bin

def eval_pc_rmse(pc_rec, pc_ori):
    torch_pc_rec = torch.from_numpy(pc_rec).float().cuda()
    torch_pc_ori = torch.from_numpy(pc_ori).float().cuda()
    _, rmse = chamfer_rmse(torch_pc_rec.unsqueeze(0),
                           torch_pc_ori.unsqueeze(0))
    return rmse

def test_seq_mpeg(args, mpeg_quant):
    TEST_DATASET = Dataset_MM(root = args.dataset_path,
                              seq_name = args.seq_name,
                              depth_width = args.depth_width,
                              depth_hight = args.depth_hight)
    test_loader = DataLoader(TEST_DATASET,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)

    all_bpp = []
    all_fsize_bit = []
    all_rmse = []

    for step, data in tqdm(enumerate(test_loader),
                           total=(len(test_loader)),
                           smoothing=0.9):
        depth = data[0].detach().numpy()
        pc_ori = depth_to_pc(depth[0])
        cur_name = args.seq_name + '_' + str(step)

        ''' compression '''
        pc_rec, path_bitstream = compression_mpeg(mpeg_quant,
                                                  pc_ori,
                                                  cur_name)

        ''' rate cal '''
        pc_num = len(pc_ori)
        bpp, fsize_bit = eval_bitstream_rate(path_bitstream, pc_num)
        all_bpp.append(bpp)
        all_fsize_bit.append(fsize_bit)

        ''' quality cal '''
        rmse = eval_pc_rmse(pc_rec, pc_ori)
        all_rmse.append(rmse)

    print('***************************************************************')
    print('* mpeg quant: %d' % mpeg_quant)
    print('***************************************************************')
    print('* avg bpp:    %f' % list_avg(all_bpp))
    print('* avg rmse:   %f (cm)' % (list_avg(all_rmse) * 100))
    print('* avg size:   %f (KB)' % (list_avg(all_fsize_bit) / (8*1024)))

if __name__ == '__main__':
    args = parse_args()
    print(args)

    list_mpeg_quant = [1000, 500, 250, 100, 50, 20, 15, 10, 5]
    for i_quant in list_mpeg_quant:
        test_seq_mpeg(args, i_quant)
