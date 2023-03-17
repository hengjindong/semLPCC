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

from dataset import Dataset_label
from utils.pc_error_wrapper import pc_error
from Chamfer3D.loss_utils import chamfer_rmse
from utils.tools import depth_to_xyz, pc_to_depth, depth_to_pc, mpeg_pcc_dmetric
from utils.pc_save_utils import save_point_cloud_to_file, depth_save_as_pc, depth_xyz_save_as_pc

def parse_args():
    parser = argparse.ArgumentParser('instance_based_pcc')
    parser.add_argument('--dataset_path', type=str,
                        default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    parser.add_argument('--seq_name', type=str, default='00')
    parser.add_argument('--save_seq_name', type=str, default='50')
    parser.add_argument('--depth_width', type=int, default=2048)
    parser.add_argument('--depth_hight', type=int, default=64)
    parser.add_argument('--loss_bit_range', type=int, default=5)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    TEST_DATASET = Dataset_label(root = args.dataset_path,
                              seq_name = args.seq_name,
                              depth_width = args.depth_width,
                              depth_hight = args.depth_hight)
    test_loader = DataLoader(TEST_DATASET,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)

    seq_save_path = args.dataset_path + '/' + args.save_seq_name
    temp_save_path = Path(seq_save_path)
    temp_save_path.mkdir(exist_ok=True)
    temp_save_path = temp_save_path.joinpath('labels')
    temp_save_path.mkdir(exist_ok=True)

    for step, data in tqdm(enumerate(test_loader),
                           total=(len(test_loader)),
                           smoothing=0.9):
        depth = data[0].detach().numpy()
        depth_sem = data[2].detach().numpy()

        label_name = str(step)
        while len(label_name) < 6:
            label_name = '0' + label_name
        label_name = label_name + '.label'

        label = depth_sem.flatten()

        label = label.astype(np.uint32)
        label_save_path = seq_save_path + '/labels/' + label_name
        label.tofile(label_save_path)

