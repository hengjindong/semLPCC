import os
import pdb
import json
from glob import glob

import h5py
import yaml
import torch
import open3d
import numpngw
import numpy as np
import open3d as o3d
import torch.utils.data as data
from tqdm import tqdm
from pytorch3d.ops import sample_farthest_points

from utils.o3d_utils import show_pc
from kitti_tool.laserscan import LaserScan, SemLaserScan
from utils.pc_save_utils import save_point_cloud_to_file

class Dataset_MM(data.Dataset):
    def __init__(self, root, seq_name='03', depth_width = 2048, depth_hight = 64):
        self.root = root
        self.seq_name = seq_name
        self.depth_width = depth_width
        self.depth_hight = depth_hight

        self.path_bin_all = self.get_path()
        self.all_data_depth, self.all_data_xyz, self.all_data_sem = \
            self.load_data(self.path_bin_all)

    def get_path(self):
        path_bin_all = []
        path_bin_all = os.path.join(
            self.root,
            self.seq_name + '/velodyne/*.bin'
        )
        path_bin_all = sorted(glob(path_bin_all))

        path_bin_all = path_bin_all[0::10]
        return path_bin_all

    def load_data(self, path):
        all_data_depth = []
        all_data_xyz = []
        all_data_sem = []
        for path_bin in tqdm(path):
            path_label = path_bin.replace('.bin', '.label')
            path_label = path_label.replace('/velodyne/', '/labels/')
            data_depth, data_xyz, data_sem = \
                load_scan(path_bin, path_label, self.depth_hight, self.depth_width)
            all_data_depth.append(data_depth)
            all_data_xyz.append(data_xyz)
            all_data_sem.append(data_sem)
        return all_data_depth, all_data_xyz, all_data_sem

    def np2torch(np_data):
        torch_data = torch.from_numpy(np_data)
        torch_data = torch_data.float()
        return torch_data

    def __getitem__(self, item):
        # data_depth = np2torch(self.all_data_depth[item])
        # data_xyz = np2torch(self.all_data_xyz[item])
        # data_sem = np2torch(self.all_data_sem[item])
        data_depth = self.all_data_depth[item]
        data_xyz = self.all_data_xyz[item]
        data_sem = self.all_data_sem[item]
        return data_depth, data_xyz, data_sem

    def __len__(self):
        return len(self.path_bin_all)

class Dataset_scene(data.Dataset):
    def __init__(self, root, seq_name='03', seq_len=100,
                 depth_width = 2048, depth_hight = 64, scene_interval = 1):
        self.root = root
        self.seq_name = seq_name
        self.seq_len = seq_len
        self.depth_width = depth_width
        self.depth_hight = depth_hight
        self.scene_interval = scene_interval

        self.path_bin_all = self.get_path()
        self.all_object, self.all_scene, self.all_ground = \
            self.load_data(self.path_bin_all)

    def get_path(self):
        path_bin_all = []
        path_bin_all = os.path.join(
            self.root,
            self.seq_name + '/velodyne/*.bin'
        )
        path_bin_all = sorted(glob(path_bin_all))
        path_bin_all = path_bin_all[0:self.seq_len]
        # path_bin_all = path_bin_all[0::10]
        return path_bin_all

    def load_data(self, path):
        all_object = []
        all_scene = []
        all_ground = []

        def distincting_label_object_scene(depth_sem):
            object_labels = [10, 11, 13, 15, 16, 18, 20, 30, 31, 32,
                             252, 253, 254, 255, 256, 257, 258, 259]
            ground_labels = [1, 40, 44, 48 , 49]
            H = len(depth_sem)
            W = len(depth_sem[0])
            object_idx = np.zeros((H, W))
            scene_idx = np.zeros((H, W))
            ground_idx = np.zeros((H, W))
            for row in range(H):
                for col in range(W):
                    if np.isin(depth_sem[row][col], object_labels):
                        object_idx[row][col] = 1
                    elif np.isin(depth_sem[row][col], ground_labels):
                        ground_idx[row][col] = 1
                    else:
                        scene_idx[row][col] = 1
            return object_idx, scene_idx, ground_idx

        for path_bin in tqdm(path):
            path_label = path_bin.replace('.bin', '.label')
            path_label = path_label.replace('/velodyne/', '/labels/')
            data_depth, data_xyz, data_sem = load_scan(path_bin,
                                                       path_label,
                                                       self.depth_hight,
                                                       self.depth_width)

            object_idx, scene_idx, ground_idx = \
                distincting_label_object_scene(data_sem)

            all_object.append(data_depth * object_idx)
            all_scene.append(data_depth * scene_idx)
            all_ground.append(data_depth * ground_idx)
        return all_object, all_scene, all_ground

    def np2torch(np_data):
        torch_data = torch.from_numpy(np_data)
        torch_data = torch_data.float()
        return torch_data

    def __getitem__(self, item):
        object_target = self.all_object[item + self.scene_interval]
        scene_target = self.all_scene[item + self.scene_interval]
        ground_target = self.all_ground[item + self.scene_interval]
        scene_source = self.all_scene[item]
        return object_target, scene_target, ground_target, scene_source

    def __len__(self):
        return len(self.path_bin_all) - self.scene_interval

class Dataset_ME(data.Dataset):
    def __init__(self, root, seq_name='03', seq_len=100,
                 depth_width = 2048, depth_hight = 64, scene_interval = 1):
        self.root = root
        self.seq_name = seq_name
        self.seq_len = seq_len
        self.depth_width = depth_width
        self.depth_hight = depth_hight
        self.scene_interval = scene_interval

        self.path_bin_all = self.get_path()
        self.all_total = self.load_data(self.path_bin_all)

    def get_path(self):
        path_bin_all = []
        path_bin_all = os.path.join(
            self.root,
            self.seq_name + '/velodyne/*.bin'
        )
        path_bin_all = sorted(glob(path_bin_all))
        path_bin_all = path_bin_all[0:self.seq_len]
        return path_bin_all

    def load_data(self, path):
        all_total = []
        for path_bin in tqdm(path):
            path_label = path_bin.replace('.bin', '.label')
            path_label = path_label.replace('/velodyne/', '/labels/')

            data_depth, data_xyz, data_sem = \
                load_scan(path_bin,path_label,self.depth_hight,self.depth_width)

            all_total.append(data_depth)
        return all_total

    def np2torch(np_data):
        torch_data = torch.from_numpy(np_data)
        torch_data = torch_data.float()
        return torch_data

    def __getitem__(self, item):
        target = self.all_total[item + self.scene_interval]
        source = self.all_total[item]
        return target, source

    def __len__(self):
        return len(self.path_bin_all) - self.scene_interval

class Dataset_label(data.Dataset):
    def __init__(self, root, seq_name='03', depth_width = 2048, depth_hight = 64):
        self.root = root
        self.seq_name = seq_name
        self.depth_width = depth_width
        self.depth_hight = depth_hight

        self.path_bin_all = self.get_path()
        self.all_data_depth, self.all_data_xyz, self.all_data_sem = \
            self.load_data(self.path_bin_all)

    def get_path(self):
        path_bin_all = []
        path_bin_all = os.path.join(
            self.root,
            self.seq_name + '/velodyne/*.bin'
        )
        path_bin_all = sorted(glob(path_bin_all))

        path_bin_all = path_bin_all[0::10]
        return path_bin_all

    def load_data(self, path):
        all_data_depth = []
        all_data_xyz = []
        all_data_sem = []
        for path_bin in tqdm(path):
            path_label = path_bin.replace('.bin', '.label')
            path_label = path_label.replace('/velodyne/', '/labels/')
            data_depth, data_xyz, data_sem = \
                load_scan_save_label(path_bin, path_label, self.depth_hight, self.depth_width)
            all_data_depth.append(data_depth)
            all_data_xyz.append(data_xyz)
            all_data_sem.append(data_sem)
        return all_data_depth, all_data_xyz, all_data_sem

    def np2torch(np_data):
        torch_data = torch.from_numpy(np_data)
        torch_data = torch_data.float()
        return torch_data

    def __getitem__(self, item):
        data_depth = self.all_data_depth[item]
        data_xyz = self.all_data_xyz[item]
        data_sem = self.all_data_sem[item]
        return data_depth, data_xyz, data_sem

    def __len__(self):
        return len(self.path_bin_all)

class Dataset_pc(data.Dataset):
    def __init__(self, root, seq_name='03', depth_width = 2048, depth_hight = 64):
        self.root = root
        self.seq_name = seq_name
        self.depth_width = depth_width
        self.depth_hight = depth_hight

        self.path_bin_all = self.get_path()
        self.all_data_depth, self.all_data_xyz, self.all_data_sem = \
            self.load_data(self.path_bin_all)

    def get_path(self):
        path_bin_all = []
        path_bin_all = os.path.join(
            self.root,
            self.seq_name + '/velodyne/*.bin'
        )
        path_bin_all = sorted(glob(path_bin_all))

        path_bin_all = path_bin_all[0::10]
        return path_bin_all

    def load_data(self, path):
        all_data_depth = []
        all_data_xyz = []
        all_data_sem = []
        for path_bin in tqdm(path):
            path_label = path_bin.replace('.bin', '.label')
            path_label = path_label.replace('/velodyne/', '/labels/')
            data_depth, data_xyz, data_sem = \
                load_scan_pc(path_bin, path_label, self.depth_hight, self.depth_width)
            all_data_depth.append(data_depth)
            all_data_xyz.append(data_xyz)
            all_data_sem.append(data_sem)
        return all_data_depth, all_data_xyz, all_data_sem

    def np2torch(np_data):
        torch_data = torch.from_numpy(np_data)
        torch_data = torch_data.float()
        return torch_data

    def __getitem__(self, item):
        # data_depth = np2torch(self.all_data_depth[item])
        # data_xyz = np2torch(self.all_data_xyz[item])
        # data_sem = np2torch(self.all_data_sem[item])
        data_depth = self.all_data_depth[item]
        data_xyz = self.all_data_xyz[item]
        data_sem = self.all_data_sem[item]
        return data_depth, data_xyz, data_sem

    def __len__(self):
        return len(self.path_bin_all)

def depth_xyz_to_pc(data_xyz):
    valid_idx = np.where(np.sum(data_xyz, -1) != 0)
    point_cloud = data_xyz[valid_idx]
    return point_cloud

def load_scan(path_bin, path_label, img_H, img_W):
    CFG = yaml.safe_load(open("./kitti_tool/config/semantic-kitti.yaml", 'r'))
    color_dict = CFG["color_map"]
    nclasses = len(color_dict)

    scan = SemLaserScan(nclasses,
                        color_dict,
                        project=True,
                        H=img_H,
                        W=img_W,
                        fov_up=3,
                        fov_down=-25)
    scan.open_scan(path_bin)
    scan.open_label(path_label)
    scan.colorize()

    proj_sem_color = (scan.proj_sem_label).astype(np.uint8)
    proj_range = scan.proj_range
    proj_xyz = scan.proj_xyz
    return proj_range, proj_xyz, proj_sem_color

def load_scan_pc(path_bin, path_label, img_H, img_W):
    CFG = yaml.safe_load(open("./kitti_tool/config/semantic-kitti.yaml", 'r'))
    color_dict = CFG["color_map"]
    nclasses = len(color_dict)

    scan = SemLaserScan(nclasses,
                        color_dict,
                        project=True,
                        H=img_H,
                        W=img_W,
                        fov_up=3,
                        fov_down=-25)
    scan.open_scan(path_bin)
    scan.open_label(path_label)
    scan.colorize()

    proj_sem_color = (scan.proj_sem_label).astype(np.uint8)
    proj_range = scan.proj_range
    proj_rem = scan.proj_remission
    return proj_range, proj_rem, proj_sem_color

def load_scan_save_label(path_bin, path_label, img_H, img_W):
    CFG = yaml.safe_load(open("./kitti_tool/config/semantic-kitti.yaml", 'r'))
    color_dict = CFG["learning_map"]
    nclasses = len(color_dict)

    scan = SemLaserScan(nclasses,
                        color_dict,
                        project=True,
                        H=img_H,
                        W=img_W,
                        fov_up=3,
                        fov_down=-25)
    scan.open_scan(path_bin)
    scan.open_label(path_label)
    scan.colorize()

    proj_sem = scan.proj_sem_label
    proj_sem = np.vectorize(color_dict.__getitem__)(proj_sem)
    proj_sem = (proj_sem).astype(np.uint8)
    proj_range = scan.proj_range
    proj_xyz = scan.proj_xyz
    return proj_range, proj_xyz, proj_sem

def bin_to_range(bin_path, img_H, img_W):
    CFG = yaml.safe_load(open("./kitti_tool/config/semantic-kitti.yaml", 'r'))
    scan = LaserScan(project=True,
                        H=img_H,
                        W=img_W,
                        fov_up=3,
                        fov_down=-25)
    scan.open_scan(bin_path)
    proj_range = scan.proj_range
    proj_range = remove_balck_line_and_remote_points(proj_range)
    return proj_range, proj_range

def remove_balck_line_and_remote_points(range_image):
    range_x = range_image.shape[0]
    range_y = range_image.shape[1]
    range_image_completion = range_image
    for height in range(1, range_x-1):
        for width in range(1, range_y-1):
            left = max(width - 1, 0)
            right = min(width + 1, range_y)
            up = max(height - 1, 0)
            down = min(height + 1, range_x)
            if range_image[height, width] == 0:
                # remove straight line
                if range_image[up][width] > 0 and range_image[down][width] > 0 and (range_image[height][left] == 0 or range_image[height][right] == 0):
                    range_image_completion[height][width] = (range_image[up][width] + range_image[down][width]) / 2

    for height in range(1, range_x-1):
        for width in range(1, range_y-1):
            left = max(width - 1, 0)
            right = min(width + 1, range_y)
            up = max(height - 1, 0)
            down = min(height + 1, range_x)
            if range_image_completion[height][width] == 0:
                point_up = range_image_completion[up][width]
                point_down = range_image_completion[down][width]
                point_left = range_image_completion[height][left]
                point_right = range_image_completion[height][right]
                point_left_up = range_image_completion[up][left]
                point_right_up = range_image_completion[up][right]
                point_left_down = range_image_completion[down][left]
                point_right_down = range_image_completion[down][right]
                surround_points = int(point_up != 0) + int(point_down != 0) + int(point_left != 0) + int(
                    point_right != 0) + int(point_left_up != 0) + int(point_right_up != 0) + int(
                    point_left_down != 0) + int(point_right_down != 0)
                if surround_points >= 7:
                    surround_points_sum = point_up + point_down + point_left + point_right + point_left_up + point_right_up + point_left_down + point_right_down
                    range_image_completion[height][width] = surround_points_sum / surround_points

    return range_image_completion

'''
TRAIN_DATASET = Dataset_MM(root = '/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences',
                         dataset_name = '07',
                         depth_width = 2048,
                         depth_hight = 64)
trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=16, shuffle=False, num_workers=4)
'''
