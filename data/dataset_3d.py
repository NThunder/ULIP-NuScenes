'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

import random

import torch
import numpy as np
import torch.utils.data as data

import yaml
from easydict import EasyDict

from utils.io import IO
from utils.build import DATASETS
from utils.logger import *
from utils.build import build_dataset_from_cfg
import json
from tqdm import tqdm
import pickle
from PIL import Image

import cv2

from nuscenes.utils.geometry_utils import view_points, points_in_box


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

@DATASETS.register_module()
class ModelNet(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        split = config.subset
        self.subset = config.subset
        self.use_10k_pc = config.use_10k_pc
        self.use_colored_pc = config.use_colored_pc

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='ModelNet')

        if self.uniform:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                # make sure you have raw data in the path before you enable generate_from_raw_data=True.
                if self.generate_from_raw_data:
                    print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
                    self.list_of_points = [None] * len(self.datapath)
                    self.list_of_labels = [None] * len(self.datapath)

                    for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                        fn = self.datapath[index]
                        cls = self.classes[self.datapath[index][0]]
                        cls = np.array([cls]).astype(np.int32)
                        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                            print_log("uniformly sampled out {} points".format(self.npoints))
                        else:
                            point_set = point_set[0:self.npoints, :]

                        self.list_of_points[index] = point_set
                        self.list_of_labels[index] = cls

                    with open(self.save_path, 'wb') as f:
                        pickle.dump([self.list_of_points, self.list_of_labels], f)
                else:
                    # no pre-processed dataset found and no raw data found, then load 8192 points dataset then do fps after.
                    self.save_path = os.path.join(self.root,
                                                  'modelnet%d_%s_%dpts_fps.dat' % (
                                                  self.num_category, split, 8192))
                    print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                    if not self.use_10k_pc:
                        print_log('since no exact points pre-processed dataset found and no raw data found, load 8192 pointd dataset first, if downsampling with fps to {} happens later, the speed is excepted to be slower due to fps...'.format(self.npoints), logger='ModelNet')
                    with open(self.save_path, 'rb') as f:
                        self.list_of_points, self.list_of_labels = pickle.load(f)

            else:
                print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        self.shape_names_addr = os.path.join(self.root, 'modelnet40_shape_names.txt')
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.shape_names = lines

        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        
        if self.use_10k_pc and self.use_colored_pc:
            self.modelnet_10k_colored_pc_file = 'data/modelnet40_normal_resampled/modelnet40_colored_10k_pc.npy'
            self.modelnet_10k_rgb_data = np.load(self.modelnet_10k_colored_pc_file, allow_pickle=True)
            with open('data/modelnet40_normal_resampled/modelnet40_test_split_10k_colored.json', 'r') as f:
                self.cat_name = json.load(f)

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        if  self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            self.gravity_dim = 1
            height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)

        if self.use_10k_pc and self.use_colored_pc:
            point_set = self.modelnet_10k_rgb_data[index]['xyz']
            rgb_data = np.ones_like(point_set) * 0.4
            point_set = np.concatenate([point_set, rgb_data], axis=1)
            cat_name = self.cat_name[index]['category']
            label = [self.shape_names.index(cat_name)]
        elif self.use_colored_pc:
            rgb_data = np.ones_like(point_set) * 0.4
            point_set = np.concatenate([point_set, rgb_data], axis=1)

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label_name = self.shape_names[int(label)]

        return current_points, label, label_name

import pandas as pd

@DATASETS.register_module()
class ShapeNetv2(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.img_path = config.IMG_PATH
        self.train_transform = config.train_transform
        self.ratio = config.ratio
        
        self.text_list = {}
        self.index_list = {}
        for index, row in pd.read_json(config.TEXT_PATH).iterrows():
            self.text_list["0" + str(row['catalogue'])] = row['describe']
            self.index_list["0" + str(row['catalogue'])] = index
        self.subset = config.subset
        self.npoints = config.N_POINTS

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines[: int(self.ratio * len(lines))]:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] load ratio is {self.ratio}', logger='ShapeNet-55')
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pc = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        # img = cv2.imread(os.path.join(self.img_path, sample['file_path'].replace(".npy", ".png")))
        
        # print(os.path.join(self.img_path, sample['file_path'].replace(".npy", ".png")))
        img = pil_loader(os.path.join(self.img_path, sample['file_path'].replace(".npy", ".png")))
        img = self.train_transform(img)
        pc = self.random_sample(pc, self.sample_points_num)
        pc = self.pc_norm(pc)
        pc = torch.from_numpy(pc).float()
        text = self.text_list[sample['taxonomy_id']]
        index = self.index_list[sample['taxonomy_id']]
        label = torch.tensor(index)
        if idx < 20:
            print("index:    ", index)
            print("text:    ", text)
        return sample['taxonomy_id'], sample['model_id'], pc, img, text, label

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.img_path = config.IMG_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer if hasattr(config, 'tokenizer') else None
        self.train_transform = config.train_transform
        self.ratio = config.ratio
        
        # Load text descriptions from your original approach
        self.text_list = {}
        self.index_list = {}
        for index, row in pd.read_json(config.TEXT_PATH).iterrows():
            self.text_list["0" + str(row['catalogue'])] = row['describe']
            self.index_list["0" + str(row['catalogue'])] = index

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = self.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        
        self.file_list = []
        for line in lines[: int(self.ratio * len(lines))]:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line[len(taxonomy_id) + 1:].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] load ratio is {self.ratio}', logger='ShapeNet-55')
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

        self.permutation = np.arange(self.npoints)
        self.uniform = config.get('uniform', True)
        self.augment = config.get('augment', True)
        self.use_height = config.get('use_height', False)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        # Load and process point cloud
        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        
        if self.uniform and self.sample_points_num < data.shape[0]:
            data = farthest_point_sample(data, self.sample_points_num)
        else:
            data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)

        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        # Load and process image
        img = pil_loader(os.path.join(self.img_path, sample['file_path'].replace(".npy", ".png")))
        img = self.train_transform(img)

        # Process text using your original approach
        text = self.text_list[sample['taxonomy_id']]
        if self.tokenizer is not None:
            tokenized_text = self.tokenizer(text)
            tokenized_text = torch.stack([tokenized_text])  # Keep consistent format
        else:
            tokenized_text = text  # Fallback to raw text if no tokenizer

        # Get label from your original approach
        index = self.index_list[sample['taxonomy_id']]
        label = torch.tensor(index)
        
        if idx < 20:
            print("index:    ", index)
            print("text:    ", text)

        return sample['taxonomy_id'], sample['model_id'], tokenized_text, data, img, label

    def __len__(self):
        return len(self.file_list)


import os
import numpy as np
import torch
from PIL import Image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, Box


@DATASETS.register_module()
class NuScenesD(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.version = "v1.0-mini"
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer if hasattr(config, 'tokenizer') else None
        self.train_transform = config.train_transform
        self.ratio = config.ratio
        
        # Initialize nuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=True)
        
        # Create mapping from category to text description and index
        self.category_to_text = {}
        self.category_to_index = {}
        # print("AAA")
        for idx, category in enumerate(self.nusc.category):
            print(category['name'], category['description'])
            self.category_to_text[category['name']] = category['description']
            self.category_to_index[category['name']] = idx

        # Get all samples for the current subset (train/val)
        self.samples = []
        for scene in self.nusc.scene:
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                # print("BBB")
                # print(self.subset, sample['token'])
                # if self.subset in sample['token']:  # Simple subset filtering
                self.samples.append(sample)
                sample_token = sample['next'] if 'next' in sample else None
                
        # Apply ratio if needed
        if self.ratio < 1.0:
            self.samples = self.samples[:int(self.ratio * len(self.samples))]
        
        self.permutation = np.arange(self.npoints)
        self.uniform = config.get('uniform', True)
        self.augment = config.get('augment', True)
        self.use_height = config.get('use_height', False)

    def pc_norm(self, pc):
        """ Normalize point cloud """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get LIDAR point cloud
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_data['filename']))
        points = pc.points[:3, :].T  # Get xyz points
        
        # Process point cloud
        if self.uniform and self.npoints < points.shape[0]:
            points = farthest_point_sample(points, self.npoints)
        else:
            points = self.random_sample(points, self.npoints)
        points = self.pc_norm(points)

        if self.augment:
            points = random_point_dropout(points[None, ...])
            points = random_scale_point_cloud(points)
            points = shift_point_cloud(points)
            points = rotate_perturbation_point_cloud(points)
            points = rotate_point_cloud(points)
            points = points.squeeze()

        if self.use_height:
            height_array = points[:, 1:2] - points[:, 1:2].min()
            points = np.concatenate((points, height_array), axis=1)
            points = torch.from_numpy(points).float()
        else:
            points = torch.from_numpy(points).float()

        # Get camera image (using front camera as example)
        cam_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        img = Image.open(os.path.join(self.nusc.dataroot, cam_data['filename']))
        img = self.train_transform(img)

        # Get category text (using first annotation as example)
        ann_token = sample['anns'][0]
        ann = self.nusc.get('sample_annotation', ann_token)
        category_name = ann['category_name']
        text = self.category_to_text.get(category_name, category_name)
        
        if self.tokenizer is not None:
            tokenized_text = self.tokenizer(text)
            tokenized_text = torch.stack([tokenized_text])
        else:
            tokenized_text = text

        # Get label
        label = torch.tensor(self.category_to_index[category_name])

        return sample['token'], ann['instance_token'], tokenized_text, points, img, label

    def __len__(self):
        return len(self.samples)

@DATASETS.register_module()
class NuScenesTest(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.version = "v1.0-mini"
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer if hasattr(config, 'tokenizer') else None
        self.ratio = config.ratio
        
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=True)
        
        self.category_to_text = {}
        self.category_to_index = {}
        for idx, category in enumerate(self.nusc.category):
            self.category_to_text[category['name']] = category['description']
            self.category_to_index[category['name']] = idx


        self.annotations_points = []
        self.labels_name = []
        self.labels = []

        for sample in self.nusc.sample:

            for ann_token in sample['anns']:
                
                ann = self.nusc.get('sample_annotation', ann_token)
                
                category_name = ann['category_name']
                caption = category_name.split('.')[1]
      
                lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_data['filename']))
                
                cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
                pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
                pc.translate(np.array(cs_record['translation']))

                poserecord = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
                pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
                pc.translate(np.array(poserecord['translation']))
                
                # Create box and get points inside
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']),
                        name=ann['category_name'], token=ann['token'])

                mask = points_in_box(box, pc.points[:3, :])
                points = pc.points[:3, mask].T
                if points.shape[0] > 100:
                    self.annotations_points.append(points)
                    self.labels_name.append(caption)
                    self.labels.append(torch.tensor(self.category_to_index[category_name]))
        

        if self.ratio < 1.0:
            self.annotations_points = self.annotations_points[-int(self.ratio * len(self.annotations_points)):]
            self.labels_name = self.labels_name[-int(self.ratio * len(self.labels_name)):]
            self.labels = self.labels[-int(self.ratio * len(self.labels)):]
        
        self.permutation = np.arange(self.npoints)
        self.uniform = config.get('uniform', True)
        self.augment = config.get('augment', True)


    def pc_norm(self, pc):
        """ Normalize point cloud """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
    
    def _pad_points(self, points, target_num):
        from scipy.interpolate import interpn
        num_points = points.shape[0]
        
        if num_points == 0:
            return np.random.rand(target_num, 3)
        
        # Создаем интерполятор
        x = np.linspace(0, 1, num_points)
        x_new = np.linspace(0, 1, target_num)
        
        # Интерполируем по каждой оси
        points_interp = np.zeros((target_num, 3))
        for i in range(3):
            points_interp[:, i] = np.interp(x_new, x, points[:, i])
    
        return points_interp

    def __getitem__(self, idx):
        points = self.annotations_points[idx]
        label = self.labels[idx]
        label_name = self.labels_name[idx]
        
        if points.shape[0] < self.npoints:
            points = self._pad_points(points, self.npoints)
        if self.uniform and self.npoints < points.shape[0]:
            points = farthest_point_sample(points, self.npoints)
        else:
            points = self.random_sample(points, self.npoints)
        points = self.pc_norm(points)

        points = torch.from_numpy(points).float()


        return points, label, label_name

    def __len__(self):
        return len(self.annotations_points)
    
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from nuscenes import NuScenesExplorer

@DATASETS.register_module()
class NuScenesCropDataset(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.version = "v1.0-mini"
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer if hasattr(config, 'tokenizer') else None
        self.train_transform = config.train_transform
        self.ratio = config.ratio
        self.cam_name = 'CAM_FRONT'  # e.g. 'CAM_FRONT'
        
        # Initialize nuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=True)
        
        # Create mapping from category to text and index
        self.category_to_text = {}
        self.category_to_index = {}
        for idx, category in enumerate(self.nusc.category):
            self.category_to_text[category['name']] = category['description']
            self.category_to_index[category['name']] = idx

        # Collect all annotations for the subset
        self.annotations = []
        self.class_dist = {}
        common = 0
        for sample in self.nusc.sample:
            self.camera_channels = [chan for chan in sample['data'].keys() if chan.startswith('CAM_')]
            # if self.subset in sample['token']:  # Simple subset filtering
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                category_name = ann['category_name']
                caption = category_name.split('.')[1]
                if caption not in self.class_dist:
                    self.class_dist[caption] = 1
                else:
                    self.class_dist[caption] += 1
                common += 1
                # if not ann['category_name'].startswith('vehicle.car'):
                #     continue
                
                # sample = self.nusc.get('sample', ann['sample_token'])
                f = False
                for cam_channel in self.camera_channels:
                    cam_token = sample['data'][cam_channel]
                    _, boxes, _ = self.nusc.get_sample_data(cam_token, selected_anntokens=[ann_token])
                    if boxes:
                        f = True
                        break
                if not f:
                    break
                        
                
                # 1. Get and crop point cloud for this detection
                lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_data['filename']))
                
                # Transform points to ego vehicle frame
                cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
                pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
                pc.translate(np.array(cs_record['translation']))
                
                # Transform points to global frame
                poserecord = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
                pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
                pc.translate(np.array(poserecord['translation']))
                
                # Create box and get points inside
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']),
                        name=ann['category_name'], token=ann['token'])
                # print("BOX: ", box)

                mask = points_in_box(box, pc.points[:3, :])
                points = pc.points[:3, mask].T
                if points.shape[0] > 100:
                    self.annotations.append(ann_token)
        
        print("class_dist:  ", self.class_dist)
        for a, b in self.class_dist.items():
            print(a, b/common)
        # Apply ratio if needed
        if self.ratio < 1.0:
            self.annotations = self.annotations[:int(self.ratio * len(self.annotations))]
        
        self.permutation = np.arange(self.npoints)
        self.uniform = config.get('uniform', True)
        self.augment = config.get('augment', True)
        self.use_height = config.get('use_height', False)

    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if np.isclose(m, 0) or np.isnan(m):
            # Если все точки совпадают с центром или m=NaN
            return np.zeros_like(pc)
        
        # 4. Безопасная нормализация
        pc = pc / (m + 1e-10)  # Добавляем малую константу для стабильности
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        # print("pc:  ", pc.shape)
        pc = pc[self.permutation[:num]]
        return pc

    def _pad_points(self, points, target_num):
        from scipy.interpolate import interpn
        num_points = points.shape[0]
        
        if num_points == 0:
            return np.random.rand(target_num, 3)
        
        # Создаем интерполятор
        x = np.linspace(0, 1, num_points)
        x_new = np.linspace(0, 1, target_num)
        
        # Интерполируем по каждой оси
        points_interp = np.zeros((target_num, 3))
        for i in range(3):
            points_interp[:, i] = np.interp(x_new, x, points[:, i])
        
        return points_interp
    
    def __getitem__(self, idx):
        ann_token = self.annotations[idx]
        ann = self.nusc.get('sample_annotation', ann_token)
        sample = self.nusc.get('sample', ann['sample_token'])
        cam_data = self.nusc.get('sample_data', sample['data'][self.cam_name])
        img = Image.open(os.path.join(self.nusc.dataroot, cam_data['filename']))
        full_img = img
        f_cam_chanel = self.cam_name
        left = 0
        top = 0
        right = 0
        bottom = 0
        f_box = None
        for cam_channel in self.camera_channels:
            cam_token = sample['data'][cam_channel]
            f_cam_chanel = cam_channel
            cam_data = self.nusc.get('sample_data', cam_token)
            img = Image.open(os.path.join(self.nusc.dataroot, cam_data['filename']))
            full_img = img
            _, boxes, _ = self.nusc.get_sample_data(cam_token, selected_anntokens=[ann_token])
            if boxes:
                box = boxes[0]
                f_box = box
                try:
                    cam_data = self.nusc.get('sample_data', cam_token)
                    cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
                    
                    corners_3d = box.corners()
                    corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2]
                    
                    left = max(0, int(np.floor(corners_2d[0].min())))
                    right = min(img.width, int(np.ceil(corners_2d[0].max())))
                    top = max(0, int(np.floor(corners_2d[1].min())))
                    bottom = min(img.height, int(np.ceil(corners_2d[1].max())))
                    if right > left and bottom > top:
                        img =  img.crop((left, top, right, bottom))
                        break
                except Exception as e:
                    print(f"Ошибка при обработке {cam_channel}: {e}")
                    continue
        if f_box is  None:
            print("f_box:   ", f_box, ann['category_name'], idx)
            img.save("./no_box.jpg")
                
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_data['filename']))
        cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))
        poserecord = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))
        pc_lidar = pc.points.copy()
        sample_record = self.nusc.get('sample', sample['token'])
        camera_token = sample_record['data'][f_cam_chanel]
        cam = self.nusc.get('sample_data', camera_token)
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        mask = points_in_box(f_box, pc.points[:3, :])
        points_in_box1 = pc.points[:3, mask].T
        points = points_in_box1.copy()
        
        if points_in_box1.shape[0] == 0:
            size = np.array(ann['size'])
            translation = np.array(ann['translation'])
            points = np.random.rand(self.npoints, 3) * size + translation - size/2
        elif points_in_box1.shape[0] < self.npoints:
            points = self._pad_points(points_in_box1, self.npoints)
        if self.uniform and self.npoints < points.shape[0]:
            points = farthest_point_sample(points, self.npoints)
        else:
            points = self.random_sample(points, self.npoints)
        points = self.pc_norm(points)

        if self.augment:
            points = random_point_dropout(points[None, ...])
            points = random_scale_point_cloud(points)
            points = shift_point_cloud(points)
            points = rotate_perturbation_point_cloud(points)
            points = rotate_point_cloud(points)
            points = points.squeeze()
        points = torch.from_numpy(points).float()


        
        # if idx < 20:
        #     fig = plt.figure(figsize=(12, 6))
            
        #     ax1 = fig.add_subplot(131)
        #     ax1.imshow(full_img)

        #     rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=1, edgecolor='r', facecolor='none')
        #     ax1.add_patch(rect)
            
        #     # output_path1 = f"visualizationD_{idx}.png"
        #     # self.nusc.render_pointcloud_in_image(sample['token'], pointsensor_channel='LIDAR_TOP', camera_channel=f_cam_chanel
        #     #                                      , out_path=output_path1, filter_lidarseg_labels=[24])

        #     points3 = view_points(points_in_box1.T, np.array(cs_record['camera_intrinsic']), normalize=True)
            
        #     rect_left, rect_top = rect.get_xy()
        #     rect_width = rect.get_width()
        #     rect_height = rect.get_height()
        #     rect_right = rect_left + rect_width
        #     rect_bottom = rect_top + rect_height
        #     mask = (
        #         (points3[0] >= rect_left) & 
        #         (points3[0] <= rect_right) & 
        #         (points3[1] >= rect_top) & 
        #         (points3[1] <= rect_bottom)
        #     )

        #     filtered_points_2d = points3[:, mask]
            
        #     img_array = np.array(full_img)
            
            
            
        #     colors = []
        #     for i in range(filtered_points_2d.shape[1]):
        #         x, y = int(round(filtered_points_2d[0, i])), int(round(filtered_points_2d[1, i]))
        #         if 0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]:
        #             colors.append(img_array[y, x])  # OpenCV использует порядок (y,x)
        #         else:
        #             colors.append([0, 0, 0])  # Черный для точек вне изображения

        #     colors = np.array(colors) / 255.0  # Нормализуем в [0,1] для matplotlib
            
            

        #     ax1.scatter(filtered_points_2d[0], filtered_points_2d[1], c=points_in_box1[mask, 2], s=5)
        #     ax2 = fig.add_subplot(132, projection='3d')
    
        #     ax2.scatter(points_in_box1[mask, 0], points_in_box1[mask, 1], points_in_box1[mask, 2], c=colors, s=5)        
        #     for corner in f_box.corners().T:
        #         ax2.scatter(corner[0], corner[1], corner[2], color='red', s=25)
            
        #     ax2.set_xlabel('X')
        #     ax2.set_ylabel('Y')
        #     ax2.set_zlabel('Z')
            
        #     ax3 = fig.add_subplot(133, projection='3d')
        #     ax3.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)        
        #     category_name = ann['category_name']
        #     text = self.category_to_text.get(category_name, category_name)
        #     ax3.set_title(text)   
                         
        #     output_path = f"visualization_{idx}.png"
        #     plt.tight_layout()
        #     plt.savefig(output_path)
        #     plt.close()
            
        #     # print(f"Визуализация {idx} сохранена в {output_path}")
        #     label = torch.tensor(self.category_to_index[category_name])
        #     # print("All: ", idx, text, label)
                
                
        img = self.train_transform(img)

        # Get text and label
        category_name = ann['category_name']
        caption = category_name.split('.')[1]
        tokenized_captions = []
        tokenized_captions.append(self.tokenizer(caption))
        tokenized_captions = torch.stack(tokenized_captions)
        
        text = self.category_to_text.get(category_name, category_name)
        if self.tokenizer is not None:
            tokenized_text = self.tokenizer(text)
            tokenized_text = torch.stack([tokenized_text])
        else:
            tokenized_text = text
        label = torch.tensor(self.category_to_index[category_name])
        
        # if idx < 20:
        #     print("category_name:    ", category_name)
        #     print("text:    ", text)
        #     print("caption:    ", caption)

        return sample['token'], ann['instance_token'], tokenized_captions, points, img, label

    def __len__(self):
        return len(self.annotations)
  
@DATASETS.register_module()
class Objaverse_Lvis_Colored(data.Dataset):
    def __init__(self, config):

        self.npoints = 10000
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform

        self.lvis_list_addr = 'data/objaverse-lvis/lvis.json'
        self.lvis_metadata_addr = 'data/objaverse-lvis/objaverse_lvis_metadata.json'

        with open(self.lvis_list_addr, 'r') as f:
            self.npy_file_map = json.load(f)

        self.file_list = list(self.npy_file_map.keys())

        with open(self.lvis_metadata_addr, 'r') as f:
            self.lvis_metadata = json.load(f)

        self.prompt_template_addr = 'data/templates.json'
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.sample_points_num = self.npoints

        print_log(f'Objaverse lvis {len(self.file_list)} instances were loaded', logger='objaverse_lvis')

        self.permutation = np.arange(self.npoints)

        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = False
        self.use_color = True
        
        self.objaverse_lvis_path = 'data/objaverse-lvis'
        
        if self.use_color:
            print("use color")
        else:
            print("don't use color")

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):

        sample = self.file_list[idx]
        pc_addr = self.npy_file_map[sample]
        pc_addr = os.path.join(self.objaverse_lvis_path,self.npy_file_map[sample])
        data = np.load(pc_addr, allow_pickle=True)
        dict_data = data.item()
        xyz_data = dict_data['xyz']
        rgb_data = dict_data['rgb']

        data = self.pc_norm(xyz_data)
        if self.use_color:
            data = np.concatenate([data, rgb_data], axis=1)

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        data = data.contiguous()

        name = self.lvis_metadata["value_to_key_mapping"][sample]
        label = self.lvis_metadata["key_to_id"][name]

        return data, label, name

    def __len__(self):
        return len(self.file_list)

import collections.abc as container_abcs
int_classes = int
from torch._six import string_classes

import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')

def customized_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(batch, list):
        batch = [example for example in batch ]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return customized_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customized_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customized_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [customized_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class Dataset_3D():
    def __init__(self, args, tokenizer, dataset_type, train_transform=None):
        if dataset_type == 'train':
            self.dataset_name = args.pretrain_dataset_name
        elif dataset_type == 'val':
            self.dataset_name = args.validate_dataset_name
        else:
            raise ValueError("not supported dataset type.")
        with open('./data/dataset_catalog.json', 'r') as f:
            self.dataset_catalog = json.load(f)
            self.dataset_usage = self.dataset_catalog[self.dataset_name]['usage']
            self.dataset_split = self.dataset_catalog[self.dataset_name][self.dataset_usage]
            self.dataset_config_dir = self.dataset_catalog[self.dataset_name]['config']
        self.tokenizer = tokenizer
        self.train_transform = train_transform
        self.pretrain_dataset_prompt = args.pretrain_dataset_prompt
        self.validate_dataset_prompt = args.validate_dataset_prompt
        if 'colored' in args.model.lower():
            self.use_colored_pc = True
        else:
            self.use_colored_pc = False
        if args.npoints == 10000:
            self.use_10k_pc = True
        else:
            self.use_10k_pc = False
        self.build_3d_dataset(args, self.dataset_config_dir)

    def build_3d_dataset(self, args, config):
        config = cfg_from_yaml_file(config)
        config.tokenizer = self.tokenizer
        config.train_transform = self.train_transform
        config.pretrain_dataset_prompt = self.pretrain_dataset_prompt
        config.validate_dataset_prompt = self.validate_dataset_prompt
        config.args = args
        config.use_height = args.use_height
        config.npoints = args.npoints
        config.use_colored_pc = self.use_colored_pc
        config.use_10k_pc = self.use_10k_pc
        config_others = EasyDict({'subset': self.dataset_split, 'whole': True})
        self.dataset = build_dataset_from_cfg(config, config_others)
