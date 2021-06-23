#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import yaml
import cv2
import math
import torch
import scipy
import numpy as np
import pandas as pd
from os.path import join, isfile
from utils import transformations as tf
from utils import transformations_torch as ttf

def read_yaml_file(path):
    """Read content in yaml file"""
    with open(path) as f:
        config=None
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e)
        return config

def read_depth_exr_file(path):
    """ Load depth maps in the folder from the Event Camera Dataset:
    http://rpg.ifi.uzh.ch/davis_data.html """

    import OpenEXR as exr
    import Imath
    exrfile = exr.InputFile(path)
    raw_bytes = exrfile.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = np.reshape(depth_vector, (height, width))
    return depth_map


def loadDataFromFolder(dataset_folder, dataset_format='txt'):
    """ Load events and camera calib from a folder into formats:
    "txt" Format from the Event Camera Dataset: http://rpg.ifi.uzh.ch/davis_data.html
    "numpy" Format from the DENSE dataset: http://rpg.ifi.uzh.ch/E2DEPTH.html"""

    print('Loading data from folder: {}'.format(dataset_folder))

    ev_df = None
    print('Reading events...', end=" ")
    if dataset_format == 'txt':
        if isfile(join(dataset_folder, "events.txt")):
            ev_df = pd.read_csv(join(dataset_folder, "events.txt"), delim_whitespace=True, header=None,
                                names=['time', 'x', 'y', 'polarity'],
                                dtype={'time': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.bool})

            print('[OK]')
        else:
            print('there are not events [FAILED]')

    elif dataset_format == 'numpy':
        events_path = "events/data"
        if os.path.exists(join(dataset_folder, events_path)):
            ev_files = glob.glob(os.path.join(join(dataset_folder, events_path, "*.npy")))
            for ev in sorted(ev_files):
                ev_df = pd.concat([ev_df, pd.DataFrame(np.load(ev), columns=['time', 'x', 'y', 'polarity'])])
            ev_df.reset_index(drop=True)
            # Time is in nanoseconds (CARLA datset)
            ev_df.time = ev_df.time/1e09
            print('[OK]')
        else:
            print('there are not events [FAILED]')

    print('Reading camera calib...', end=" ")
    K = None
    D = None
    if isfile(join(dataset_folder, "calib.txt")):
        raw_calib = np.loadtxt(join(dataset_folder, "calib.txt"))
        K = np.eye(3)
        K[0, 0] = raw_calib[0]
        K[1, 1] = raw_calib[1]
        K[0, 2] = raw_calib[2]
        K[1, 2] = raw_calib[3]
        D = raw_calib[4:]
        print('[OK]')
    else:
        print('there is not calibration file [FAILED].')

    # Load Images
    print('Reading images file...', end=" ")
    images_df, img = None, None
    if dataset_format == 'txt':
        if isfile(join(dataset_folder, "images.txt")):
            images_df = pd.read_csv(join(dataset_folder, "images.txt"), delim_whitespace=True, header=None,
                                    names=['time', 'name'],
                                    dtype={'time': np.float64, 'name': str})
            print('[OK]')
        else:
            print('there is not images file [FAILED]')

        if images_df is not None:
            print('Reading images...', end=" ")
            images_df['data'] = None
            for i, name in enumerate(images_df['name']):
                images_df.at[i, 'data'] = cv2.imread(join(dataset_folder, name), cv2.IMREAD_GRAYSCALE)
            if i + 1 == len(images_df):
                print('read %d images in folder [OK]' % (i))
            img = images_df['data'][0]

    elif dataset_format == 'numpy':
        images_path = "rgb/frames"
        if isfile(join(dataset_folder, images_path, "timestamps.txt")):
            images_df = pd.read_csv(join(dataset_folder, images_path, "timestamps.txt"), delim_whitespace=True, header=None,
                                    names=['name', 'time'],
                                    dtype={'name': np.int32, 'time': np.float64})
            print('[OK]')
        else:
            print('there is not images file [FAILED]')

        if os.path.exists(join(dataset_folder, images_path)) and images_df is not None:
            img_files = glob.glob(os.path.join(join(dataset_folder, images_path, "*.png")))

            print('Reading images...', end=" ")
            images_df['data'] = None
            for i, name in enumerate(sorted(img_files)):
                images_df.at[i, 'data'] = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            img = images_df['data'][0]
            print('read %d images in folder [OK]' % (i))


    # Since the calib.txt file does not contain the sensor size, we get it
    # by loading the first image in the dataset.
    height, width = None, None
    if img is not None:
        height, width = img.shape[:2]

    print('width , height =', width, ',', height)
    print('K =', K)
    print('D =', D)

    # Load depth maps
    print('Reading depthmaps file...', end=" ")
    depthmaps_df = None
    if dataset_format == 'txt':
        if isfile(join(dataset_folder, "depthmaps.txt")):
            depthmaps_df = pd.read_csv(join(dataset_folder, "depthmaps.txt"), delim_whitespace=True, header=None,
                                    names=['time', 'name'],
                                    dtype={'time': np.float64, 'name': str})
            print('[OK]')
        else:
            print('there is not depthmaps file [FAILED]')

        if depthmaps_df is not None:
            print('Reading depthmaps...', end=" ")
            depthmaps_df['data'] = None
            for i, name in enumerate(depthmaps_df['name']):
                _, extension = os.path.splitext(name)
                if extension == '.exr':
                    depthmaps_df.at[i, 'data'] = read_depth_exr_file(join(dataset_folder, name))
                else:
                    depthmaps_df.at[i, 'data'] = cv2.imread(join(dataset_folder, name),
                                                            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if i + 1 == len(depthmaps_df):
                print('read %d depthmaps in folder [OK]' % (i))
    elif dataset_format == 'numpy':
        depthmaps_path = "depth/data"
        if isfile(join(dataset_folder, depthmaps_path, "timestamps.txt")):
            depthmaps_df = pd.read_csv(join(dataset_folder, depthmaps_path, "timestamps.txt"), delim_whitespace=True, header=None,
                                    names=['name', 'time'],
                                    dtype={'name': np.int32, 'time': np.float64})
            print('[OK]')
        else:
            print('there is not depthmaps file [FAILED]')

        if os.path.exists(join(dataset_folder, depthmaps_path)) and depthmaps_df is not None:
            depthmap_files = glob.glob(os.path.join(join(dataset_folder, depthmaps_path, "*.npy")))

            print('Reading depthmaps...', end=" ")
            depthmaps_df['data'] = None
            for i, name in enumerate(sorted(depthmap_files)):
                depthmaps_df.at[i, 'data'] = np.load(name)
            print('read %d depthmaps in folder [OK]' % (i))


    # Load IMU data from text file
    print('Reading IMU data...', end=" ")
    imu_df = None
    if dataset_format == 'txt':
        if isfile(join(dataset_folder, "imu.txt")):
            imu_df = pd.read_csv(join(dataset_folder, "imu.txt"), delim_whitespace=True, header=None,
                                names=['time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz'],
                                dtype={'time': np.float64, 'ax': np.float64, 'ay': np.float64, 'az': np.float64,
                                        'wx': np.float64, 'wy': np.float64, 'wz': np.float64})
            print('[OK]')
        else:
            print('there is not imu file [FAILED]')
    elif dataset_format == 'numpy':
        imu_path = 'pose/imu'
        if isfile(join(dataset_folder, imu_path, "imu.npy")):
            imu_values = np.load(join(dataset_folder, imu_path,"imu.npy"), allow_pickle=True)

            # Check shape issues from pandas to numpy in the CARLA saver
            if len(imu_values.shape) == 1:
                array = np.array([], ndmin=2)
                for i in imu_values:
                    array = np.append(array, i)
                    array = np.reshape(array, (int(len(array)/8), 8))
                imu_values = array

            # Create the pandas dataframe
            imu_df = pd.DataFrame(imu_values, columns=['time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz', 'compass'])
            print('[OK]')
        else:
            print('there is not imu file [FAILED]')

    print('Reading ground truth data...', end=" ")
    poses_df = None
    if dataset_format == 'txt':
        if isfile(join(dataset_folder, "groundtruth.txt")):
            poses_df = pd.read_csv(join(dataset_folder, "groundtruth.txt"), delim_whitespace=True, header=None,
                                names=['time', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'],
                                dtype={'time': np.float64, 'px': np.float64, 'py': np.float64, 'pz': np.float64,
                                        'qx': np.float64, 'qy': np.float64, 'qz': np.float64, 'qw': np.float64})
            print('[OK]')
        else:
            print('there is not ground truth data [FAILED]')
    elif dataset_format == 'numpy':
        poses_path = 'pose/vehicle'
        if isfile(join(dataset_folder, poses_path, "poses.npy")):
            poses_values = np.load(join(dataset_folder, poses_path, "poses.npy"), allow_pickle=True)
            poses_df = pd.DataFrame(poses_values, columns=['time', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])
            print('[OK]')
        else:
            print('there is not ground truth file [FAILED]')

    print('Data loading finished!')
    return ev_df, images_df, depthmaps_df, imu_df, poses_df, K, D, width, height

def events_block_by_images(ts, ev_df):
    blocks = []
    for t, next_t in zip(ts, ts[1:]):
        b = (ev_df[(ev_df.time >= t) & (ev_df.time < next_t)])[['x', 'y', 'time', 'polarity']].to_numpy()
        blocks.append(b)
    return np.array(blocks, dtype='object')

def bootstrap_flow(grad, event_frame):
    ''' Compute optical flow using Lucas-Kanade tracker
        Lucas-Kanade assumes planar scene
        grad is a H x W x 2
        event_frame is H x W x 1
    '''
    Ix, Iy = grad[:, :, 0], grad[:, :, 1]
    if len(Ix.shape) == 3 and Ix.shape[-1] > 1:
        Ix = cv2.cvtColor(Ix, cv2.COLOR_BGR2GRAY)
    if len(Iy.shape) == 3 and Iy.shape[-1] > 1:
        Iy = cv2.cvtColor(Iy, cv2.COLOR_BGR2GRAY)

    s_Ixx = np.sum(Ix * Ix)
    s_Iyy = np.sum(Iy * Iy)
    s_Ixy = np.sum(Ix * Iy)
    s_Ixt = np.sum(Ix * event_frame)
    s_Iyt = np.sum(Iy * event_frame)

    M = np.array([[s_Ixx, s_Ixy], [s_Ixy, s_Iyy]])
    M = -1.0 * M
    b = np.array([s_Ixt, s_Iyt])

    flow = np.linalg.inv(M).dot(b)
    flow_angle = math.atan2(flow[0], flow[1])
    return flow, flow_angle

def split_in_patches(image, coord, patch_radius=7):
    ''' Convinient methods to split an image (Pytorch Tensor) in patches
    centered in the coordinates
    image: the image to split in patches [H x W x C]
    coord: pixel coordinates of the center patch
    patch_radius: radius for the patch
    '''
    assert(image.ndim==3)
    device = image.device
    img_size = image.shape
    num_channels = image.shape[2]

    # Patch size based on radius
    patch_size = 2 * patch_radius + 1

    # Create a bigger  image (padding)
    img = torch.zeros((img_size[0] + patch_size, img_size[1] + patch_size, num_channels)).to(device)
    img[patch_radius:img_size[0]+patch_radius, patch_radius:img_size[1]+patch_radius] = image

    # New coordinate (adjusted to the new image size)
    coord = (coord + patch_radius).type(torch.long)

    # Get the patches for the gradients
    kernel = torch.arange(-patch_radius, patch_radius+1).to(device)
    k1 = kernel.expand(patch_size, patch_size)
    kernel = torch.arange(-patch_radius, patch_radius+1).unsqueeze(-1).to(device)
    k2 = kernel.expand(patch_size, patch_size)
    kernel = torch.stack([k1, k2], dim=-1) # [patch_size x patch_size x 2]
    mask = coord[:, None, None, :] + kernel[None, :, :, :] # [N x patch_size x patch_size x 2] N is the number of points
    patches = img[mask[:, :, :, 1], mask[:, :, :, 0]] # Mask first index, mask second index result: [N x patch_size x patch_size]

    return patches


def klt_tracker(img_grad, kf_coord, ef_frame, ef_coord, img_size, init_mask, patch_radius=7, device='cpu'):
    ''' KLT tracker bewteen keyframe (model) and
    event frame (target)
    return: the delta increment of coord
    '''

    # Mask (only track coord which are still visible)
    coord_mask = init_mask & ((ef_coord[:, 0]>-1) & (ef_coord[:, 1]>-1)) & ((ef_coord[:, 0]<img_size[1]) & (ef_coord[:, 1]<img_size[0]))

    # Patch size based on radius
    patch_size = 2 * patch_radius + 1

    # Create a bigger model image (padding)
    grad = torch.zeros((img_size[0] + patch_size, img_size[1] + patch_size, 2)).to(device)
    grad[patch_radius:img_size[0]+patch_radius, patch_radius:img_size[1]+patch_radius] = img_grad

    # New coordinate (adjusted to the new image size)
    coord = (kf_coord[coord_mask] + patch_radius).type(torch.long)

    # Get the patches for the gradients
    kernel = torch.arange(-patch_radius, patch_radius+1).to(device)
    k1 = kernel.expand(patch_size, patch_size)
    kernel = torch.arange(-patch_radius, patch_radius+1).unsqueeze(-1).to(device)
    k2 = kernel.expand(patch_size, patch_size)
    kernel = torch.stack([k1, k2], dim=-1) # [patch_size x patch_size x 2]
    mask = coord[:, None, None, :] + kernel[None, :, :, :] # [N x patch_size x patch_size x 2] N is the number of points
    grad_patches = grad[mask[:, :, :, 1], mask[:, :, :, 0]] # Mask first index, mask second index result: [N x patch_size x patch_size]

    # event frame 
    event_frame = torch.zeros((img_size[0] + patch_size, img_size[1] + patch_size)).to(device)
    event_frame[patch_radius:img_size[0]+patch_radius, patch_radius:img_size[1]+patch_radius] = ef_frame

    # New coordinate (adjusted to the new image size)
    coord = (ef_coord[coord_mask] + patch_radius).type(torch.long)

    # Get the patches for the events
    kernel = torch.arange(-patch_radius, patch_radius+1).to(device)
    k1 = kernel.expand(patch_size, patch_size)
    kernel = torch.arange(-patch_radius, patch_radius+1).unsqueeze(-1).to(device)
    k2 = kernel.expand(patch_size, patch_size)
    kernel = torch.stack([k1, k2], dim=-1) # [patch_size x patch_size x 2]
    mask = coord[:, None, None, :] + kernel[None, :, :, :] # [N x patch_size x patch_size x 2] N is the number of points
    event_patches = event_frame[mask[:, :, :, 1], mask[:, :, :, 0]] # Mask first index, mask second index result: [N x patch_size x patch_size]

    # Perform KLT
    s_Ixx = torch.sum(grad_patches[:, :, :, 0] * grad_patches[:, :, :, 0], dim=(1,2))
    s_Iyy = torch.sum(grad_patches[:, :, :, 1] * grad_patches[:, :, :, 1], dim=(1,2))
    s_Ixy = torch.sum(grad_patches[:, :, :, 0] * grad_patches[:, :, :, 1], dim=(1,2))
    s_Ixt = torch.sum(grad_patches[:, :, :, 0] * event_patches, dim=(1,2))
    s_Iyt = torch.sum(grad_patches[:, :, :, 1] * event_patches, dim=(1,2))

    M = torch.zeros((coord.shape[0], 2, 2), device=device)
    M[:, 0, 0], M[:, 0, 1] = s_Ixx, s_Ixy
    M[:, 1, 0], M[:, 1, 1] = s_Ixy, s_Iyy
    b = torch.cat([s_Ixt.unsqueeze(-1), s_Iyt.unsqueeze(-1)], dim=1).unsqueeze(-1)

    flow = torch.bmm(torch.inverse(M), -b)
    return flow.reshape(flow.shape[0], flow.shape[1]), coord_mask

def ssd_brighness_response_frame(kf_frame, ef_frame, x_coord,  patch_radius=7, device='cpu'):
    ''' Compute the SSD brightness response between keyframe (brightness model)
    and event frame (measurement model)
    '''
    img_size = kf_frame.shape
    ssd_response = torch.zeros(img_size, device=device)

    # Patch size based on radius
    patch_size = 2 * patch_radius + 1

    # Create the keyframe with patch marging
    key_frame = torch.zeros((img_size[0] + patch_size, img_size[1] + patch_size)).to(device)
    key_frame[patch_radius:img_size[0]+patch_radius, patch_radius:img_size[1]+patch_radius] = kf_frame

    # Create the eventframe with patch marging
    event_frame = torch.zeros((img_size[0] + patch_size, img_size[1] + patch_size)).to(device)
    event_frame[patch_radius:img_size[0]+patch_radius, patch_radius:img_size[1]+patch_radius] = ef_frame

    # Patch template around x_coord in key frame
    center = x_coord + patch_radius # center in the new key_frame
    x_patch = key_frame[center[1]-patch_radius:center[1]+patch_radius+1, center[0]-patch_radius:center[0]+patch_radius+1]

    # Compute the SSD
    for i in range(patch_radius, img_size[0]+patch_radius):
        for j in range(patch_radius, img_size[1]+patch_radius):
            template = event_frame[i-patch_radius:i+patch_radius+1, j-patch_radius: j+patch_radius+1]
            ssd_response[i-patch_radius, j-patch_radius] =  ((template[None, :, :] - x_patch)**2).sum()
    
    return ssd_response

def flow_from_sensitivity_matrix_iter(keypoints, v, w):
    ''' Estimates flow from linear and angular velocities
        using the sensitivity matrix (iterative version)
    '''
    flow_map = []
    velo = np.hstack([v, w])
    for p in keypoints:
        J = np.array([[-1.0 / p[2], 0, p[0] / p[2], p[0] * p[1], -(1 + pow(p[0], 2)), p[1]],
                      [0, -1.0 / p[2], p[1] / p[2], 1 + pow(p[1], 2), -p[0] * p[1], -p[0]]])
        flow = J.dot(velo)
        flow_map.append(flow)
    return np.array(flow_map)


def flow_from_sensitivity_matrix(keypoints, v, w):
    ''' Estimates flow from linear and angular velocities
        using the sensitivity matrix
    '''
    jacobs = np.array([np.array([[-1.0 / p[2], 0, p[0] / p[2], p[0] * p[1], -(1 + p[0] ** 2), p[1]],
                                 [0, -1.0 / p[2], p[1] / p[2], 1 + p[1] ** 2, -p[0] * p[1], -p[0]]]) for p in
                       keypoints])
    velo = np.hstack([v, w])
    return np.dot(jacobs, velo)

def flow_from_sensitivity_matrix_torch(coord, idp, v, w, device='cpu'):
        ''' Estimates flow from linear and angular velocities
            using the sensitivity matrix (2x6 jacobian)
            v = [3] linear velocity
            w = [3] angular velocity
            return [N x 2 x 1] N times 2 x 1 optical flow
        '''
        # The tensor of velocities
        velo = torch.cat([v, w]).unsqueeze(-1).type(torch.float32) #[6 x 1] velocities vector
        velo = velo.to(device)
        print("velo in sensitivity matrix: ", velo)

        # The Tensor of Jacobians
        jacobs = torch.zeros(coord.shape[0], 2, 6, dtype=torch.float32).to(device) # [N x 2 x 6]

        # First row
        jacobs[:, 0, 0] = -idp
        jacobs[:, 0, 2] = coord[:, 0] * idp
        jacobs[:, 0, 3] = coord[:, 0] * coord[:, 1]
        jacobs[:, 0, 4] = -(1.0 + coord[:, 0]**2)
        jacobs[:, 0, 5] = coord[:, 1]

        # Second row
        jacobs[:, 1, 1] = -idp
        jacobs[:, 1, 2] = coord[:, 1] * idp
        jacobs[:, 1, 3] = 1.0 + coord[:, 1]**2
        jacobs[:, 1, 4] = -coord[:, 0] * coord[:, 1]
        jacobs[:, 1, 5] = -coord[:, 0]

        return torch.matmul(jacobs, velo) # N x 2 x 1]


def velo_from_sensitivity_matrix(keypoints, flow):
    ''' Estimates linear and angular velocities from flow
        using the sensitivity matrix
        flow = [N x 2] where N = W x H
        return mean of linear and angular velocities
    '''
    jacobs = np.array([np.array([[-1.0 / p[2], 0, p[0] / p[2], p[0] * p[1], -(1 + pow(p[0], 2)), p[1]],
                                 [0, -1.0 / p[2], p[1] / p[2], 1 + pow(p[1], 2), -p[0] * p[1], -p[0]]]) for p in
                       keypoints])
    return np.dot(np.linalg.pinv(jacobs), flow)


def jacobian_points_pi(kp):
    ''' Jacobian of the projection (pi)
    kp = [N x 3] where N is number of points
    return [N x 2 x 3] N times 2x3 jacobian
    '''
    return np.array([np.array([[1.0 / p[2], 0, -p[0] / p[2]], [0, 1.0 / p[2], -p[1] / p[2]]]) for p in kp])


def jacobian_points_invpi(kp):
    ''' Jacobian of the inverse projection (in_inv)
    kp = [N x 3] where N is number of points
    return [N x 3 x 2] N times 3x2 jacobian
    '''
    return np.array([np.array([[p[2], 0], [0, p[2]], [0, 0]]) for p in kp])


def imu_from_poses(r_init, poses_df):
    ''' Return IMU in the form of vx, vy, vz, wx, wy, wz '''
    poses = poses_df[['time', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw']].to_numpy()
    imu = []
    for p_cur, p_next in zip(poses[0:], poses[1:]):
        dt = p_next[0] - p_cur[0]
        v = np.dot(r_init.transpose(), (p_next[1:4] - p_cur[1:4])) / dt  # velocity in the camera frame
        delta_q = tf.quaternion_multiply(tf.quaternion_inverse(p_cur[4:8]), p_next[4:8])  # q_camera(t-1)_camera(t)
        w = tf.logmap_so3(tf.matrix_from_quaternion(delta_q)) / dt
        imu.append(np.hstack([p_cur[0], v, w]))  # [time, vx, vy, vz, wx, wy, wz]
    return np.array(imu)


def velo_from_imu(imu, delta_time, events_first_time, events_last_time, pose=None, delta_pose=None, prev_v=np.array([]), prev_w=np.array([]), device='cpu'):

    gyro_in_between = imu[(imu[:, 0] >= events_first_time) & (imu[:, 0] <= events_last_time)][:, 4:7]  # angular velocity
    velo_in_between = imu[(imu[:, 0] >= events_first_time) & (imu[:, 0] <= events_last_time)][:, 1:4]  # linear velocity
 
    # Get the mean of the angular and linear velocity
    if gyro_in_between.shape[0] is not 0:
        w = torch.from_numpy(np.mean(gyro_in_between, axis=0)).to(device)
    else:
        w = prev_w
    if velo_in_between.shape[0] is not 0:
        v = torch.from_numpy(np.mean(velo_in_between, axis=0)).to(device)
    else:
        v = prev_v

    print("v {} w {}".format(v, w))

    if pose != None and delta_pose != None:
        assert(pose.shape == (4, 4))
        assert(delta_pose.shape == (4, 4))
        g = torch.tensor([[0.0], [0.0], [9.81]], device=device) #gravity in world
        print("g_world: ", g)
        g = torch.matmul(torch.inverse(pose)[:3, :3], g).reshape(3) # gravity in KF
        print("g_kf: ", g)
        v = v - g # substract gravity from acceleration
        print("acc - gravity: ", v)
        dt = events_last_time - events_first_time
        print("delta_time {} and dt {}".format(delta_time, dt))

        # Linear velocity = delta pose * delta_time_to_kf + a * dt
        v_k_1 = (delta_pose[:3, 3] / delta_time)
        print("v_k-1:",  v_k_1)
        v = v_k_1 + v * dt
        print("final v: ", v)

    return v, w


def deg2rad(x):
    return (x * math.pi) / 180.0


def rad2deg(x):
    return (x * 180.0) / math.pi

def degto360(x):
    return (x % 360.00)

def rad2deg360(x: float):
    if type(x) is np.ndarray:
        return np.array([rad2deg(e) if e >=0 else 360.00 + rad2deg(e) for e in x])
    else:
        if x >= 0:
            angle = rad2deg(x) 
        else:
            angle = 360.0 + rad2deg(x)
            #angle = (2.0*math.pi + x) * 360.0 / (2*math.pi)
        return angle

def events_format(events_array, in_txyp=True, out_xytp=True):
    if in_txyp is True and out_xytp is not True:
        return events_array

    if in_txyp:
        t = events_array[:, 0]
        x = events_array[:, 1]
        y = events_array[:, 2]
        p = events_array[:, 3]
    else:
        x = events_array[:, 0]
        y = events_array[:, 1]
        t = events_array[:, 2]
        p = events_array[:, 3]

    if out_xytp:
        res = np.array([x, y, t, p]).transpose()
    else:
        res = np.array([t, x, y, p]).transpose()
    return res


def imu_parser(imu, with_timestamp=False):
    gyro = []
    acc = []
    heading = []
    i = 1 if with_timestamp else 0
    for values in imu:
        gyro.append(values[i:i + 3])
        acc.append(values[i + 3:i + 6])
        heading.append(values[i + 6])
    return np.array(gyro, dtype=np.float32), np.array(acc, dtype=np.float32), np.array(heading, dtype=np.float32)


def poses_parser(poses, with_timestamp=False):
    ang_velo = []
    lin_velo = []
    orient = []
    trans = []
    i = 1 if with_timestamp else 0
    for values in poses:
        ang_velo.append(values[i])
        lin_velo.append(values[i + 1])
        orient.append(values[i + 2])
        trans.append(values[i + 3])
    return np.array(ang_velo, dtype=np.float32), np.array(lin_velo, dtype=np.float32), np.array(orient,
                                                                                                dtype=np.float32), np.array(
        trans, dtype=np.float32)

def censusTransform(img, inv=False):
    h, w = img.shape
    # print('image size: %d x %d = %d' % (w, h, w * h))

    # Initialize output array
    census = np.zeros((h - 2, w - 2), dtype='uint8')

    # centre pixels, which are offset by (1, 1)
    cp = img[1:h - 1, 1:w - 1]

    # offsets of non-central pixels
    offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]

    # Do the pixel comparisons
    for u, v in offsets:
        census = (census << 1) | (img[v:v + h - 2, u:u + w - 2] >= cp)

    if inv:
        return -(census - np.max(census))
    else:
        return census


def cornerResponse(img, patchsize=9, method="Harris", kappa=0.08):
    from scipy import signal

    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Ix = signal.convolve2d(img, Sx, boundary='symm', mode='same')
    Iy = signal.convolve2d(img, Sy, boundary='symm', mode='same')
    Ixx = cv2.pow(Ix, 2.0)
    Iyy = cv2.pow(Iy, 2.0)
    Ixy = cv2.multiply(Ix, Iy)

    patch = np.ones((9, 9), dtype=np.float32)
    pr = np.floor(9 / 2.0)

    sIxx = signal.convolve2d(Ixx, patch, boundary='symm', mode='same')
    sIyy = signal.convolve2d(Iyy, patch, boundary='symm', mode='same')
    sIxy = signal.convolve2d(Ixy, patch, boundary='symm', mode='same')

    trace = sIxx + sIyy
    determinant = cv2.multiply(sIxx, sIyy)
    sIxy_pow = cv2.pow(sIxy, 2.0)

    if method is "Harris":
        scores = (determinant - sIxy_pow) - kappa * cv2.pow(trace, 2.0)
    else:
        determinant = cv2.subtract(determinant, sIxy_pow)
        half_trace = trace / 2.0
        determinant = determinant.astype(dtype=np.float32)
        scores = trace / 2.0 - cv2.sqrt(cv2.pow(half_trace, 2.0) - determinant)

    scores[np.isnan(scores)] = 0.0
    scores[scores <= 0] = 0.0
    return scores


def warpEventsHomography(events_bearings, w, v, n, K, warp_to_begin=True, batch_size=1024):
    """Warps a set of events according to an instantaneous homography and returns the warped events.
    :param events_bearings: A [Nx4] array containing one event per row in the form with the coordinates expressed as bearing vectors: [f_x,f_y,t,p]
    :param w: A [3x1] array containing the angular velocity
    :param v: A [3x1] array containing the linear velocity normalized by the distance to the plane
    :param n: A [3x1] array containing the unit normal of the plane
    :param K: A [3x3] matrix containing the destination camera matrix
    :param warp_to_begin: If True, warp the events to the first timestamp,
                          if False, warp the events to the last timestamp
    :param batch_size: For speed reasons, events are grouped in batches and
                       assigned an identical timestamp across the batch.
                       This parameter controls the number of events in each such batch.
    
    Assuming that 'ref' is the reference frame [I | 0], then
    the homography H_cur_ref(t) that transfers a pixel position x_ref to the image plane
    of a camera position 'cur' at a time t is given by:
    
        H_cur_ref(t) = (R - T n^T) where R = exp(w * t) and T = v * t
    
    E.g. to warp an event (x_cur, t) to the reference image plane:

        x_ref ~ K_ref * H_cur_ref(t)^-1 * x_cur
    
    """

    t0 = events_bearings[0, 2]
    delta_t = events_bearings[-1, 2] - t0
    warped_events = np.zeros(events_bearings.shape)
    warped_events[:, 2:] = events_bearings[:, 2:]

    eps = 1e-8
    v = v.reshape((3, 1))
    n = n.reshape((3, 1))

    if warp_to_begin:
        H_ref_begin = np.eye(3)
    else:
        R_end_begin = tf.expmap_so3(w * delta_t + eps)
        T_end_begin = v * delta_t
        H_end_begin = R_end_begin + T_end_begin.dot(n.T)
        H_ref_begin = H_end_begin

    I3 = np.eye(3)
    Sw = tf.skew(w)

    # Split the data into batches of K events (with K being a small number)
    # and use the same timestamp (the first event timestamp) for the entire batch
    num_events = events_bearings.shape[0]
    events_batch_homogeneous = np.ones((batch_size, 3))
    i = 0
    treated_all_events = False
    while not treated_all_events:

        if i + batch_size >= num_events:
            # We are about to process the rest of the events,
            # those that did not fit in a packet of size batch_size
            idx_begin, idx_end = i, events_bearings.shape[0]
            events_batch_homogeneous = np.ones((events_bearings[i:].shape[0], 3))
            treated_all_events = True
        else:
            idx_begin, idx_end = i, i + batch_size

        events_batch_homogeneous[:, :2] = events_bearings[idx_begin:idx_end, :2]
        t = events_bearings[i, 2] - t0

        T = v * t  # T_cur_begin
        # R = tf.expmap_so3(w * t + eps) # R_cur_begin
        R = I3 + Sw * t
        H_cur_begin = R + T.dot(n.T)
        H_begin_cur = np.linalg.inv(H_cur_begin)

        H_ref_cur = H_ref_begin.dot(H_begin_cur)

        events_batch_warped = events_batch_homogeneous.dot((K.dot(H_ref_cur)).T)
        # print (events_batch_warped)
        events_batch_warped[:, 0] = events_batch_warped[:, 0] / events_batch_warped[:, 2]
        events_batch_warped[:, 1] = events_batch_warped[:, 1] / events_batch_warped[:, 2]

        warped_events[idx_begin:idx_end, :2] = events_batch_warped[:, :2]

        i += batch_size

    return warped_events


def getEventFrame(batch, pos, patch, sigma, corners=None):
    """
    Construct an event frame from events in batch around the position *pos* and side length *2*patch+1*.
    Each entry (i,j) of the event frame denotes the sum of the polarities of all events which fall on that pixel in
    the batch.
    :param batch: data array of events (Nx4) with columns [x_coord, y_coord, timestamp, polarity]
    :param pos: position of the patch (2,)
    :param patch: half length of the patch
    :param sigma: smoothing parameter used to smooth the resulting event frame. This is to speed up convergence.
    :param corners: corners that define the area around which the histogram is filled. Is constructed from pos and patch
    if not given.
    :return: Event frame (image_heigth x image_width) with accumulated events in the pixels.
    """
    if corners is None:
        # find boundaries of the patch. Must be ints
        x0, y0 = np.floor(pos[0] - patch), np.floor(pos[1] - patch)
        x1, y1 = x0 + 2 * patch + 1, y0 + 2 * patch + 1
    else:
        x0 = np.min(corners[:, 0])
        x1 = np.max(corners[:, 0])
        y0 = np.min(corners[:, 1])
        y1 = np.max(corners[:, 1])

    # compute bin ranges for histogram. +1 and -0.5 are used since the ranges must contain all
    # whole numbers x0,...,x1 within their boundaries
    x_range = np.arange(x0, x1 + 1) - 0.5
    y_range = np.arange(y0, y1 + 1) - 0.5

    # compute event frame using histogram2d and using different weights for the datapoints depending on polarity
    event_frame = np.histogram2d(batch[:, 0], batch[:, 1], bins=(x_range, y_range), weights=batch[:, 3])

    # blur and transpose
    blurred_frame = filters.gaussian_filter(event_frame[0].transpose(), sigma)

    return blurred_frame


def vector_to_mat(p):
    """
    Convert parameters into (3x3) matrix as follows:

                                    a1 a2 a5
    [a1, a2, a3, a4, a5, a6] <->  a3 a4 a6
                                    0  0  1
    :param p: row of shape (6,)
    :return: matrix of shape (3x3)
    """
    array = np.eye(3, dtype="float64")
    array[:2, :2] = p[:4].reshape((2, 2))
    array[:2, 2] = p[4:]

    return array


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1, array[idx - 1]
    else:
        return idx, array[idx]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_file_and_remove(path, filename):
    if os.path.exists(os.path.join(path,filename)):
        value = input('File: "{}" already exists in {}. Overwrite (y/n)?: '.format(filename, path))
        if value.lower() == 'y':
            os.remove(os.path.join(path, filename))
        else:
            return False
    return True

def compensatedBlock(events, imu, lin_vel=None, K=None):
    """
    events in format [x, y, t, p] in pixel coordinates when K is given
    imu in format [ax, ay, az, wx, wy, wz]
    lin_vel in format [vx, vy, vz] in camera frame
    """
    compensated_events = np.zeros(events.shape)
    t0 = events[0, 2]  # start timestamp

    for i, e in enumerate(events):
        dt = e[2] - t0  # delta time = current time - start time
        R = tf.expmap_so3(imu[3:] * dt)
        x_m = np.zeros(imu.shape)
        x_m[:4] = R[:2, :2].reshape(1, 4)
        if lin_vel is not None:
            x_m[4:] = lin_vel[:2] * dt
        else:
            x_m[4:] = imu[:2] * dt * dt
        # print (x_m[4:])
        compensated_events[i, :2] = Warp(x_m)(e[:2], K=K)
        compensated_events[i, 2:] = e[2:]

    return compensated_events

def build_pyramids(img, num_levels=4):
    imgs = []
    imgs.append(img)
    for i in range(1, num_levels):
        imgs.append(cv2.pyrDown(imgs[i-1]))

    return imgs

def create_search_tree(x):
    import scipy.spatial
    if x.is_cuda:
        tree = scipy.spatial.cKDTree(x.detach().cpu().numpy())
    else:
        tree = scipy.spatial.cKDTree(x.detach().numpy())
    return tree

def gives_k_closest_points(x, y, k=[1]):
    tree = create_search_tree(x)
    if y.is_cuda:
        _, ii = tree.query(y.detach().cpu().numpy(), k=k)
    else:
        _, ii = tree.query(y.detach().numpy(), k=k)
    return ii


def compute_epipolar_lines(t_delta, r_delta, x_kf, K, device='cpu'):
    # Check if the arguments come in tensors
    t = t_delta if type(t_delta) is torch.Tensor else torch.from_numpy(t_delta).type(torch.float32).to(device)
    r = r_delta if type(r_delta) is torch.Tensor else torch.from_numpy(r_delta).type(torch.float32).to(device)

    # Pixel coordinates in homogenous
    x_kf = torch.cat([x_kf, torch.ones((x_kf.shape[0], 1), device=device)], dim=1) # [N x 3]
    x_kf.unsqueeze_(-1) # [N x 3 x 1]

    # Compute Fundamental matrix
    t_cross = torch.tensor([[0.0, -t[2,0], t[1,0]], [t[2,0], 0.0, -t[0,0]], [-t[1,0], t[0,0], 0.0]], device=device)
    R = ttf.expmap_so3(r, device=device) # [3 x 3] Rotation matrix
    E = torch.matmul(t_cross, R)
    F = torch.matmul(torch.matmul(torch.inverse(K).t(), E),torch.inverse(K))

    # Epipolar lines in event frame
    l_ef = torch.matmul(F, x_kf)

    return l_ef, F


def compute_points_on_epiline(line, img_size, device='cpu'):
    y, x = torch.arange(0, img_size[0], 1, dtype=torch.long).to(device), torch.arange(0, img_size[1], 1, dtype=torch.long).to(device)

    x_from_y = -(line[2]+line[1]*y)/line[0]
    mask = (x_from_y >= 0) & (x_from_y < img_size[1])
    pts1 = torch.cat([x_from_y[mask], y[mask]], dim=0).t().type(torch.long)

    y_from_x = -(line[2]+line[0]*x)/line[1]
    mask = (y_from_x >= 0) & (y_from_x < img_size[0])
    pts2 = torch.cat([x[mask], y_from_x[mask]], dim=0 ).t().type(torch.long)

    pts = torch.cat([pts1, pts2], dim=0)
    return pts

def restrict_epi_search(x, t_delta, r_delta, depth, sigma, epi_pts, K, method='norm', device='cpu'):
    """ Reduce the number of points along the epiline based on the depth uncertainty
    x: pixel coordinate
    """
    # Get intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
 
    # Project (inv_pi)the px coord to the 3D point base on the min and max depth
    d_min, d_max = depth-sigma, depth+sigma
    kp = torch.zeros(2, 3, dtype=torch.float32).to(device) # [2 x 3]
    kp[0, 2], kp[1, 2] = d_min, d_max 
    kp[:, 0] = (x[0] - cx) * (kp[:, 2] / fx)
    kp[:, 1] = (x[1] - cy) * (kp[:, 2] / fy)
    kp.unsqueeze_(-1) # [2 x 3 x 1]

    # Rotation matrix T_EF_KF to convert points in KF to points in EF
    R = ttf.expmap_so3(r_delta, device=device) # [3 x 3] Rotation matrix
    kp = torch.matmul(R, kp)# [2 x 3 x 1] The Min and Max points in Event frame
    kp = kp + t_delta

    # Project (pi) the points back to the EF
    x_prime = torch.zeros((kp.shape[0], 2)).to(device)
    x_prime[:, 0] = fx * (kp[:, 0, 0] / kp[:, 2, 0]) + cx
    x_prime[:, 1] = fy * (kp[:, 1, 0] / kp[:, 2, 0]) + cy

    if method == 'norm':
        # Norm of the projected points
        x_norm_prime, _ = torch.sort(torch.norm(x_prime, dim=1))
        # Norm of the epipoints
        epi_pts_norm = torch.norm(epi_pts.type(torch.float), dim=1)
        # Mask for the epipolar points
        mask_epi_pts = (epi_pts_norm>=x_norm_prime[0]) & (epi_pts_norm<=x_norm_prime[1])
    else:
        # X' in integer and round to the biggest int
        x_prime[0,:] = torch.floor(x_prime[0,:])
        x_prime[1,:] = torch.ceil(x_prime[1,:])
        x_prime = x_prime.type(torch.long)
        # Restrict the epipoints to the ones withing this range
        mask_min = (epi_pts[:,0]>=x_prime[0,0]) & (epi_pts[:,1]>=x_prime[0,1])
        mask_max = (epi_pts[:,0]<=x_prime[1,0]) & (epi_pts[:,1]<=x_prime[1,1])
        mask_epi_pts = mask_min & mask_max

    return epi_pts[mask_epi_pts]

def ssd_along_epiline(img_model, coord_model, target_img, epi_pts, patch_radius=7, device='cpu'):
    """
    """
    img_size = img_model.shape

    # Patch size based on radius
    patch_size = 2 * patch_radius + 1

    # Create a bigger model image (padding)
    img = torch.zeros((img_size[0] + patch_size, img_size[1] + patch_size)).to(device)
    img[patch_radius:img_size[0]+patch_radius, patch_radius:img_size[1]+patch_radius] = img_model

    # Translate the point coord to the new image resolution
    x_coord = (coord_model + patch_radius).type(torch.long)

    # Get the model image template
    target_patch =  img[x_coord[1]-patch_radius:x_coord[1]+patch_radius+1 , x_coord[0]-patch_radius: x_coord[0]+patch_radius+1]

    # Create the bigger target image (padding)
    img = torch.zeros((img_size[0] + patch_size, img_size[1] + patch_size)).to(device)
    img[patch_radius:img_size[0]+patch_radius, patch_radius:img_size[1]+patch_radius] = target_img

    # Translate the epi points to the new image resolution
    target_pts = (epi_pts + patch_radius).type(torch.long)

    # Get the templates from the target image
    kernel = torch.arange(-patch_radius, patch_radius+1).to(device)
    k1 = kernel.expand(patch_size, patch_size)
    kernel = torch.arange(-patch_radius, patch_radius+1).unsqueeze(-1).to(device)
    k2 = kernel.expand(patch_size, patch_size)
    kernel = torch.stack([k1, k2], dim=-1) # [patch_size x patch_size x 2]
    mask = target_pts[:, None, None, :] + kernel[None, :, :, :] # [N x patch_size x patch_size x 2] N is the number of points
    patches = img[mask[:, :, :, 1], mask[:, :, :, 0]] # Mask first index, mask second index result: [N x patch_size x patch_size]

    ssd = ((target_patch[None, :, :] - patches)**2).sum(dim=[1,2])

    return target_patch, patches, ssd

def inv_depth_param_update(T_ef_kf, p_kf):
    '''
    Inverse distance computation as explained in the report Local tracking and Mapping
    for Direct Visual SLAM (Pablo Rodriguez Palafox)
    T_ef_kf: transformation p_ef = T_ef_kf * p_kf
    p_kf: N times [4x1] unnormalized homogenous vectors (bearing points [bx, by, bz dp])
    return the inverse depth
    '''
    # Extract rotation and translation
    R, t = T_ef_kf[:3, :3], T_ef_kf[:3, 3].unsqueeze(-1)

    # Expanded rotation rows [N x 1 x 3]
    r0, r1, _= R[0, :].expand(p_kf.shape[0], 3).unsqueeze(1), R[1, :].expand(p_kf.shape[0], 3).unsqueeze(1), R[2, :].expand(p_kf.shape[0], 3).unsqueeze(1)

    tmp = (p_kf[:, :3]/p_kf[:, 3][:,None]).unsqueeze(-1) #bearing normalized unit points
    p_ef = torch.matmul(R, tmp) # rotate the bearing points
    p_ef = p_ef + t # [N x 3 x 1] points in the EF
    b_ef = p_ef / torch.norm(p_ef, dim=1)[:, None]

    b_kf = p_kf[:, :3] # bearing in the KF
    # reshape
    b_ef = b_ef.reshape(b_ef.shape[0], b_ef.shape[1])
    p_ef = p_ef.reshape(p_ef.shape[0], p_ef.shape[1])

    # Calculate the inverse distance
    r0_b, r1_b = torch.bmm(r0, b_kf.unsqueeze(-1)).reshape(p_kf.shape[0]), torch.bmm(r1, b_kf.unsqueeze(-1)).reshape(p_kf.shape[0])
    dp = (b_ef[:, 1]*r0_b - b_ef[:, 0]*r1_b)/(b_ef[:, 0]*t[1] - b_ef[:, 1]*t[0])

    return dp

def inv_depth_estimation(T_ef_kf, p_kf, b_ef):
    '''
    Inverse distance computation as explained in the report Local tracking and Mapping
    for Direct Visual SLAM (Pablo Rodriguez Palafox)
    T_ef_kf: transformation p_ef = T_ef_kf * p_kf
    p_kf: N times [4x1] (N x 4) unnormalized homogenous vectors in keyframe (bearing points [bx, by, bz dp])
    b_ef: N times [3x1] (N x 3) bearing vector in the event frame ([bx, by, bz])
    return nonething (the dp element in p_kf is updated)
    '''
    # Extract rotation and translation
    R, t = T_ef_kf[:3, :3], T_ef_kf[:3, 3].unsqueeze(-1)

    # Expanded rotation rows [N x 1 x 3]
    r0, r1, _= R[0, :].expand(p_kf.shape[0], 3).unsqueeze(1), R[1, :].expand(p_kf.shape[0], 3).unsqueeze(1), R[2, :].expand(p_kf.shape[0], 3).unsqueeze(1)

    b_kf = p_kf[:, :3] # bearing in the KF [3 first elemnet of p_kf]

    # Calculate the inverse distance
    r0_b, r1_b = torch.bmm(r0, b_kf.unsqueeze(-1)).reshape(p_kf.shape[0]), torch.bmm(r1, b_kf.unsqueeze(-1)).reshape(p_kf.shape[0])
    dp = (b_ef[:, 1]*r0_b - b_ef[:, 0]*r1_b)/(b_ef[:, 0]*t[1] - b_ef[:, 1]*t[0])

    return dp #p_kf[:, 3] = dp


def matrix_to_angle_axis_translation(T, invert=False):
    assert(T.shape == (4,4))
    T = T.copy()
    if invert == True:
        T = np.linalg.inv(T)
    angle_axis = tf.logmap_so3(T[:3, :3])
    translation = T[:3,3]
    return np.hstack([angle_axis, translation])


def coord_dilation_search(xp, coord, patch_radius=1, device='cpu'):
    ''' Search a image pixel in to a coord map along a patch_radius dilation
    xp is the pixle coordinate to search 
    coord is the 2D map of coordinates where you search
    patch_radius, is the radius used to dilate
    '''
    # Patch size from radius
    patch_size = 2* patch_radius + 1

    # Make the kernel of dilation
    kernel = torch.arange(-patch_radius, patch_radius+1).to(device)
    k1 = kernel.expand(patch_size, patch_size)
    kernel = torch.arange(-patch_radius, patch_radius+1).unsqueeze(-1).to(device)
    k2 = kernel.expand(patch_size, patch_size)
    kernel = torch.stack([k1, k2], dim=-1) # [patch_size x patch_size x 2]

    # Dilate the xp point
    dilated_xp = xp + kernel[:, :] # [patch_size x patch_size x 2] dilated points
    # Reshape
    dilated_xp = dilated_xp.reshape(patch_size * patch_size, 2) #[patch_size^2 x 2]

    # Find the dilated points
    for d_xp in dilated_xp:
        index = ((coord[:, 0] == d_xp[0]) & (coord[:, 1] == d_xp[1])).nonzero(as_tuple=True)[0]
        if index.nelement() > 0:
            return d_xp, index[0]

    return xp, None


def bilinear_interpolate(image, samples_x, samples_y):
    '''It perform bilinear interpolation using a number of pixel coordinates
    image is the tensor input [H x W x C]
    samples_x, x pixel coordinates (along W) in float [1 x N]
    samples_y, y pixel coordinates (along H) in float [1 x N]
    '''
    assert(image.ndim == 3)
    assert(samples_x.ndim == 2)
    assert(samples_y.ndim == 2)
    assert(samples_x.shape == samples_y.shape)
                                                # input image is: W x H x C
    image = image.permute(2,0,1)                # change to:      C x W x H
    image = image.unsqueeze(0)                  # change to:  1 x C x W x H
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)
    samples = torch.cat([samples_x, samples_y],3)
    samples[:,:,:,0] = (samples[:,:,:,0]/(image.shape[3]-1)) # normalize to between  0 and 1
    samples[:,:,:,1] = (samples[:,:,:,1]/(image.shape[2]-1)) # normalize to between  0 and 1
    samples = samples*2-1                       # normalize to between -1 and 1
    result = torch.nn.functional.grid_sample(image, samples, mode='bilinear', align_corners=True)
    result = result.reshape(result.shape[1], result.shape[2]) # [1 x N]
    return result.permute(1,0) # return [N x 1]


def fill_missing_values(target_for_interp, method='nearest'):
    ''' Fill missing values in a 2D tensor
    returns a numpy array
    '''
    # convert to numpy
    target_for_interp = target_for_interp.copy() if type(target_for_interp) is not torch.Tensor else target_for_interp.detach().cpu().numpy()

    import scipy.interpolate

    if method == 'nearest': interpolator=scipy.interpolate.NearestNDInterpolator
    elif method == 'linear': interpolator=scipy.interpolate.LinearNDInterpolator
    elif method == 'cubic': interpolator=scipy.interpolate.CloughTocher2DInterpolator
    else: return None

    def getPixelsForInterp(img):
        """
        Calculates a mask of pixels neighboring invalid values - 
        to use for interpolation. 
        """
        # mask invalid pixels
        invalid_mask = np.isnan(img) + (img == 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        #dilate to mark borders around invalid regions
        dilated_mask = cv2.dilate(invalid_mask.astype('uint8'), kernel,
                        borderType=cv2.BORDER_CONSTANT, borderValue=int(0))

        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  ~invalid_mask
        return masked_for_interp.astype('bool'), invalid_mask

    # Mask pixels for interpolation
    mask_for_interp, invalid_mask = getPixelsForInterp(target_for_interp)

    # Interpolate only holes, only using these pixels
    points = np.argwhere(mask_for_interp)
    values = target_for_interp[mask_for_interp]
    interp = interpolator(points, values)

    target_for_interp[invalid_mask] = interp(np.argwhere(invalid_mask))
    return target_for_interp

def inv_translation_angle_axis(pose_vector):
    assert(len(pose_vector) == 6)
    T = np.eye(4)
    T[:3, 3] = pose_vector[:3].reshape(3)
    T[:3, :3] = tf.expmap_so3(pose_vector[3:6])

    T = np.linalg.inv(T)
    w = tf.logmap_so3(T[:3, :3])
    return np.concatenate([T[:3, 3].reshape(3), w], axis=0)

def inv_angle_axis_translation(pose_vector):
    assert(len(pose_vector) == 6)
    T = np.eye(4)
    T[:3, 3] = pose_vector[3:6].reshape(3)
    T[:3, :3] = tf.expmap_so3(pose_vector[:3])

    T = np.linalg.inv(T)
    w = tf.logmap_so3(T[:3, :3])
    return np.concatenate([w, T[:3, 3].reshape(3)], axis=0)

def inv_translation_quaternion(pose_vector):
    assert(len(pose_vector) == 7)
    T = np.eye(4)
    T[:3, 3] = pose_vector[:3].reshape(3)
    T[:3, :3] = tf.matrix_from_quaternion(pose_vector[3:7])[:3, :3]

    T = np.linalg.inv(T)
    q = tf.quaternion_from_matrix(T)
    return np.concatenate([T[:3, 3].reshape(3), q], axis=0)

def inv_quaternion_translation(pose_vector):
    assert(len(pose_vector) == 7)
    T = np.eye(4)
    T[:3, 3] = pose_vector[4:7].reshape(3)
    T[:3, :3] = tf.matrix_from_quaternion(pose_vector[:4])[:3, :3]

    T = np.linalg.inv(T)
    q = tf.quaternion_from_matrix(T)
    return np.concatenate([q, T[:3, 3].reshape(3)], axis=0)
