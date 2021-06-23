#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import math
import torch
from matplotlib import pyplot as plt

from utils.functions import rad2deg360, inv_translation_angle_axis, inv_angle_axis_translation, inv_quaternion_translation, inv_translation_quaternion
import utils.transformations as tf

# Define the Brightmess model color map

#Option1: Blue negative, Green positive
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
top = cm.get_cmap('Blues_r', 128)
bottom = cm.get_cmap('Greens', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),  bottom(np.linspace(0, 1, 128))))
cmap_blue_green_1 = ListedColormap(newcolors, name='BluesGreen')

#Option2: Blue negative, Green Positive
levs=range(256)
cmap_blue_green_2 = cm.colors.LinearSegmentedColormap.from_list(name='BlueGreen',
                            colors =[(0, 0, 1),(1, 1., 1),(0, 1, 0)],N=len(levs)-1,)

# Option3: Red negative, Green Positive
levs=range(256)
cmap_red_green = cm.colors.LinearSegmentedColormap.from_list(name='RedGreen',
                            colors =[(1, 0, 0),(1, 1., 1),(0, 1, 0)],N=len(levs)-1,)

# Option4: Back negative, Blue Positive
levs=range(256)
cmap_black_blue = cm.colors.LinearSegmentedColormap.from_list(name='BlackBlue',
                            colors =[(0, 0, 0),(1, 1., 1),(0, 0, 1)],N=len(levs)-1,)

#####

def drawPointsOnImage(points_, values_, img_size, voting='nn', inverse=False, s=0.5):
    """Draws a set of points (with individual values) on an image and returns the image.
    :param points: A [Nx2] array containing one point per row in the form: [x,y]
    :param values: A [Nx1] array containing othe values for each point
    :param img_size: Size of the output image (height, width)
    :param voting: Type of voting. Can be 'nn' (nearest-neighbor), or 'bilinear'.
    :param s: Standard deviation of the Gaussian kernel that is used to smooth the image (0 = no smoothing)
    """
    points = points_ if type(points_) is not torch.Tensor else points_.detach().cpu().numpy()
    values = values_ if type(values_) is not torch.Tensor else values_.detach().cpu().numpy()

    assert (points.shape[1] == 2)
    assert (values.shape[0] == points.shape[0])
    height, width = img_size
    assert (width > 0)
    assert (height > 0)
    assert (voting in ['nn', 'bilinear'])

    if inverse:
        img = np.ones(height * width)
    else:
        img = np.zeros(height * width)

    x, y = points[:, 0], points[:, 1]

    if voting == 'nn':
        # nearest-neighbor voting
        x, y = np.round(x).astype(int), np.round(y).astype(int)
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        # If there is not interpolation put work better than add
        # Nevethless, you can still change put by add and the method is the same
        np.put(img, (x + width * y).astype(np.int), values)

    else:
        # bilinear voting
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = x0 + 1, y0 + 1

        # compute the voting weights. Note: assign weight 0 if the point is out of the image
        wa = (x1 - x) * (y1 - y) * (x0 < width) * (y0 < height) * (x0 >= 0) * (y0 >= 0)
        wb = (x1 - x) * (y - y0) * (x0 < width) * (y1 < height) * (x0 >= 0) * (y1 >= 0)
        wc = (x - x0) * (y1 - y) * (x1 < width) * (y0 < height) * (x1 >= 0) * (y0 >= 0)
        wd = (x - x0) * (y - y0) * (x1 < width) * (y1 < height) * (x1 >= 0) * (y1 >= 0)

        x0 = np.clip(x0, 0, width - 1)
        x1 = np.clip(x1, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)
        y1 = np.clip(y1, 0, height - 1)

        np.add.at(img, x0 + width * y0, wa * values)
        np.add.at(img, x0 + width * y1, wb * values)
        np.add.at(img, x1 + width * y0, wc * values)
        np.add.at(img, x1 + width * y1, wd * values)

    img = np.reshape(img, (height, width))

    if s > 0:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=s, sigmaY=s)

    return img

def drawSparsePointsOnImage(points_, values_, img_size, voting='nn', s=0.5):
    """Draws a set of points (with individual values) on an image and returns the image.
    :param points: A [Nx2] array containing one point per row in the form: [x,y]
    :param values: A [Nx1] array containing othe values for each point
    :param img_size: Size of the output image (height, width)
    :param voting: Type of voting. Can be 'nn' (nearest-neighbor), or 'bilinear'.
    :param s: Standard deviation of the Gaussian kernel that is used to smooth the image (0 = no smoothing)
    """
    points = points_ if type(points_) is not torch.Tensor else points_.detach().cpu().numpy()
    values = values_ if type(values_) is not torch.Tensor else values_.detach().cpu().numpy()

    height, width = img_size

    img = np.zeros(height * width)

    x, y = points[:, 0], points[:, 1]

    if voting == 'nn':
        # nearest-neighbor voting
        x, y = np.round(x).astype(int), np.round(y).astype(int)
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        np.add.at(img, (x + width * y).astype(np.int), values)
        #np.put(img, (x + width * y).astype(np.int), values)

    else:
        # bilinear voting
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = x0 + 1, y0 + 1

        # compute the voting weights. Note: assign weight 0 if the point is out of the image
        wa = (x1 - x) * (y1 - y) * (x0 < width) * (y0 < height) * (x0 >= 0) * (y0 >= 0)
        wb = (x1 - x) * (y - y0) * (x0 < width) * (y1 < height) * (x0 >= 0) * (y1 >= 0)
        wc = (x - x0) * (y1 - y) * (x1 < width) * (y0 < height) * (x1 >= 0) * (y0 >= 0)
        wd = (x - x0) * (y - y0) * (x1 < width) * (y1 < height) * (x1 >= 0) * (y1 >= 0)

        x0 = np.clip(x0, 0, width - 1)
        x1 = np.clip(x1, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)
        y1 = np.clip(y1, 0, height - 1)

        np.add.at(img, x0 + width * y0, wa * values)
        np.add.at(img, x0 + width * y1, wb * values)
        np.add.at(img, x1 + width * y0, wc * values)
        np.add.at(img, x1 + width * y1, wd * values)

    img = np.reshape(img, (height, width))

    if s > 0:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=s, sigmaY=s)

    return img

def plotImages(imgs, legends, layout, title=None, figsize=(10, 8), dpi=150, interpolation='none', cmap='jet',
               showbar=False, shownorm=False, vmin=None, vmax=None, log_map=None):
    from matplotlib import colors
    assert (len(imgs) == len(legends))
    N = len(imgs)
    assert (N > 0)

    num_rows, num_cols = layout
    assert (N <= num_rows * num_cols)

    plt.figure(figsize=figsize, dpi=dpi)
    for img_index, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, img_index + 1)
        tmp = img if type(img) is not torch.Tensor else img.detach().cpu().numpy()
        cnorm = None
        if shownorm:
            vcenter = float('inf') if vmax != None and vmin != None else 0.0
            vmin_ = np.min(tmp) if vmin == None else vmin
            vmax_ = np.max(tmp) if vmax == None else vmax
            if vcenter == float('inf'):
                vcenter = vmin_ + (vmax_-vmin_)/2.0 # vmin < vcenter < vmax
            if vcenter == vmin_:
                vcenter += 1e-06# vmin < vcenter < vmax
            if log_map:
                cnorm = colors.LogNorm(vmin=vmin_ + 1e-05, vmax=vmax_)
            else:
                cnorm = colors.TwoSlopeNorm(vmin=vmin_, vcenter=vcenter, vmax=vmax_)
        im = plt.imshow(tmp, interpolation=interpolation, cmap=cmap, norm=cnorm)
        if showbar: plt.colorbar()
        if legends[img_index]:
            plt.title(legends[img_index])
    if title:
        plt.suptitle(title)
    return im  # then you can update the image with im.set_data(new_img)

def plotImagesGrid(imgs, layout, title=None, figsize=(10, 8), dpi=150, interpolation='none', cmap='seismic_r',
               showbar=False, shownorm=False, showaxis=True):
    from matplotlib import colors
    N = len(imgs)
    assert (N > 0)

    num_rows, num_cols = layout
    assert (N <= num_rows * num_cols)

    plt.figure(figsize=figsize, dpi=dpi)
    for img_index, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, img_index + 1)
        tmp = img if type(img) is not torch.Tensor else img.detach().cpu().numpy()
        cnorm=None
        if shownorm:
            cnorm = colors.TwoSlopeNorm(vmin=np.min(tmp), vcenter=0., vmax=np.max(tmp))
        plt.imshow(tmp, interpolation=interpolation, cmap=cmap, norm=cnorm)
        if showbar: plt.colorbar()
        if showaxis is not True:
            plt.axis('off')
    if title:
        plt.suptitle(title)


def plotImage(img, title=None, figsize=(10, 8), dpi=150, interpolation='none', cmap=None, showbar=False,
            shownorm=False, vmin=None, vmax=None, log_map=None):
    """ PlotImage: plot a single image
    cmap='seismic_r' it is nice for events
    """
    from matplotlib import colors
    tmp = img if type(img) is not torch.Tensor else img.detach().cpu().numpy()
    plt.figure(figsize=figsize, dpi=dpi)
    cnorm=None
    if shownorm:
        vcenter = float('inf') if vmax != None and vmin != None else 0.0
        vmin = np.min(tmp) if vmin == None else vmin
        vmax = np.max(tmp) if vmax == None else vmax
        if vcenter == float('inf'):
            vcenter = vmin + (vmax-vmin)/2.0 # vmin < vcenter < vmax
        if vcenter == vmin:
            vcenter += 1e-06# vmin < vcenter < vmax
        if vmin < vcenter < vmax:
            if log_map:
                cnorm = colors.LogNorm(vmin=vmin + 1e-05, vmax=vmax)
            else:
                cnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            cnorm = None
    im = plt.imshow(tmp, interpolation=interpolation, cmap=cmap, norm=cnorm)
    if showbar: plt.colorbar()
    plt.suptitle(title)
    return im  # then you can update the image with im.set_data(new_img)


def plotImage3D(img, title=None, figsize=(10, 8), dpi=150, interpolation='none', cmap='jet', showbar=False):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import colors
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    
    tmp = img if type(img) is not torch.Tensor else img.detach().cpu().numpy()

    fig = plt.figure()
    height, width = img.shape
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    x, y = np.meshgrid(x, y)
    ax = fig.gca(projection='3d')
    cnorm = colors.TwoSlopeNorm(vmin=np.min(tmp), vcenter=0., vmax=np.max(tmp))
    surf = ax.plot_wireframe(y, x, tmp, rstride=20, cstride=20, norm=cnorm)
    # surf = ax.plot_surface(y, x, img,  rstride=8, cstride=8, cmap='seismic_r', norm=cnorm, linewidth=0, antialiased=False)
    if title:
        plt.suptitle(title)
    plt.show()
    return surf


def plotOpticalFlow(flow, img_size, title=None, figsize=(10, 8), dpi=150, interpolation='none', cmap='seismic_r', shownorm=False):
    ''' It maps the optical flow to an img (does not plot)
    flow [H x W x 2] H is height and W is width
    Deprecated : use plot_flow_map instead
    '''     
    tmp = flow if type(flow) is not torch.Tensor else  flow.detach().cpu().numpy()
    if len(tmp.shape) < 3:
        flow_map = tmp.reshape((img_size[0], img_size[1], 2))
    else:
        flow_map = tmp

    hsv = np.zeros((flow_map.shape[0], flow_map.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255.0
    mag, ang = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    from matplotlib import colors
    plt.figure(figsize=figsize, dpi=dpi)
    cnorm = None
    if shownorm:
        cnorm = colors.TwoSlopeNorm(vmin=np.min(rgb), vcenter=0., vmax=np.max(rgb))
    im = plt.imshow(rgb, interpolation=interpolation, cmap=cmap, norm=cnorm)
    plt.suptitle(title)
    return im


def plotInertialValues(time, data, figsize=(10, 8), tick_distance=(10, 5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time[:, 1], data[:, 0], color='blue')
    ax.plot(time[:, 1], data[:, 1], color='green')
    ax.plot(time[:, 1], data[:, 2], color='red')

    # Major ticks every 20, minor ticks every 5
    min_value_x = min(time[:, 1])
    max_value_x = max(time[:, 1])
    min_value_y = min(min(data[:, 0]), min(data[:, 1]), min(data[:, 2]))
    max_value_y = max(max(data[:, 0]), max(data[:, 1]), max(data[:, 2]))

    major_ticks_x = np.arange(min_value_x, max_value_x, tick_distance[0])
    minor_ticks_x = np.arange(min_value_x, max_value_x, tick_distance[1])
    major_ticks_y = np.arange(min_value_y, max_value_y, tick_distance[0])
    minor_ticks_y = np.arange(min_value_y, max_value_y, tick_distance[1])

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.grid()

def plot_image_with_arrow_flow(img, flow, angle, title=None, figsize=(10,5), dpi=150, scale=10.0, interpolation='none', cmap='seismic_r'):
    from matplotlib import colors

    # Infos for the arrow
    height, width = img.shape
    arrow_base = (int(width/2.0), int(height/2.0))

    # Arrow tip
    direction = np.array([math.sin(angle), math.cos(angle)]) # y and x coord in this order
    arrow_length = scale * np.linalg.norm(flow)
    arrow_tip = arrow_length * direction

    plt.figure(figsize=figsize, dpi= dpi)
    cnorm = colors.TwoSlopeNorm(vmin=np.min(img), vcenter=0., vmax=np.max(img))
    im_plt = plt.imshow(img, interpolation=interpolation, cmap=cmap, norm=cnorm)
    im_arrow = plt.arrow (arrow_base[0], arrow_base[1], arrow_tip[0], arrow_tip[1], alpha=0.5, width=scale/3,)
    plt.colorbar()
    plt.suptitle(title)
    return im_plt, im_arrow # then you can update the image with im.set_data(new_img)

def plot_flow_map(flow_map, img_size, title=None, figsize=(10,5), dpi=150, scale=10.0, interpolation='none', cmap='seismic_r'):
    ''' It maps the optical flow to an img (does not plot)
    flow_map [N x 2] N is H * W points
    '''     
    tmp = flow_map if type(flow_map) is not torch.Tensor else  flow_map.detach().cpu().numpy()
    assert(img_size[0]*img_size[1] == tmp.shape[0])
    angle = np.array(np.arctan2(tmp[:,0], tmp[:,1]), dtype=np.float32)
    angle = np.array([degto360(rad2deg(a)) for a in angle], dtype=np.float32) # angle in deg 0 to 360
    #angle = np.abs(angle) * 180.0/math.pi # angle in deg 0 to 360
    magnitude = np.array(np.linalg.norm(tmp, axis=1), dtype=np.float32)

    # scale the magnitue [0-1]
    #magnitude = (magnitude-np.min(magnitude))/(np.max(magnitude) - np.min(magnitude))
    magnitude = magnitude/np.max(magnitude)
    
    # values
    value = np.ones(tmp.shape[0], dtype=np.float32)

    hsv = np.array([angle, magnitude, value]).transpose()
    hsv = hsv.reshape((img_size[0], img_size[1], 3))

    #from matplotlib.colors import hsv_to_rgb
    #rgb = hsv_to_rgb(hsv)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 
    return img

def optical_flow_opencv(frame1, frame2):
    hsv = np.zeros((frame1.shape[0], frame2.shape[1], 3), dtype=np.uint8)
    hsv[...,1] = 255
    flow_map = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow_map[...,0], flow_map[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return flow_map.reshape((flow_map.shape[0] * flow_map.shape[1], 2)), rgb

def put_flow_arrows_on_image(image, coord, flow, color=(0, 255, 0), mask=None, skip_amount=5):
    # Convert to numpy in case it comes in Tensor
    img = image if type(image) is not torch.Tensor else image.detach().cpu().numpy()
    points = coord if type(coord) is not torch.Tensor else coord.detach().cpu().numpy()
    of = flow if type(flow) is not torch.Tensor else flow.detach().cpu().numpy()
    #of = of.reshape(of.shape[0], of.shape[1])

    # Img to color in case it comes in grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # This is the same as the opencv above.
        #img = np.stack((img,)*3, axis=2)

    # Arrow tip
    angle = np.array(np.arctan2(of[:,0], of[:,1]), dtype=np.float32)
    magnitude = np.array(np.linalg.norm(of, axis=1), dtype=np.float32)
    direction = np.array([np.sin(angle), np.cos(angle)]).transpose() # y and x coord in this order
    arrow_tip =  direction
    arrow_tip[:,0] = arrow_tip[:,0] * magnitude
    arrow_tip[:,1] = arrow_tip[:,1] * magnitude
    arrow_tip = arrow_tip.reshape(points.shape)
    points_end = points + arrow_tip
    points_end = points_end.astype(np.int64)

    darrows = list(zip(points, points_end))

    #Draw all the nonzero values
    for i in range(0, len(darrows), skip_amount):
        if mask != None:
            if mask[i] == True:
                cv2.arrowedLine(img, tuple(darrows[i][0]), tuple(darrows[i][1]), color=color, thickness=1, tipLength=0.3)
        else:
            cv2.arrowedLine(img, tuple(darrows[i][0]), tuple(darrows[i][1]), color=color, thickness=1, tipLength=0.3)

    return img

def put_flow_on_image(image, coord, flow):
    # Convert to numpy in case it comes in Tensor
    img = image if type(image) is not torch.Tensor else image.detach().cpu().numpy()
    points = coord if type(coord) is not torch.Tensor else coord.detach().cpu().numpy()
    of = flow if type(flow) is not torch.Tensor else flow.detach().cpu().numpy()
    of = of.reshape(of.shape[0], of.shape[1])
    #points = points.reshape(180*240, 2)

    # Img to color in case it comes in grayscale
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=2)

    angle = np.array(np.arctan2(of[:,0], of[:,1]), dtype=np.float32)
    angle = np.array([degto360(rad2deg(a)) for a in angle], dtype=np.float32) # angle in deg 0 to 360
    magnitude = np.array(np.linalg.norm(of, axis=1), dtype=np.float32)

    # scale the magnitue [0-1]
    value = np.ones(of.shape[0], dtype=np.float32)
    magnitude = (magnitude-np.min(magnitude))/(np.max(magnitude) - np.min(magnitude))
    hsv = np.array([angle, magnitude, value]).transpose()
    hsv = np.expand_dims(hsv, axis=0)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    for i, p in enumerate(points):
        cv2.circle(img, tuple(p), 1, color=(int(rgb[0][i][0]), int(rgb[0][i][1]), int(rgb[0][i][2])))
    return img

def put_events_on_image(image, coord, polarities):
    # Convert to numpy in case it comes in Tensor
    img = image if type(image) is not torch.Tensor else image.detach().cpu().numpy()
    events = coord if type(coord) is not torch.Tensor else coord.detach().cpu().numpy()
    pol = polarities if type(polarities) is not torch.Tensor else polarities.detach().cpu().numpy()

    # Img to color in case it comes in grayscale
    if len(img.shape) == 2:
        img = img[:, : , None] * np.ones(3) #[H x W x 3 ]

    # Draw all the nonzero values
    for i, e in enumerate(events):
        if pol[i] > 0:
            cv2.circle(img, tuple(e), 1, color=(0, 255, 0)) # Green
        else:
            cv2.circle(img, tuple(e), 1, color=(0, 0, 255)) # Blue
    return img

def plot_trajectory(poses_df, estimated_df, keyframes=None, title=None, figsize=(5, 5), dpi=150, view='xy'):

    # Get the KFs position
    if keyframes != None:
        poses_kfs = []
        for kf in keyframes:
            poses_kfs.append(kf.get_pose(data_type='numpy')[:3, 3])
        poses_kfs = np.array(poses_kfs)


    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    if view =='xy':
        ax.plot(poses_df['px'].values, poses_df['py'].values, color='blue')
        ax.plot(estimated_df['px'].values, estimated_df['py'].values, color='red')
        if keyframes != None:
            ax.plot(poses_kfs[:,0], poses_kfs[:,1], 'g^')
    elif view =='xz':
        ax.plot(poses_df['px'].values, poses_df['pz'].values, color='blue')
        ax.plot(estimated_df['px'].values, estimated_df['pz'].values, color='red')
        if keyframes != None:
            ax.plot(poses_kfs[:,0], poses_kfs[:,2], 'g^')
        
    if keyframes != None:
        for i in range(poses_kfs.shape[0]):
            if view == 'xy':
                ax.text(poses_kfs[i,0], poses_kfs[i,1], str(i))
            elif view =='xz':
                ax.text(poses_kfs[i,0], poses_kfs[i,2], str(i))

    plt.grid()
    return

def plot_trajectory_with_dict(poses_df, estimated_df, traj_dict_kfs, title=None, figsize=(5, 5), dpi=150, view='xy', transform='matrix', dict_inv=False):

    # Sanity check
    assert(transform in ['matrix', 'angle_axis_translation', 'translation_angle_axis', 'quaternion_translation', 'translation_quaternion'] )

    # Get the KFs position
    poses_kfs = []
    for idx in traj_dict_kfs:
        if transform == 'matrix':
            if dict_inv:
                poses_kfs.append(np.linalg.inv(traj_dict_kfs[idx])[1][:3,3]) #take the translation
            else:
                poses_kfs.append(traj_dict_kfs[idx][1][:3,3]) #take the translation
        elif transform == 'angle_axis_translation':
            if dict_inv:
                poses_kfs.append(inv_angle_axis_translation(traj_dict_kfs[idx])[1][3:6]) #take the translation
            else:
                poses_kfs.append(traj_dict_kfs[idx][1][3:6]) #take the translation
        elif transform == 'translation_angle_axis':
            if dict_inv:
                poses_kfs.append(inv_translation_angle_axis(traj_dict_kfs[idx])[1][:3]) #take the translation
            else:
                poses_kfs.append(traj_dict_kfs[idx][1][:3]) #take the translation
        elif transform == 'quaternion_translation':
            if dict_inv:
                poses_kfs.append(inv_quaternion_translation(traj_dict_kfs[idx])[1][4:7]) #take the translation
            else:
                poses_kfs.append(traj_dict_kfs[idx][1][4:7]) #take the translation
        elif transform == 'translation_quaternion':
            if dict_inv:
                poses_kfs.append(inv_translation_quaternion(traj_dict_kfs[idx])[1][:3]) #take the translation
            else:
                poses_kfs.append(traj_dict_kfs[idx][1][:3]) #take the translation

    poses_kfs = np.array(poses_kfs)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    if view =='xy':
        ax.plot(poses_df['px'].values, poses_df['py'].values, color='blue')
        ax.plot(estimated_df['px'].values, estimated_df['py'].values, color='red')
        if poses_kfs.size > 0:
            ax.plot(poses_kfs[:,0], poses_kfs[:,1], 'g^')
    elif view =='xz':
        ax.plot(poses_df['px'].values, poses_df['pz'].values, color='blue')
        ax.plot(estimated_df['px'].values, estimated_df['pz'].values, color='red')
        if poses_kfs.size > 0:
            ax.plot(poses_kfs[:,0], poses_kfs[:,2], 'g^')
    
    if poses_kfs.size > 0:
        for i in range(poses_kfs.shape[0]):
            if view == 'xy':
                ax.text(poses_kfs[i,0], poses_kfs[i,1], str(i))
            elif view =='xz':
                ax.text(poses_kfs[i,0], poses_kfs[i,2], str(i))

    plt.xlabel("distance in meters")
    plt.ylabel("distance in meters")
    plt.grid()
    return

def plot_trajectory_euler(poses_df, estimated_df, kf_poses_dict, title=None, figsize=(5, 5), dpi=150, view='sxyz'):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gt_time = poses_df['time'].values
    gt_euler = np.array([tf.euler_from_quaternion(q) for q in poses_df[['qx', 'qy', 'qz', 'qw']].values])
    es_time = estimated_df['time'].values
    es_euler = np.array([tf.euler_from_quaternion(q) for q in estimated_df[['qx', 'qy', 'qz', 'qw']].values])

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title("x-axis")
    ax1.plot(gt_time, rad2deg360(gt_euler[:, 0]), color='red', linestyle='solid', label="groundtruth")
    ax1.plot(es_time, rad2deg360(es_euler[:, 0]), color='red', linestyle='dashed', label="estimate")
    ax1.legend()
    ax1.grid()
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("y-axis")
    ax2.plot(gt_time, rad2deg360(gt_euler[:, 1]), color='green', linestyle='solid', label="groundtruth")
    ax2.plot(es_time, rad2deg360(es_euler[:, 1]), color='green', linestyle='dashed', label="estimate")
    ax2.legend()
    ax2.grid()
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title("z-axis")
    ax3.plot(gt_time, rad2deg360(gt_euler[:, 2]), color='blue', linestyle="solid", label="groundtruth")
    ax3.plot(es_time, rad2deg360(es_euler[:, 2]), color='blue', linestyle='dashed', label="estimate")
    ax3.legend()
    ax3.grid()

    return


def show_pic(p):
    ''' use esc to see the results'''
    print(type(p))
    cv2.imshow('Color image', p)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27: break
    cv2.destroyAllWindows()
    return

def drawlines(image,lines, skip_amount = 10):
    ''' img - image on which we draw the epilines
    lines - corresponding epilines '''
    img = image if type(image) is not torch.Tensor else image.detach().cpu().numpy()

    r,c = img.shape
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in range(0, len(lines), skip_amount):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -lines[i, 2]/lines[i, 1] ])
        x1,y1 = map(int, [c, -(lines[i, 2]+lines[i, 0]*c)/lines[i, 1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color,1)
    return img


def viz_map(cloud: torch.tensor, intensity: torch.tensor, trajectory: list, last_kf_cloud: torch.tensor, save=False, nb=None, radius=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    last_pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.detach().cpu().numpy())
    if intensity != None:
        if intensity.dim() < 2:
            intensity = intensity.unsqueeze(-1)
        intensity = intensity.expand(intensity.shape[0], 3)
        pcd.colors = o3d.utility.Vector3dVector((intensity.expand(intensity.shape[0], 3)).detach().cpu().numpy())
    inliers = None
    if nb != None and radius != None:
        _, inliers = pcd.remove_radius_outlier(nb_points=nb, radius=radius)
        pcd  = pcd.select_by_index(inliers)
    if last_kf_cloud != None:
        last_pcd.points = o3d.utility.Vector3dVector(last_kf_cloud.detach().cpu().numpy())
        last_pcd.paint_uniform_color([1, 0, 0])
    if trajectory != None and len(trajectory) > 0:
        from pytransform3d import visualizer as pv
        fig = pv.figure()
        trajectory = np.array(trajectory)
        viz_traj = pv.Trajectory(trajectory, n_frames=trajectory.shape[0], s=0.1, c=[1.0, 0, 0])
        viz_traj.add_artist(fig)
        fig.add_geometry(pcd)
        fig.add_geometry(last_pcd)
        fig.plot_transform(A2B=np.eye(4), s=0.2)
        cam2world = trajectory[-1]
        # default parameters of a camera in Blender
        sensor_size = np.array([0.046, 0.034])
        intrinsic_matrix = np.array([
            [0.05, 0, sensor_size[0] / 2.0],
            [0, 0.05, sensor_size[1] / 2.0],
            [0, 0, 1]
        ])
        virtual_image_distance = 0.3
        fig.plot_camera(
            cam2world=cam2world, M=intrinsic_matrix, sensor_size=sensor_size,
            virtual_image_distance=virtual_image_distance)
        #vc = fig.visualizer.get_view_control()
        if save:
            return fig
        else:
            fig.show()
    else:
        o3d.visualization.draw_geometries([pcd])


def display_inlier_outlier(cloud: torch.tensor, ind: list):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.detach().cpu().numpy())

    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def computation_time(time_data: list, legend: list):

    fig, ax = plt.subplots()
    bplot = ax.boxplot(time_data,
                         vert=True,   # vertical box aligmnent
                         patch_artist=True)   # fill with color
    
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(time_data))], )
    ax.set_xlabel('')
    ax.set_ylabel('Computation time in [ms]')
    plt.setp(ax, xticks=[y+1 for y in range(len(time_data))],
         xticklabels=legend)
    plt.show()