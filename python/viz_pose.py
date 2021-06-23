import argparse
import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from pocolog_pybind import *
from pathlib import Path
from matplotlib import pyplot as plt

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Pose Plots')
    parser.add_argument('pocolog_file_1', type=str, help='Path to pocolog.log file')
    parser.add_argument('pocolog_file_2', type=str, help='Path to pocolog.log file')
    parser.add_argument('--pose_name_1', '-p_name_1', type=str, default='/eds.pose_w_ef', help='Pocolog Rigid Body State port name')
    parser.add_argument('--pose_name_2', '-p_name_2', type=str, default='/eds.pose_w_ef', help='Pocolog Rigid Body State port name')
    args = parser.parse_args()

    log_filepath = Path(args.pocolog_file_1)
    poses_name = args.pose_name_1

    trajectory = []
    multi_file_index = pocolog.MultiFileIndex()
    multi_file_index.create_index([str(log_filepath)])
    streams = multi_file_index.get_all_streams()
    stream = streams[poses_name]
    for t in range(stream.get_size()):
        value = stream.get_sample(t)
        py_value = value.cast(recursive=True)
        value.destroy()
        t = base.Time.from_microseconds(py_value['time']['microseconds'])
        p = np.array(py_value['position']['data'])
        q = np.concatenate([py_value['orientation']['im'], np.array([py_value['orientation']['re']])])
        trajectory.append(np.array([t.to_seconds(), p[0], p[1], p[2], q[0], q[1], q[2], q[3]]))

    poses_df = pd.DataFrame(trajectory, columns=['time', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])

    print(poses_df)

    poses_name = args.pose_name_2
    log_filepath = Path(args.pocolog_file_1)

    trajectory = []
    multi_file_index = pocolog.MultiFileIndex()
    multi_file_index.create_index([str(log_filepath)])
    streams = multi_file_index.get_all_streams()
    stream = streams[poses_name]
    for t in range(stream.get_size()):
        value = stream.get_sample(t)
        py_value = value.cast(recursive=True)
        value.destroy()
        t = base.Time.from_microseconds(py_value['time']['microseconds'])
        p = np.array(py_value['position']['data'])
        q = np.concatenate([py_value['orientation']['im'], np.array([py_value['orientation']['re']])])
        trajectory.append(np.array([t.to_seconds(), p[0], p[1], p[2], q[0], q[1], q[2], q[3]]))

    estimated_df = pd.DataFrame(trajectory, columns=['time', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])


    print(estimated_df)
    plot_trajectory(poses_df, estimated_df)
    plt.show()