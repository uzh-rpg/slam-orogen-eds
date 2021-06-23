import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm
from pocolog_pybind import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert Pocolog Pose to Coma Separated Values')
    parser.add_argument('pocolog_file', type=str, help='Path to pocolog.log file')
    parser.add_argument('--port_name', '-p_name', type=str, default='/eds.global_map', help='Output Port name')
    args = parser.parse_args()

    log_filepath = Path(args.pocolog_file)
    portname = args.port_name

    multi_file_index = pocolog.MultiFileIndex()
    multi_file_index.create_index([str(log_filepath)])
    streams = multi_file_index.get_all_streams()
    stream = streams[portname]
    for t in range(stream.get_size()):
        value = stream.get_sample(t)
        py_value = value.cast(recursive=True)
        value.destroy()
        points = np.array(py_value['points'])
        colors = np.array(py_value['colors'])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd])
        #input('Press ENTER to continue...')

