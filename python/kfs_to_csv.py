import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from pocolog_pybind import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert Pocolog Pose to Coma Separated Values')
    parser.add_argument('pocolog_file', type=str, help='Path to pocolog.log file')
    parser.add_argument('output_file', help='Path to write csv file')
    parser.add_argument('--port_name', '-p_name', type=str, default='/eds.pose_w_kfs', help='Pocolog Rigid Body State port name')
    parser.add_argument('--kf_order', '-kf_id', type=int, default=-1, help='KF (order) in the slising window')
    args = parser.parse_args()

    log_filepath = Path(args.pocolog_file)
    csv_filepath = Path(args.output_file)
    portname = args.port_name
    kf_id = args.kf_order

    assert csv_filepath.parent.is_dir(), "Directory {} does not exist".format(str(csv_filepath.parent))

    multi_file_index = pocolog.MultiFileIndex()
    multi_file_index.create_index([str(log_filepath)])
    streams = multi_file_index.get_all_streams()
    stream = streams[portname]
    for t in range(stream.get_size()):
        value = stream.get_sample(t)
        py_value = value.cast(recursive=True)
        value.destroy()
        rbs = py_value['kfs'][kf_id] # take last KF
        with open(csv_filepath, "a") as f_handle:
            f_handle.write("%s %s %s %s %s %s %s %s\n" % (rbs.time.to_seconds(), rbs.position[0], rbs.position[1], rbs.position[2],
                                                        rbs.orientation.x(),rbs.orientation.y(), rbs.orientation.z(), rbs.orientation.w()))

