import argparse
from pathlib import Path

import numpy as np
import cv2 as cv
from tqdm import tqdm
from pocolog_pybind import *
from matplotlib import pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get the number of points used for the event-to-image alignment')
    parser.add_argument('pocolog_file', type=str, help='Path to pocolog.log file')
    parser.add_argument('output_file', help='Path to write csv file')
    parser.add_argument('--port_name', '-p_name', type=str, default='/eds.eds_info', help='Output Port name')
    args = parser.parse_args()

    log_filepath = Path(args.pocolog_file)
    csv_filepath = Path(args.output_file)
    portname = args.port_name

    assert csv_filepath.parent.is_dir(), "Directory {} does not exist".format(str(csv_filepath.parent))

    multi_file_index = pocolog.MultiFileIndex()
    multi_file_index.create_index([str(log_filepath)])
    streams = multi_file_index.get_all_streams()
    stream = streams[portname]
    for t in range(stream.get_size()):
        value = stream.get_sample(t)
        py_value = value.cast(recursive=True)
        value.destroy()
        t = base.Time.from_microseconds(py_value['time']['microseconds'])
        p = py_value['kf_num_points']
        with open(csv_filepath, "a") as f_handle:
            f_handle.write("%s %s\n" % (t.to_seconds(), p))
