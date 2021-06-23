import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from pocolog_pybind import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert Pocolog Pose to Coma Separated Values')
    parser.add_argument('pocolog_file', type=str, help='Path to pocolog.log file')
    parser.add_argument('output_file', help='Path to write csv file')
    parser.add_argument('--port_name', '-p_name', type=str, default='/eds.pose_w_ef', help='Pocolog Rigid Body State port name')
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
        p = np.array(py_value['position']['data'])
        q = np.concatenate([py_value['orientation']['im'], np.array([py_value['orientation']['re']])])
        with open(csv_filepath, "a") as f_handle:
            f_handle.write("%s %s %s %s %s %s %s %s\n" % (t.to_seconds(), p[0], p[1], p[2], q[0], q[1], q[2], q[3]))

