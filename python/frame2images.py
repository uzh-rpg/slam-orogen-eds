import os
import argparse
import cv2 as cv
import math
import numpy as np
import pandas as pd
import tqdm
from pocolog_pybind import *
from pathlib import Path
from matplotlib import pyplot as plt

#Utils
from utils.functions import ensure_dir

# ROS
import rosbag
import rospy
from sensor_msgs.msg import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pocolog Frames to images (ROS or folder) ')
    parser.add_argument('pocolog_file', type=str, help='Path to pocolog.log file')
    parser.add_argument('--port_name', '-p', type=str, default='/camera_spinnaker.image_frame', help='Pocolog port for images')
    parser.add_argument('--output_file', '-o', type=str, default='/tmp/ros_images.bag', help='Output directory or filename for rosbag')
    parser.add_argument('--topic_name', '-t', type=str, default='/image_raw', help='ROS topic name')
    parser.add_argument('--kalibr_format', action='store_true', help='use kalibr format to save the image name with timestamp')
    parser.add_argument('--flip_x', action='store_true', help='Flip the image along x-axis')
    parser.add_argument('--flip_y', action='store_true', help='Flip the image along y-axis')

    args = parser.parse_args()

    log_filepath = Path(args.pocolog_file)
    output_file = Path(args.output_file)
    port_name = args.port_name

    if output_file.suffix is '':
        print("** Saving images and timestamps in folder {}".format(output_file))
        ensure_dir(output_file)
        if args.kalibr_format == False:
            with open(os.path.join(output_file, "timestamps.txt"), "w") as f_handle:
                f_handle.write("#timestamp[nanosec] frame_image\n")
        outbag = None
    else:
        assert output_file.parent.is_dir(), "Directory {} does not exist".format(output_file.parent)
        if os.path.exists(output_file):
            print('Detected existing rosbag: {}.'.format(output_file))
            print('Will overwrite the existing bag.')
        print("** Saving images in rosbag file: {}".format(str(output_file)))
        outbag = rosbag.Bag(str(output_file), 'w')

    multi_file_index = pocolog.MultiFileIndex()
    multi_file_index.create_index([str(log_filepath)])
    streams = multi_file_index.get_all_streams()
    stream = streams[port_name]
    idx, prev_ts = 0, None
    pbar = tqdm.tqdm(total=stream.get_size())
    for t in range(stream.get_size()):
        value = stream.get_sample(t)
        py_value = value.cast(recursive=True)
        value.destroy()
        ts = base.Time.from_microseconds(py_value['time']['microseconds'])
        height, width, channels = py_value['size']['height'], py_value['size']['width'], py_value['pixel_size']
        depth = py_value['data_depth']
        img = np.array(py_value['image'], dtype=np.uint8)
        img = img.reshape(height, width, channels)
        if args.flip_x: img = np.flip(img, axis=0)
        if args.flip_y: img = np.flip(img, axis=1)
        delta_t = ts.to_seconds() - prev_ts.to_seconds() if prev_ts is not None else 0
        #print("t: {} [{:.3f}] height {} width {} image {}".format(ts.to_seconds(), delta_t, height, width, img.size))
        if outbag is not None:
            try:
                stamp_ros = rospy.Time(nsecs=int(ts.to_microseconds()*1e03))
                rosimage = Image()
                rosimage.header.stamp = stamp_ros
                rosimage.height = height
                rosimage.width = width
                rosimage.step = width * depth * channels
                rosimage.encoding = "bgr8"
                #rosimage.encoding = "mono8"
                rosimage.data = img.tobytes()
                outbag.write(args.topic_name, rosimage, stamp_ros)
            except:
                print("error in reading file ", output_file)
        else:
            if args.kalibr_format == False:
                frame_name = "frame_%010d.png" % idx
                with open(os.path.join(output_file, "timestamps.txt"), "a") as f_handle:
                    f_handle.write("%s %s\n" % (int(ts.to_microseconds()*1e03), frame_name))
            else:
                frame_name = "%019d.png" % int(ts.to_microseconds()*1e03)
            cv.imwrite(os.path.join(str(output_file), frame_name), img[:, :, ::-1])
        #input('Press ENTER to continue...')
        prev_ts = ts
        idx = idx + 1
        pbar.update(1)

    if outbag is not None:
        outbag.close()