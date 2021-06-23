#!/usr/bin/env python
import os
import argparse
from pathlib import Path
import numpy as np
import rosbag
import tqdm


def timestamp_str(ts):
    return str(ts.secs) + "." + str(ts.nsecs).zfill(9)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Rosbag pose to textfile""")
    parser.add_argument('bag_file', type=str, help='Path to ROS bag file')
    parser.add_argument('output_file', type=str, help='Path to write csv file')
    parser.add_argument('--topic_name', '-t', type=str, default='/evo/pose', help='Topic name for the pose in the rosbag')
    args = parser.parse_args()

    bag = rosbag.Bag(args.bag_file, 'r')
    csv_filepath = Path(args.output_file)

    assert csv_filepath.parent.is_dir(), "Directory {} does not exist".format(str(csv_filepath.parent))

    f_handle = open(str(csv_filepath), "w")

    pbar = tqdm.tqdm(total=bag.get_message_count(args.topic_name))
    for topic, msg, t in bag.read_messages(topics=[args.topic_name]):
        time = msg.header.stamp
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        #print ("t: %s new t: %s x: %f w: %f" %(t, timestamp_str(time), p[0], q[3]))
        f_handle.write("%s %f %f %f %f %f %f %f\n" % (timestamp_str(time), p[0], p[1], p[2], q[0], q[1], q[2], q[3]))
        pbar.update(1)

    bag.close()
