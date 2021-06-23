import os
import argparse
import cv2
import math
import numpy as np
import pandas as pd
import tqdm
from pocolog_pybind import *
from pathlib import Path
from matplotlib import pyplot as plt

#Utils
from utils.functions import ensure_dir

# Viz
from viz.plotting import cmap_blue_green_1, cmap_blue_green_2, cmap_red_green, cmap_black_blue, plotImage, drawPointsOnImage

# ROS
import rosbag
import rospy
from sensor_msgs.msg import Image


def generate_and_save_event_frame(idx, events, start_t, end_t, delta_t, num_events, img_size, output_file, kalibr_format = False):
    event_frame = None
    next_id = -1
    if end_t is not None:
        if (end_t.to_seconds() - start_t.to_seconds()) >= delta_t:
            #print(np.array(events).shape)
            current_t = events[-1][0] #current time in seconds
            if current_t > end_t.to_seconds():
                #print("Generate image[{}] start_t: {} delta_t: {}".format(idx, start_t.to_seconds(), end_t.to_seconds() - start_t.to_seconds()))
                events_df = pd.DataFrame(events, columns=['time', 'x', 'y', 'p'])
                events_df.set_index('time')
                t = base.Time.from_microseconds((int)(start_t.to_microseconds()+(delta_t*1e06)))
                print(" t: ", t.to_seconds())
                e = events_df.loc[events_df["time"] <= t.to_microseconds()].values
                next_id = e.shape[0]
                event_frame = drawPointsOnImage(e[:, 1:3], e[:, 3], img_size, voting='bilinear')
        else:
            return False, next_id
    elif len(events) > num_events:
            #print("Generate image[{}] start_t: {} with {} events".format(idx, start_t.to_seconds(), len(events)))
            e = np.array(events[:num_events])
            next_id = num_events
            event_frame = drawPointsOnImage(e[:, 1:3], e[:, 3], img_size, voting='bilinear')
    else:
        return False, next_id

    if event_frame is not None:
        ts = start_t # base.Time.from_microseconds((int)(e[len(e)//2, 0]))
        print(" ts: ", ts.to_seconds())
        if type(output_file) is rosbag.bag.Bag:
            try:
                #ts = base.Time.from_seconds(events[len(events)//2][0])
                #print("ts: {}".format(ts.to_seconds()))
                stamp_ros = rospy.Time(nsecs=int(ts.to_microseconds()*1e03))
                rosimage = Image()
                rosimage.header.stamp = stamp_ros
                rosimage.height = event_frame.shape[0]
                rosimage.width = event_frame.shape[1]
                rosimage.step = rosimage.width  #only with mono8! (step = width * byteperpixel * numChannels)
                rosimage.encoding = "mono8"
                event_frame = (event_frame - np.min(event_frame))/(np.max(event_frame) - np.min(event_frame))
                event_frame = (event_frame * 255).astype(np.ubyte)
                rosimage.data = event_frame.tobytes()
                output_file.write('event_frame', rosimage, stamp_ros)
            except:
                print("error in saving the image into rosbag.")
            return True, next_id
        else:
            fig_events = plotImage(img=event_frame, showbar=False, cmap=cmap_black_blue, shownorm=True)

            if kalibr_format:
                frame_name = "%019d.png" % int(ts.to_microseconds()*1e03)
            else:
                frame_name = "frame_%010d.png" % idx
                with open(os.path.join(output_file, "timestamps.txt"), "a") as f_handle:
                    f_handle.write("%s %s\n" % (int(ts.to_microseconds()*1e03), frame_name))
            fig_events.figure.savefig(os.path.join(output_file, frame_name), bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig_events.figure)
        return True, next_id
    else:
        return False, next_id




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Events')
    parser.add_argument('pocolog_file', type=str, help='Path to pocolog.log file')
    parser.add_argument('--port_name', '-p', type=str, default='/camera_prophesee.events', help='Pocolog port for the event array')
    parser.add_argument('--freq_hz', '-fhz', type=int, default=0, help='Frequency for saving the reconstructed images from events')
    parser.add_argument('--num_events', '-nevents', type=int, default=10000, help='Number of events to reconstruct the images from events')
    parser.add_argument('--output', '-o', type=str, default='/tmp/event_frames', help='Output: path to folder to output images or filename for rosbag')
    parser.add_argument('--kalibr_format', action='store_true', help='use kalibr format to save the image name with timestamp')
    args = parser.parse_args()

    log_filepath = Path(args.pocolog_file)
    port_name = args.port_name

    output_file = Path(args.output)
    if output_file.suffix is '':
        print("** Generating events frames images and timestamps in folder: {}".format(output_file))
        ensure_dir(output_file)
        if args.kalibr_format == False:
            with open(os.path.join(output_file, "timestamps.txt"), "w") as f_handle:
                f_handle.write("#timestamp[nanosec] frame_image\n")
    else:
        assert output_file.parent.is_dir(), "Directory {} does not exist".format(output_file.parent)
        print("** Generating event frames in rosbag file: {}".format(str(output_file)))
        output_file = rosbag.Bag(str(output_file), 'w')

    if args.freq_hz > 0:
        delta_t = 1.0/args.freq_hz
        print("** Generating events frames based on freq {}Hz".format(args.freq_hz))
        num_events = None
    else:
        delta_t = None
        num_events = args.num_events
        print("** Generating events frames based on {} number of events".format(num_events))

    multi_file_index = pocolog.MultiFileIndex()
    multi_file_index.create_index([str(log_filepath)])
    streams = multi_file_index.get_all_streams()
    stream = streams[port_name]
    idx, next_id, start_t, height, width, events = 0, 0, None, None, None, []
    pbar = tqdm.tqdm(total=stream.get_size())
    for t in range(stream.get_size()):
        value = stream.get_sample(t)
        py_value = value.cast(recursive=True)
        value.destroy()
        height, width = py_value['height'], py_value['width']
        #print(py_value.keys(),len(py_value['events']))
        for x in py_value['events']: events.append(np.array([x.ts.to_microseconds(),x.x, x.y, (2*x.p)-1], dtype=np.float64))
        start_t = base.Time.from_microseconds(int(events[0][0])) if start_t is None else start_t
        end_t = base.Time.from_microseconds(int(events[-1][0])) if delta_t is not None else None
        success, next_id = generate_and_save_event_frame(idx, events, start_t, end_t, delta_t, num_events, (height, width), output_file, args.kalibr_format)
        #print("t: {} e: {} delta_t[{}] height {} height {} num_events {}".format(start_t.to_seconds(), end_t.to_seconds(), end_t.to_seconds()-start_t.to_seconds() , height, width, len(events)))
        if success:
            events = events[next_id:-1]
            start_t = None
            idx = idx + 1
            #input('Press ENTER to continue...')
        pbar.update(1)

    if type(output_file) is rosbag.bag.Bag:
        output_file.close()