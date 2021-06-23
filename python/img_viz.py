import argparse
from pathlib import Path

import numpy as np
import cv2 as cv
from tqdm import tqdm
from pocolog_pybind import *
from matplotlib import pyplot as plt

def plotImage(number, img, title=None, figsize=(10, 8), dpi=150, interpolation='none', cmap=None, showbar=False,
            shownorm=False, vmin=None, vmax=None, log_map=None):
    """ PlotImage: plot a single image
    cmap='seismic_r' it is nice for events
    """
    from matplotlib import colors
    tmp = img
    plt.figure(num=number, figsize=figsize, dpi=dpi, frameon=False)
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
    im = plt.imshow(tmp, interpolation=interpolation, cmap=cmap, norm=cnorm, aspect='auto')
    if showbar: plt.colorbar()
    plt.suptitle(title)
    return im  # then you can update the image with im.set_data(new_img)





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Images on Pocolog Frames')
    parser.add_argument('pocolog_file', type=str, help='Path to pocolog.log file')
    parser.add_argument('--port_name', '-p_name', type=str, default='/eds.inv_depth_frame', help='Output Port name')
    args = parser.parse_args()

    log_filepath = Path(args.pocolog_file)
    portname = args.port_name

    # plot show on
    plt.ion()

    multi_file_index = pocolog.MultiFileIndex()
    multi_file_index.create_index([str(log_filepath)])
    streams = multi_file_index.get_all_streams()
    stream = streams[portname]
    for t in range(stream.get_size()):
        value = stream.get_sample(t)
        py_value = value.cast(recursive=True)
        value.destroy()
        img = np.array(py_value['image'])
        width = py_value['size']['width']
        height = py_value['size']['height']
        channels = py_value['pixel_size']
        img = img.reshape(height, width, channels)
        plotImage(1, img)
        input('Press ENTER to continue...')

