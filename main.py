"""
Functions used for intelligent scissors.
Adapted from OpenCV tutorial:
https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
"""
import sys
import cv2 as cv
import numpy as np
import heapq
import math
from typing import Dict
import matplotlib.pyplot as plt
import time
import pickle

from cost import construct_dict
from dijkstra import graph_search

if __name__ == "__main__":
    image_name = 'thanos.jpg'
    try:
        with open(f"{image_name}.picke", 'rb') as ptr:
            G = pickle.load(ptr)
    except FileNotFoundError:
        G, height, width = construct_dict(image_name)
        with open(f"{image_name}.picke", 'wb') as ptr:
            pickle.dump(G, ptr)

    ddepth = cv.CV_16S
    kernel_size = 3
    src = cv.imread(cv.samples.findFile(image_name), cv.IMREAD_COLOR) # Load an image
    if src is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default lena.jpg]')
        exit(-1)
    imshow_obj = None
    # src = cv.GaussianBlur(src, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    display_img = np.dstack([src_gray, src_gray, src_gray])
    start = None
    def button_pressed(event):
        global start
        print (int(event.ydata), int(event.xdata))
        if start is None:
            start = (int(event.ydata), int(event.xdata))
            
        else:
            end =  (int(event.ydata), int(event.xdata))
            print("begin searching")
            end_node = graph_search(G, start, end)
            print("searching Complete")
            cur_node = end_node
            while cur_node is not None:
                display_img[cur_node.row, cur_node.col, :] = [225, 225, 0]
                cur_node = cur_node.pred
            imshow_obj.set_data(display_img)
            plt.draw()
            start = None
            print("Finished")
    plt.connect('button_release_event', button_pressed)
    imshow_obj = plt.imshow(display_img)
    plt.autoscale(False);
    plt.show()
