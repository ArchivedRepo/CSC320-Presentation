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


def cal_zero_crossing(laplacian_result):
    """
    The function takes in an a 2d array store the laplancian of each pixel.
    Return an array of same dimension that store the zero crossing cost 
    """
    height, width = laplacian_result.shape
    result = np.ones(shape=laplacian_result.shape)
    def in_range(_i, _j):
        if _i < height and _i >= 0 and _j < width and _j >= 0:
            return True
        return False

    for i in range(height):
        for j in range(width):
            # Iterate Through the neighboring pixels
            cur_pixel = laplacian_result[i][j]
            if cur_pixel == 0:
                result[i, j] = 0
                continue

            for ver in range(-1, 2):
                for hor in range(-1, 2):
                    cur_x, cur_y = i + ver, j + hor
                    if in_range(cur_x, cur_y):
                        neigh_pixel = laplacian_result[cur_x][cur_y]
                        if neigh_pixel < 0 and cur_pixel > 0 and abs(cur_pixel) <= abs(neigh_pixel):
                            result[i, j] = 0
                        elif neigh_pixel > 0 and cur_pixel < 0 and abs(cur_pixel) <= abs(neigh_pixel):
                            result[i, j] = 0
    return result


def cal_gradient_magnitude(src):
    """
    The function takes in the source image and calculate the gradient Magnitude 
    cost. Return an array of same dimension as src, which stores the gradient 
    magnitude at each point.
    This need to be scalled based on location information.
    Also return x partial and y_partial
    """
    # Use sobel operatior to find derivatives
    sobelx = cv.Sobel(src,cv.CV_64F,1,0,ksize=5)  # partial x
    sobely = cv.Sobel(src,cv.CV_64F,0,1,ksize=5)  # partial y
    G_sqr = np.power(sobelx, 2) + np.power(sobely, 2)
    G = np.sqrt(G_sqr)
    max_value = np.max(G)
    return 1 - G / max_value, sobelx, sobely
    

def calculate_info(image_name):
    """
    Calculate the cost of the edges. 
    Return zero_crossing, gradient_magnitude, partial_x and partial_y
    """
    # Declare the variables we are going to use
    ddepth = cv.CV_16S
    kernel_size = 3
    image_name = 'trump.jpg'
    src = cv.imread(cv.samples.findFile(image_name), cv.IMREAD_COLOR) # Load an image
    if src is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default lena.jpg]')
        return -1
    src = cv.GaussianBlur(src, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    height, width = src_gray.shape
    laplacian_result = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)

    zero_crossing = cal_zero_crossing(laplacian_result)
    
    gradient_magnitude, partial_x, partial_y = cal_gradient_magnitude(src_gray)
    # gradient_magnitude = gradient_magnitude.flatten()
    # partial_x = partial_x.flatten()
    # partial_y = partial_y.flatten()

    all_nodes = {}
    def index(_i, _j):
        return _i * width + _j
    for i in range(height):
        for j in range(width):
            cur_index = index(i, j)
            divisor = math.sqrt(partial_x[i, j] ** 2 + partial_y[i, j] ** 2)
            new_node = Node(i,j,cur_index,zero_crossing[i,j],\
                gradient_magnitude[i,j], partial_x[i, j]/divisor, partial_y[i, j]/divisor)
            all_nodes[cur_index] = new_node
    return all_nodes, width, height


class Node:
    """
    zero_crossing: binary zero crossing cost
    Attribute:
    pred: points to the node on the shortest path to node
    cost: total cost from this node to root
    """
    def __init__(self, x, y, index, zero_crossing, g_magnitude, p_x, p_y):
        self.index = index
        self.x = x
        self.y = y
        self.zero_crossing = zero_crossing
        self.g_magnitude = g_magnitude
        # (p_x, p_y) must be unit vector
        self.p_x = p_x
        self.p_y = p_y 
        self.pred = None
        self.cost = float('inf')
        self.explored = False
    
    def cal_cost(self, other):
        def dot(_x, _y):
            return _x[0] * _y[0] + _x[1] * _y[1]
        q_p = (other.x - self.x, other.y - self.y)
        d_prime = (self.p_y, -self.p_x)
        l_p_q = q_p if dot(d_prime, q_p) >= 0 else (-q_p[0], -q_p[1])
        tmp = math.sqrt(l_p_q[0]**2 + l_p_q[1] ** 2)
        l_p_q = (l_p_q[0] / tmp, l_p_q[1] / tmp)
        d_p = dot(d_prime, l_p_q)

        d_q_prime = (other.p_y, -other.p_x)
        d_q = dot(l_p_q, d_q_prime)
        f_d = 1/math.pi * (math.acos(d_p) + math.acos(d_q))
        return 0.43 * self.zero_crossing + 0.43 * f_d + 0.14 * other.g_magnitude
        
    def __lt__(self, other):
        return self.cost < other.cost
    
    def __eq__(self, value):
        return self.index == value.index


def graph_search(start: Node, all_nodes: Dict[int, Node], width, height, end: int):
    def in_range(_i, _j):
        if _i < height and _i >= 0 and _j < width and _j >= 0:
            return True
        return False
    def index(_i, _j):
        return _i * width + _j
    queue = [start]
    start.cost = 0
    count = 0
    while len(queue) != 0:
        this_node = heapq.heappop(queue)
        this_node.explored = True
        if this_node.index == end:
            return this_node
        for ver in range(-1, 2):
            for hor in range(-1, 2):
                cur_x, cur_y = this_node.x + ver, this_node.y + hor
                cur_index = cur_x * width + cur_y
                if in_range(cur_x, cur_y):
                    cur_neighbor = all_nodes[cur_index]
                    if not cur_neighbor.explored:
                        g_tmp = this_node.cost + this_node.cal_cost(cur_neighbor)
                        if cur_neighbor in queue and g_tmp < cur_neighbor.cost:
                            queue.remove(cur_neighbor)
                            heapq.heapify(queue)
                        if cur_neighbor not in queue:
                            cur_neighbor.cost = g_tmp
                            cur_neighbor.pred = this_node
                            queue.append(cur_neighbor)
                            heapq.heappush(queue, cur_neighbor)


if __name__ == "__main__":
    

    ddepth = cv.CV_16S
    kernel_size = 3
    image_name = 'thanos.jpg'
    src = cv.imread(cv.samples.findFile(image_name), cv.IMREAD_COLOR) # Load an image
    if src is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default lena.jpg]')
    src = cv.GaussianBlur(src, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    all_nodes, width, height = calculate_info(image_name)
    display_img = np.dstack([src_gray, src_gray, src_gray])
    print("start search......")
    start_index = None
    def button_pressed(event):
        global start_index
        print (int(event.ydata), int(event.xdata))
        if start_index is None:
            start_index = int(event.ydata) * width + int(event.xdata)
            
        else:
            end_index = int(event.ydata) * width + int(event.xdata)
            print("begin searching")
            end_node = graph_search(all_nodes[start_index], all_nodes, width, height, end_index)
            print("searching Complete")
            cur_node = end_node
            while cur_node is not None:
                display_img[cur_node.x, cur_node.y, :] = [225, 225, 0]
                cur_node = cur_node.pred
            plt.close()
            plt.imshow(display_img)
            plt.show()
            print("Finished")
    plt.connect('button_release_event', button_pressed)
    plt.imshow(display_img)
    plt.show()
    # abs_dst = cv.convertScaleAbs(dst)
    # cv.imshow(window_name, abs_dst)
    # cv.waitKey(0)