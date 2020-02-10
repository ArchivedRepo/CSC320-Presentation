"""
File that contains all the functions to construct the cost dictionary G
G[(col, row)] = {key: neighbour pixel, value: cost relate to this pixel}
"""
import sys
import cv2 as cv
import numpy as np
import heapq
import math
from typing import Dict
import matplotlib.pyplot as plt
import time

WZ, WD, WG = 0.43, 0.43, 0.14

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


def construct_dict(image_name):
    """
    Construct and return the cost dictionary G as specified in the description
    of the file.
    """
    # Declare the variables we are going to use
    ddepth = cv.CV_16S
    kernel_size = 3
    image_name = image_name
    src = cv.imread(cv.samples.findFile(image_name), cv.IMREAD_COLOR) # Load an image
    if src is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default lena.jpg]')
        return -1
    # src = cv.GaussianBlur(src, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    height, width = src_gray.shape
    laplacian_result = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)

    zero_crossing = cal_zero_crossing(laplacian_result)
    print("Finish Computing Zero Crossing");
    gradient_magnitude, partial_x, partial_y = cal_gradient_magnitude(src_gray)
    print("Finish calculating gradient cost")

    def in_range(_i, _j):
        if _i < height and _i >= 0 and _j < width and _j >= 0:
            return True
        return False
    G = {}
    print("")
    for row in range(height):
        print(f"Finsih line {row}", end="\r")
        for col in range(width):
            neighbors = []
            for row_other in range(row-1,row+2):
                for col_other in range(col-1, col+2):
                    if in_range(row_other, col_other):
                        neighbors.append((row_other, col_other))
            dist = {}
            for row_other, col_other in neighbors:
                # let p = (row, col), q = (row_other, col_other)
                # Calculate l(p,q)
                f_z = zero_crossing[(row_other, col_other)]
                f_g = gradient_magnitude[(row_other, col_other)]

                def dot(_x, _y):
                    return _x[0] * _y[0] + _x[1] * _y[1]
                def norm(_x):
                    return math.sqrt(_x[0]**2+_x[1]**2)

                q_p = (row_other-row, col_other-col)
                d_p_prime = (partial_y[row, col], -partial_x[row, col])
                if norm(d_p_prime) != 0:
                    d_p_prime = (d_p_prime[0]/norm(d_p_prime),d_p_prime[1]/norm(d_p_prime))
                l_p_q = q_p if dot(d_p_prime, q_p) >=0 else (-q_p[0], -q_p[1])
                if (norm(l_p_q) != 0):
                    l_p_q = (l_p_q[0]/norm(l_p_q), l_p_q[1]/norm(l_p_q))

                d_p = dot(d_p_prime, l_p_q)
                d_q_prime = (partial_y[row_other, col_other], -partial_x[row_other, col_other])
                if norm(d_q_prime) != 0:
                    d_q_prime = (d_q_prime[0]/norm(d_q_prime), d_q_prime[1]/norm(d_q_prime))
                d_q = dot(d_q_prime, l_p_q)
                f_d = 1/math.pi * (math.acos(d_p) + math.acos(d_q))
                dist[(row_other, col_other)] = WZ * f_z + WD * f_d + WG * f_g
            G[(row, col)] = dist
    return G, height, width

                
                    
