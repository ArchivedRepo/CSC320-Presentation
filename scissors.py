"""
Functions used for intelligent scissors.
Adapted from OpenCV tutorial:
https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
"""
import sys
import cv2 as cv
import numpy as np


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
    

def calculate_cost(image_name):
    """
    Calculate the cost of the edges. And construct a graph of all the pixels.
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
    laplacian_result = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)

    zero_crossing = cal_zero_crossing(laplacian_result)
    
    gradient_magnitude, partial_x, partial_y = cal_gradient_magnitude(src_gray)
    # cv.imshow("zero_crossing", gradient_magnitude)
    # cv.waitKey(0)


   
    return 0


if __name__ == "__main__":
    calculate_cost("trump.jpp")
    # abs_dst = cv.convertScaleAbs(dst)
    # cv.imshow(window_name, abs_dst)
    # cv.waitKey(0)