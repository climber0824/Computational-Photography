# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 22:47:59 2021

@author: ABC
"""

import numpy as np
from functools import partial
import cv2 as cv

def local_tone_mapping(HDRIMG, Filter, window_size, sigma_s, sigma_r):
    """ Perform Local tone mapping on HDRIMG
            Note:
                1.Please remember clip the range of intensity to [0, 1] and convert type of LDRIMG to "uint8" with range [0, 255] for display. You can use the following code snippet.
                  >> LDRIMG = np.round(LDRIMG*255).astype("uint8")
                2.Make sure the LDRIMG's range is in 0-255(uint8). If the value is larger than upperbound, modify it to upperbound. If the value is smaller than lowerbound, modify it to lowerbound.
            Args:
                HDRIMG (np.ndarray): The input image to process
                Filter (function): 'Filter' is a function that is used for filter operation to get base layer. It can be gaussian or bilateral.
                                   It's input is log of the intensity and filter's parameters. And the output is the base layer.
                window size(diameter) (int): default 35
                sigma_s (int): default 100
                sigma_r (float): default 0.8
            Returns:
                LDRIMG (np.ndarray): The processed corresponding low dynamic range image of HDRIMG
            Todo:
                - implement local tone mapping here
    """
    gamma = 2.2
    scale = 3
    LDRIMG = np.zeros(np.shape(HDRIMG))
    R = HDRIMG[:,:,0]
    G = HDRIMG[:,:,1]
    B = HDRIMG[:,:,2]
    I = (R + G + B)/3
    C_R = R / I
    C_G = G / I
    C_B = B / I
    L = np.log2(I)
    L_B = Filter(HDRIMG, window_size, sigma_s, sigma_r)
    L_D = L - L_B
    L_min = np.min(L_B)
    L_max = np.max(L_B)
    L_B_p = (L_B - L_max) * scale / (L_max - L_min)
    I_p = 2 ** (L_B_p + L_D)
    LDRIMG[:,:,0] = (C_R*I_p)**(1.0/gamma)
    LDRIMG[:,:,1] = (C_G*I_p)**(1.0/gamma)
    LDRIMG[:,:,2] = (C_B*I_p)**(1.0/gamma)
    # print(np.max(LDRIMG))
    LDRIMG[LDRIMG > 1] = 1
    LDRIMG[LDRIMG < 0] = 0
    LDRIMG = np.round(LDRIMG * 255).astype('uint8')
    return LDRIMG


def bilateral(HDRIMG,window_size,sigma_s,sigma_r):
    """ Perform bilateral filter
            Notes:
                Please use "symmetric padding" for image padding
            Args:
                HDRIMG: HDR image
                window size(diameter) (int): default 39
                sigma_s (int): default 100
                sigma_r (float): default 0.8
            Returns:
                LB (np.ndarray): The base layer
            Todo:
                - implement bilateral filter for local tone mapping
    """
    half_window_size = (window_size-1)//2
    rows, cols, __ = np.shape(HDRIMG)
    R = HDRIMG[:,:,0]
    G = HDRIMG[:,:,1]
    B = HDRIMG[:,:,2]
    I = (R + G + B)/3
    L = np.log2(I)
    L_pad = np.pad(L, half_window_size, 'symmetric')
    
    gaussian_windows = np.zeros((window_size,window_size))
    for i in range(window_size):
        for j in range(window_size):
            gaussian_windows[i,j] = np.exp(-((i-half_window_size)**2+(j-half_window_size)**2) / (2.0*sigma_s**2))

    LB = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            #windows = np.exp(-(L_pad_t[i:i+window_size, j:j+window_size] - L_pad_t[i+half_window_size, j+half_window_size])**2) / (2.0*sigma_r**2) * gaussian_windows
            windows = np.exp(-(L_pad[i:i+window_size, j:j+window_size] - L_pad[i+half_window_size, j+half_window_size])**2) / (2.0*sigma_r**2) * gaussian_windows
            LB[i, j] = np.sum(np.multiply(L_pad[i:i+window_size, j:j+window_size], windows)) / np.sum(windows)

    return LB

HDRIMG = np.load('../ref/p4_imgpatch.npy')
img_joint_bilateral = local_tone_mapping(HDRIMG,Filter=bilateral,window_size=35,sigma_s=100,sigma_r=0.8)
golden = cv.imread('../ref/p4_ltm_patch.png')
psnr = cv.PSNR(golden, img_joint_bilateral)
print(psnr)