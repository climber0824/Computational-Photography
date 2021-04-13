# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:58:31 2021

@author: ABC
"""

import numpy as np
from functools import partial
import cv2 as cv



def localTM(src, imgFilter, scale=3):
    """Local tone mapping (section 1-3)
    Args:
        src (ndarray, float32): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float, optional): scaling factor (Defaults to 3)
    """
    
    result = np.zeros_like(src, dtype=np.uint8)
    gamma = 2.2
    R = src[:,:,0]
    G = src[:,:,1]
    B = src[:,:,2]
    I = (R + G + B)/3
    I_new = np.zeros_like(src, dtype=np.uint8)
    
    for i in range(3):
        I_new[:,:,i] = I
    
    print('I_new', I_new.shape)
    print('I', I.shape)
    C_R = R / I
    C_G = G / I
    C_B = B / I
    L = np.log2(I_new)
    L_B = imgFilter(src)
    print('L', L.shape)
    print('L_B', L_B.shape)
    L_D = L - L_B  
    L_min = np.min(L_B)
    L_max = np.max(L_B)
    L_B_p = (L_B - L_max) * scale / (L_max - L_min)
    print('L_B_p', L_B_p.shape)
    print('L_D', L_D.shape)
    I_p = 2 ** (L_B_p + L_D)
    result[:,:,0] = (C_R*I_p[:,:,0])**(1.0/gamma)
    result[:,:,1] = (C_G*I_p[:,:,1])**(1.0/gamma)
    result[:,:,2] = (C_B*I_p[:,:,2])**(1.0/gamma)
    #result[result > 1] = 1
    #result[result < 0] = 0
    result = np.round(result * 255).astype('uint8')
    
    return result

def gaussianFilter(src, N=35, sigma_s=100):
    """Gaussian filter (section 1-3)
    Args:
        src (ndarray): source image
        N (int, optional): window size of the filter (Defaults to 35)
            filter indices span [-N/2, N/2]
        sigma_s (float, optional): standard deviation of Gaussian filter (Defaults to 100)
    """
   
    assert N % 2
    dtype = np.float32
    result = np.zeros_like(src, dtype=dtype)
    
    """ pass the gaussian_test
    rows, cols, _ = np.shape(src)
    half_window_size = (N-1)//2
    L_pad = np.pad(src, half_window_size, 'symmetric')
    windows = np.zeros((N, N))
    
    print('L_pad', L_pad.shape)
    print('windows', windows.shape)
   
    for i in range(N):
        for j in range(N):
            windows[i,j] = np.exp(-float((i-half_window_size)**2+(j-half_window_size)**2) / (2.0*sigma_s**2))

    for i in range(rows):
        for j in range(cols):
            result[i, j] = np.sum(np.multiply(L_pad[i:i+N, j:j+N], windows))
            
    result = result / np.sum(windows)
    """
    R = src[:,:,0]
    G = src[:,:,1]
    B = src[:,:,2]
    I = (R + G + B)/3
    L = np.log2(I)
    rows, cols, __ = np.shape(src)
    half_window_size = (N-1)//2
    L_pad = np.pad(L, half_window_size, 'symmetric')
   # L_pad.reshape(src.shape[0], src.shape[1])
   # print(L_pad.shape)

    windows = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            windows[i,j] = np.exp(-float((i-half_window_size)**2+(j-half_window_size)**2) / (2.0*sigma_s**2))

    print('window', windows.shape)
    for i in range(rows):
        for j in range(cols):
            result[i, j] = np.sum(np.multiply(L_pad[i:i+N, j:j+N], windows))
            
            
    result = result / np.sum(windows)
 
    return result

radiance = cv.imread('../TestImg/vinesunset.hdr', -1)
golden = cv.imread('../ref/p3_ltm.png')
gauhw1 = partial(gaussianFilter, N=35, sigma_s=100)
test = localTM(radiance, gauhw1, scale=3)
psnr = cv.PSNR(golden, test)
print(psnr)