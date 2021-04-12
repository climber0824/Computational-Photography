import unittest
import numpy as np
from functools import partial
import cv2 as cv
import math

def globalTM(src, scale=1.0):
    """Global tone mapping (section 1-2)
    Args:
        src (ndarray, float32): source radiance image
        scale (float, optional): scaling factor (Defaults to 1.0)
    """
    result = np.zeros_like(src, dtype=np.uint8)
    gamma = 2.2
    x_max = np.max(radiance, axis = 2)
    
    for width in range(src.shape[0]):
        for height in range(src.shape[1]):
            for channel in range(src.shape[2]):
                if radiance[width][height][channel] >= 1:
                    radiance[width][height][channel] = 1
                result[width][height][channel]=255 * ( 2 ** (scale*(np.log2(radiance[width][height][channel])-\
                     np.log2(x_max[width][height]) + np.log2(x_max[width][height]))))** (1/gamma)

    return result


def whiteBalance(src, y_range, x_range):
    """White balance based on Known to be White(KTBW) region
    Args:
        src (ndarray): source image
        y_range (tuple of 2): location range in y-dimension
        x_range (tuple of 2): location range in x-dimension
    """
    
    result = np.zeros_like(src)
    B_channel = src[:,:,0]
    G_channel = src[:,:,1] 
    R_channel = src[:,:,2]
    print(x_range[0],x_range[1], y_range[0],y_range[1])
    print(src.shape[2])
    B_ktbw = B_channel[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    G_ktbw = G_channel[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    R_ktbw = R_channel[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    
    print(B_ktbw)
    B_avg = np.mean(B_ktbw)
    G_avg = np.mean(G_ktbw)
    R_avg = np.mean(R_ktbw)
    
    print(R_avg / B_avg)
    print(R_avg / G_avg)
    
    for width in range(src.shape[0]):
        for height in range(src.shape[1]):
            for channel in range(src.shape[2]): 
                if channel == 0:
                    result[width][height][channel] = src[width][height][channel] * (R_avg / B_avg)
                if channel == 1:
                    result[width][height][channel] = src[width][height][channel] * (R_avg / G_avg)
                if channel == 2:
                    result[width][height][channel] = src[width][height][channel]
    return result


if __name__ == 'main':
    ### for globalTM
    radiance = cv.imread('../TestImg/memorial.hdr', -1)
    golden = cv.imread('../ref/p2_gtm.png')    
    ldr = globalTM(radiance, scale=1.0)       
    psnr = cv.PSNR(golden, ldr)
    print('psnr', psnr)

    ### for white balance
    img = np.random.rand(30, 30, 3)
    ktbw = (slice(0, 15), slice(0, 15))
    w_avg = img[0:15, 0:15, 2].mean()
    wb_result = whiteBalance(img, (0, 15), (0, 15))     # the x, y range is known to be white 
    result_avg = wb_result[ktbw].mean(axis=(0, 1))
    print(result_avg[0])    # result_avg[0] 和 [1] 要跟 w_avg近似
    print(result_avg[1])
    print(w_avg)

    ### for wb + globalTM
    golden_2 = cv.imread('../ref/p5_wb_gtm.png')
    wb_hdr = whiteBalance(radiance, (457, 481), (400, 412))
    test = globalTM(wb_hdr)
    psnr_2 = cv.PSNR(golden_2, test)
    print(psnr_2)

