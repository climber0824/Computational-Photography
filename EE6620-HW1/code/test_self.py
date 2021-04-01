import unittest
import numpy as np
from functools import partial
import cv2 as cv
from cr_calibration import estimateResponse, constructRadiance, loadExposures
from tm import globalTM, localTM, gaussianFilter, bilateralFilter, whiteBalance


### cr_calibration
"""
samples = np.load('../ref/p1_pixel_samples.npy')
etime = np.load('../ref/p1_et_samples.npy')
golden = np.load('../ref/p1_resp.npy')

print('samples',  samples)
print('etime', etime)
#print('golden', golden)
print('math', (np.log(etime)))
"""

def estimateResponse(img_samples, etime_list, lambda_=50):
    """Estimate camera response for bracketing images

    Args:
        img_samples (list of ndarray): list of bracketing images (1ch)
        etime_list (list of float32): list of exposure time
        lambda_ (float32): Lagrange multiplier
    """



    response = np.zeros(N)
    return response


"""
resp_test = estimateResponse(samples, etime)
mse = np.mean((golden - resp_test)**2)
assertLessEqual(mse, 0.1)
print(mse)
"""
### cr_calibration_end


"""
### global_tone_mapping
radiance = cv.imread('../TestImg/memorial.hdr', -1)
golden = cv.imread('../ref/p2_gtm.png')

print('radiance', radiance.shape)
print('golden', golden.shape)
"""

def globalTM(src, scale=1.0):
    """Global tone mapping (section 1-2)

    Args:
        src (ndarray, float32): source radiance image
        scale (float, optional): scaling factor (Defaults to 1.0)
    """
    result = np.zeros_like(src, dtype=np.uint8)

    x_max = []
    x_hat = []
    x_max_candi = []

    for width in range(src.shape[0]):              # photo_width
        for height in range(src.shape[1]):         # photo_height
            for channels in range(src.shape[2]):   # photo_channel
                x_max_candi.append(radiance[width][height][channels])
            x_max = np.array(max(x_max_candi))
            x_hat[width][height] = np.power(2, 1*(np.log2(src[width][height]) - np.log2(x_max) ) + np.log2(x_max))
            x_hat = x_hat.append(x_hat[width][height])

    gamma = 2.2
    result = result + x_hat
    result = result ^ (1 / gamma)

    return result

"""
### problem: how to find out X_max ??
#ldr = globalTM(radiance, scale=1.0)
#psnr = cv.PSNR(golden, ldr)
x_max_candi = []
result = np.zeros_like(radiance, dtype=np.uint8)
for width in range(radiance.shape[0]):                  # photo_width
        for height in range(radiance.shape[1]):         # photo_height
            for channels in range(radiance.shape[2]):   # photo_channel
                x_max_candi.append(radiance[width][height][channels])
        x_max = np.array(max(x_max_candi))
        print('x_max', x_max.shape)
        #print('rad', np.log2(radiance[width][height]))

#print('x_max', x_max)

#print('result', result.shape[0])
#print((result+radiance).shape)
#ssertGreaterEqual(psnr, 45)
#print('psnr', psnr)
"""

##################################

### Local tone mapping with Gaussian filter

def localTM(src, imgFilter, scale=3):
    """Local tone mapping (section 1-3)

    Args:
        src (ndarray, float32): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float, optional): scaling factor (Defaults to 3)
    """
    
    result = np.zeros_like(src, dtype=np.uint8)
    return result

"""
radiance = cv.imread('../TestImg/vinesunset.hdr', -1)
golden = cv.imread('../ref/p3_ltm.png')
print('rad', radiance.shape)
print('gold', golden.shape)
gauhw1 = partial(gaussianFilter, N=35, sigma_s=100)
test = localTM(radiance, gauhw1, scale=3)
psnr = cv.PSNR(golden, test)
"""


def gaussianFilter(src, N=35, sigma_s=100):
    """Gaussian filter (section 1-3)

    Args:
        src (ndarray): source image
        N (int, optional): window size of the filter (Defaults to 35)
            filter indices span [-N/2, N/2]
        sigma_s (float, optional): standard deviation of Gaussian filter (Defaults to 100)
    """

    def gfunc(x, y, sigma_s):
        return np.exp(-(x**2 + y**2) / (2*(sigma_s**2)))
    # Window size should be odd
    assert N % 2
    dtype = np.float32
    result = np.zeros_like(src, dtype=dtype)
    src_padded = np.pad(src, (N//2, N//2), 'symmetric')
    assert src_padded.shape[0] == 13


    for i in range(src_padded.shape[0]):
        for j in range(src_padded.shape[1]):
            result[i, j] = gfunc(i-src_padded.shape[0]//2, j-src_padded.shape[1]//2, sigma_s)
    result = result / np.sum(result)
            


    return result


impulse = np.load('../ref/p3_impulse.npy')
golden = np.load('../ref/p3_gaussian.npy').astype(float)

N = 5
pad_a = np.pad(impulse, (N//2, N//2), 'symmetric')
print(pad_a.shape)
"""
sigma_s =15
dtype = np.float32
gaussian_kernel = np.zeros_like(impulse, dtype=dtype)
for i in range(pad_a.shape[0]):
    for j in range(pad_a.shape[1]):
        #print(pad_a[i][j])
        gaussian_kernel[i, j] = np.exp(-(np.power(i - N//2, 2) + np.power(j - N//2, 2) / 2 / np.power(2, sigma_s)))
        print('gaussian', gaussian_kernel[i, j], i, j)
print('gaussian', gaussian_kernel)
"""
test = gaussianFilter(impulse, 5, 15).astype(float)
print('test', test)
psnr = cv.PSNR(golden, test)




####################
### bilateral Filter

def bilateralFilter(src, N=35, sigma_s=100, sigma_r=0.8):
    """Bilateral filter (section 1-4)

    Args:
        src (ndarray): source image
        N (int, optional): window size of the filter (Defaults to 35)
            filter indices span [-N/2, N/2]
        sigma_s (float, optional): spatial standard deviation of bilateral filter (Defaults to 100)
        sigma_r (float, optional): range standard deviation of bilateral filter (Defaults to 0.8)
    """
    # Window size should be odd
    assert N % 2
    dtype = np.float32
    result = np.zeros_like(src, dtype=dtype)
    return result


step = np.load('../ref/p4_step.npy')
golden = np.load('../ref/p4_bilateral.npy').astype(float)
#print('step', step)
#print('gold', golden)
#test = bilateralFilter(step, 9, 50, 10).astype(float)
#psnr = cv.PSNR(golden, test)
