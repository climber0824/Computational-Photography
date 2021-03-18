# Functions for camera response calibration
# Put all your related functions for section 1-1 in this file.

import os
import cv2 as cv
import numpy as np

N = 256  # intensity levels
Z_max = 255
Z_min = 0
MID = int(0.5*(Z_max+Z_min))


def toy():
    """ A toy example to demonstrate how to setup equation matrix
    """
    # Setup Lagrange multiplier
    lambda_ = 20
    # Two pixel measurement examples for data term
    num_pixels = 17
    eq_num = 3
    var_num = N + num_pixels
    A = np.zeros((eq_num, var_num))
    b = np.zeros((eq_num, 1))
    w = np.arange(256)  # example
    # measurement 1: i=5 Zij = 15, delta_t = 7
    k = 0
    # w(Zij) * g(Zij) - w(Zij)lnEi = w(Zij) ln delt_t
    A[k, 15] = w[15]
    A[k, N+5] = -w[15]
    b[k] = w[15] * np.log(7)
    k += 1
    # measurement 2: i=2 Zij = 17, delta_t = 8
    A[k, 17] = w[17]
    A[k, N+2] = -w[17]
    b[k] = w[17] * np.log(8)
    k += 1
    # smoothness for z=7
    A[k, 6] = lambda_ * w[7]
    A[k, 7] = -2 * lambda_ * w[7]
    A[k, 8] = lambda_ * w[7]


def loadExposures(source_dir):
    """load bracketing images folder

    Args:
        source_dir (string): folder path containing bracking images and a image_list.txt file
        image_list.txt contains lines of 
        image_file_name, exposure time(or scale*time), ... others
    Returns:
       tuples (img_list, exposure_times) : ndarray(N, height, width, ch), list of float
    """
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *_) = line.split()
        filenames += [filename]
        exposure_times += [float(exposure)]
    img_list = [cv.imread(os.path.join(source_dir, f), 1) for f in filenames]
    img_list = np.array(img_list)
    return img_list, exposure_times


def estimateResponse(img_samples, etime_list, lambda_=50):
    """Estimate camera response for bracketing images

    Args:
        img_samples (list of ndarray): list of bracketing images (1ch)
        etime_list (list of float32): list of exposure time
        lambda_ (float32): Lagrange multiplier
    """
    response = np.zeros(N)
    return response


def constructRadiance(img_list, response, etime_list):
    """Construct irradiance map from brackting images

    Args:
        img_list (ndarray (N, Y, X)): N bracketing images (1 channel)
        response (ndarray (256,)): response mapping
        etime_list (list of float32): list of exposure time
    """
    dtype = np.float32
    result = np.zeros(img_list.shape[1:], dtype=dtype)
    return result


def pixelSample(img_list):
    # trivial periodic sample
    sample = img_list[:, ::64, ::64, :].reshape(len(img_list), -1, 3)
    return sample


def wholeFlow(src_path, lambda_):
    img_list, exposure_times = loadExposures(src_path)
    radiance = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_samples = pixelSample(img_list)
    for ch in range(3):
        response = estimateResponse(
            pixel_samples[..., ch], exposure_times, lambda_)
        radiance[..., ch] = constructRadiance(
            img_list[..., ch], response, exposure_times)
    return radiance


if __name__ == '__main__':
    """ 
    list your develop log or experiments for cr calibration here
    """
    print('cr_calibration')
    # Example matrix Ax = b for image samples(/ref/p1_pixel_samples.npy) and exposure times (/ref/p1_etimes.npy).
    # These two matrix should generate same response result as in test_HW1.test_estimateResponse()
    example_A = np.load('../ref/p1_A.npy')
    example_B = np.load('../ref/p1_b.npy')