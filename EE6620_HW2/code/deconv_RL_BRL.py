"""
RL and BRL functions
"""
import numpy as np
import cv2 as cv
import time

from load_psnr import PSNR_UCHAR3

def RL(img_in, k_in, max_iter, to_linear):
    """ RL deconvolution
            Args:
                img_in (int np.array, value:[0,255]): Blurred image
                k_in (int np.array, value:[0,255]): blur kernel
                max_iter (int): total iteration count
                to_linear (bool): The flag indicates whether the input image should deblur in linear domain color space or not
            Returns:
                im_out (int np.array, value:[0,255]): RL-deblurred image
            Todo:
                RL deconvolution
    """
    ### I is intensity, should be normalized to [0, 1]
    ### kernel K should be normalized s.t. (k1+k2+...ki) = 1
    ### use symmetric padding
    ### B is blurred img

     # Get the shape of img_in
    rows, cols, ch = img_in.shape

    # Convert the type to float and normalize it
    img_in = img_in.astype(np.float)
    img_in /= 255
    k_in = k_in.astype(np.float)
    k_in = k_in / np.sum(k_in)

    # Convert img_in to linear domain if to_linear == 'True'
    if to_linear == 'True':
        gamma = 2.2
        DBL_MIN = sys.float_info.min
        R = img_in[:,:,0] ** gamma
        G = img_in[:,:,1] ** gamma
        B = img_in[:,:,2] ** gamma
        R[R < DBL_MIN] = DBL_MIN
        G[G < DBL_MIN] = DBL_MIN
        B[B < DBL_MIN] = DBL_MIN
        img_in[:,:,0] = R
        img_in[:,:,1] = G
        img_in[:,:,2] = B

    # Calculate k_in_star
    k_in_star = k_in[::-1, ::-1]

    # Deblur the image
    RL_result = img_in.copy()
    tmp = np.zeros_like(RL_result, dtype = np.float32)
    for k in range(ch):
        for _ in range(max_iter):
            # $\text{tmp} = \frac{B}{I^t \otimes K}$
            #tmp = img_in[:,:,k] / (scipy.signal.convolve2d(RL_result[:,:,k], k_in, boundary='symm', mode='same') + 0.0001)
            tmp[:,:,k] = img_in[:,:,k] / (scipy.signal.convolve2d(RL_result[:,:,k], k_in[:,:,k], boundary='symm', mode='same') + 0.0001)
            # $I^{t+1} = I^t [K^* \otimes \text{tmp}]$
            print(tmp.shape)
            #RL_result[:,:,k] = RL_result[:,:,k] * scipy.signal.convolve2d(tmp[:,:,k], k_in_star[:,:,k], boundary='symm', mode='same')
            RL_result[:,:,k] = RL_result[:,:,k] * scipy.signal.convolve2d(tmp[:,:,k], k_in_star[:,:,k], boundary='symm', mode='same')
            print('iter:', _ , 'PSNR', PSNR_UCHAR3(golden, RL_result))
    # Convert img_in back to nonlinear domain if to_linear == 'True'
    if to_linear == 'True':
        gamma = 2.2
        DBL_MIN = sys.float_info.min
        R = RL_result[:,:,0] ** (1/gamma)
        G = RL_result[:,:,1] ** (1/gamma)
        B = RL_result[:,:,2] ** (1/gamma)
        R[R < DBL_MIN] = DBL_MIN
        G[G < DBL_MIN] = DBL_MIN
        B[B < DBL_MIN] = DBL_MIN
        RL_result[:,:,0] = R
        RL_result[:,:,1] = G
        RL_result[:,:,2] = B

    # Convert RL_result back to int
    # RL_result[RL_result > 1] = 1
    # RL_result[RL_result < 0] = 0
    RL_result = np.round(RL_result * 255).astype('uint8')

    return RL_result

def BRL(img_in, k_in, max_iter, lamb_da, sigma_r, rk, to_linear):
    """ BRL deconvolution
            Args:
                img_in (int np.array, value:[0,255]): Blurred image
                k_in (int np.array, value:[0,255]): blur kernel
                max_iter (int): total iteration count
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                to_linear (bool): The flag indicates whether the input image should deblur in linear domain color space or not
            Returns:
                im_out (int np.array, value:[0,255]): BRL-deblurred image
            Todo:
                BRL deconvolution
    """

    return BRL_result


def RL_energy(img_in, k_in, I_in, to_linear):
    """ RL energy
            Args:
                img_in (int np.array, value:[0,255]): Blurred image
                k_in (int np.array, value:[0,255]): blur kernel
                I_in (int np.array, value:[0,255]): Your deblured image
                to_linear (bool): The flag indicates whether the input image should deblur in linear domain color space or not
            Returns:
                energy (float): RL-deblurred energy
            Todo:
                RL energy
    """

    
    return energy

def BRL_energy(img_in, k_in, I_in, lamb_da, sigma_r, rk, to_linear):
    """ BRL energy
            Args:
                img_in (int np.array, value:[0,255]): Blurred image
                k_in (int np.array, value:[0,255]): blur kernel
                I_in (int np.array, value:[0,255]): Your deblured image
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                to_linear (bool): The flag indicates whether the input image should deblur in linear domain color space or not
            Returns:
                energy (float): BRL-deblurred energy
            Todo:
                BRL energy
    """

    
    return energy

def convolution2d(image, kernel, bias):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) + bias
    return new_image


if __name__ == '__main__':
    rl_source = cv.imread('../data/blurred_image/curiosity_small.png')
    golden = cv.imread('../ref_ans/curiosity_small/rl_deblur25.png')
    I = rl_source / 255
    rl_ker = cv.imread('../data/kernel/kernel_small.png')
    rl_ker_nor = rl_ker / np.sum(rl_ker)
    rl_ker_nor_star = rl_ker_nor.copy()
    denominator = np.zeros_like(rl_source, dtype = np.float32)
    ratio = np.zeros_like(rl_source, dtype = np.float32)
    ratio_2 = np.zeros_like(rl_source, dtype = np.float32)
    I_iter = I.copy()
    half_window_size = (rl_ker.shape[0] - 1) // 2
    rl_source_pad = np.pad(rl_source, half_window_size, 'symmetric')
    max_iter = 25
    N = rl_ker.shape[0]

    ### k* sol_1:
    for i in range(-int((N-1)/2), int((N-1)/2)):
        for j in range(-int((N-1)/2), int((N-1)/2)):
            rl_ker_nor_star[i][j] = rl_ker_nor[-i][-j]
    
    print(rl_ker_nor_star)


    ### k* sol_2:
    k_star = np.flip(rl_ker_nor_star, axis=0)
    k_star = np.flip(k_star, axis=1)
    print('************************\n', k_star)


    
    start_time = time.time()
    for j in range(max_iter):
        for i in range(rl_source.shape[2]):
            denominator[:,:,i] = convolution2d(rl_source_pad[:,:,i], rl_ker_nor[:,:,i], 0)
            ratio[:,:,i] = rl_source[:,:,i] / denominator[:,:,i]
            ratio_pad = np.pad(ratio, half_window_size, 'symmetric')
            ratio_2[:,:,i] = convolution2d(ratio_pad[:,:,i], rl_ker_nor_star[:,:,i], 0)
            I_iter[:,:,i] = I_iter[:,:,i] * ratio_2[:,:,i]
        print('iter:', j, 'psnr', PSNR_UCHAR3(I_iter * 255, golden))
        print('time:', (time.time() - start_time) / 60)
        
    lr_photo = 255 * I_iter
    print('time:', (time.time() - start_time) / 60)
    cv.imwrite('lr.png', lr_photo)
    
