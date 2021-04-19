"""
RL and BRL functions
"""
import numpy as np
import cv2 as cv

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

    RL_result = np.zeros_like(img_in, dtype = np.float32)
    I = img_in / 255
    I_iter = np.zeros_like(I, dtype = np.float32)
    k_normalized = k_in / np.sum(k_in)
    for i in range(max_iter):
        ratio = B / np.convolve(np.convolve(I, k_normalized))
        I = I * k_normalized_star 
        I_iter += I

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


if __name__ == '__main__':
    rl_source = cv.imread('../data/blurred_image/curiosity_small.png')
    golden = cv.imread('../ref_ans/curiosity_small/rl_deblur25.png')
    I = rl_source / 255
    rl_ker = cv.imread('../data/kernel/kernel_small.png')
    rl_ker_nor = rl_ker / np.sum(rl_ker)
    rl_ker_nor_star = np.flipud(rl_ker_nor)
    denominator = np.zeros_like(rl_source, dtype = np.float32)
    ratio = np.zeros_like(rl_source, dtype = np.float32)
    ratio_2 = np.zeros_like(rl_source, dtype = np.float32)
    I_iter = I.copy()
    half_window_size = (rl_ker.shape[0] - 1) // 2
    rl_source_pad = np.pad(rl_source, half_window_size, 'symmetric')
    max_iter = 25
    
    start_time = time.time()
    for j in range(max_iter):
        for i in range(rl_source.shape[2]):
            denominator[:,:,i] = convolution2d(rl_source_pad[:,:,i], rl_ker_nor[:,:,i], 0)
            ratio[:,:,i] = rl_source[:,:,i] / denominator[:,:,i]
            ratio_pad = np.pad(ratio, half_window_size, 'symmetric')
            ratio_2[:,:,i] = convolution2d(ratio_pad[:,:,i], rl_ker_nor[:,:,i], 0)
            I_iter[:,:,i] = I_iter[:,:,i] * ratio_2[:,:,i]
        print('iter:', j, 'psnr', PSNR_UCHAR3(I_iter * 255, golden))
        
    lr_photo = 255 * I_iter
    print('time:', (time.time() - start_time) / 60)
    cv.imwrite('lr.png', lr_photo)
