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
    rl_ker = cv.imread('../data/kernel/kernel_small.png')
    #print(rl_source.shape)
    #print(rl_ker)
    #print(rl_source / 255)
    print('ker_normal', rl_ker / np.sum(rl_ker))