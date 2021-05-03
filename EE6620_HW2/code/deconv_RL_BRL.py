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

    # Calculate k_in_star, r_omega, sigma_s
    k_in_star = k_in[::-1, ::-1]
    r_omega = rk // 2
    sigma_s = (r_omega / 3)**2

    # Precompute f(|x-y|)
    ax = np.linspace(-r_omega, r_omega, r_omega*2+1)
    xx, yy = np.meshgrid(ax, ax)
    Gaussian = np.exp(-(np.square(xx) + np.square(yy)) / sigma_s / 2)

    # Deblur the image
    BRL_result = img_in.copy()
    gradient_B = np.zeros((rows, cols))
    tmp = np.zeros((rows, cols, ch))
    for k in range(ch):
        for _ in range(max_iter):
            img_in_pad = np.pad(BRL_result[:,:,k], r_omega, mode="symmetric")
            # $\nabla E_B(I^t) = 2 * \sum_{y \in \Omega} I^d_y(x)$
            # $I^d_y(x) = \exp(\frac{-(x - y)^2}{2\sigma_s})* \exp(\frac{-(I(x) - I(y))^2}{2\sigma_r}) * \frac{I(x) - I(y)}{\sigma_r}$
            for i in range(rows):
                for j in range(cols):
                    I_diff = BRL_result[i,j,k] - img_in_pad[i:i+2*r_omega+1, j:j+2*r_omega+1]
                    gradient_B[i, j] = 2 * np.sum(Gaussian * np.exp(-np.square(I_diff) / 2 / sigma_r) * I_diff / sigma_r)
            # $\text{tmp} = \frac{B}{I^t \otimes K}$
            tmp[:,:,k] = img_in[:,:,k] / (scipy.signal.convolve2d(BRL_result[:,:,k], k_in[:,:,k], boundary='symm', mode='same') + 0.0001)
            # $I^{t+1}= \frac{I^t}{1+\lambda * \nabla E_B(I^t)} [K^* \otimes \text{tmp}]$
            BRL_result[:,:,k] = BRL_result[:,:,k] * scipy.signal.convolve2d(tmp[:,:,k], k_in_star[:,:,k], boundary='symm', mode='same') / (1 + lamb_da * gradient_B)
            print('iter:', _)

    # Convert img_in back to nonlinear domain if to_linear == 'True'
    if to_linear == 'True':
        gamma = 2.2
        DBL_MIN = sys.float_info.min
        R = BRL_result[:,:,0] ** (1/gamma)
        G = BRL_result[:,:,1] ** (1/gamma)
        B = BRL_result[:,:,2] ** (1/gamma)
        R[R < DBL_MIN] = DBL_MIN
        G[G < DBL_MIN] = DBL_MIN
        B[B < DBL_MIN] = DBL_MIN
        BRL_result[:,:,0] = R
        BRL_result[:,:,1] = G
        BRL_result[:,:,2] = B

    # Convert BRL_result back to int
    BRL_result[BRL_result > 1] = 1
    BRL_result[BRL_result < 0] = 0
    BRL_result = np.round(BRL_result * 255).astype('uint8')

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
    k_in = k_in.astype(np.float)
    k_in = k_in / np.sum(k_in)
    I_in = I_in.astype(np.float)
    I_in  /= 255.0
    img_in = img_in.astype(np.float)
    img_in /= 255.0
    conv = np.zeros_like(img_in, dtype=np.float32)
    energy = np.zeros_like(img_in, dtype=np.float32)
    
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
    
    for k in range(img_in.shape[2]):
        conv[:,:,k] = scipy.signal.convolve2d(I_in[:,:,k], k_in[:,:,k], boundary='symm', mode='same')
        energy[:,:,k] = conv[:,:,k] - img_in[:,:,k] * np.log(conv[:,:,k])
    
    if to_linear == 'True':
        gamma = 2.2
        DBL_MIN = sys.float_info.min
        R = energy[:,:,0] ** (1/gamma)
        G = energy[:,:,1] ** (1/gamma)
        B = energy[:,:,2] ** (1/gamma)
        R[R < DBL_MIN] = DBL_MIN
        G[G < DBL_MIN] = DBL_MIN
        B[B < DBL_MIN] = DBL_MIN
        energy[:,:,0] = R
        energy[:,:,1] = G
        energy[:,:,2] = B
    
    energy /= 255.0
    energy = np.sum(energy)
    
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
    rl_small = cv.imread('../data/blurred_image/curiosity_small.png')
    rl_medium = cv.imread('../data/blurred_image/curiosity_medium.png')
    golden = cv.imread('../ref_ans/curiosity_small/rl_deblur25.png')
    rl_ker_small = cv.imread('../data/kernel/kernel_small.png')
    rl_ker_medium = cv.imread('../data/kernel/kernel_medium.png')
    rl_small_result = cv.imread('../result/RL_small.png')
    rl_medium_result = cv.imread('../result/RL_medium.png')
    
    """
    rl_result = RL(rl_medium, rl_ker, 55, False)
    cv.imwrite('../result/RL_medium.png', rl_result)
    """
    print(RL_energy(rl_small, rl_ker_small, rl_small_result, False))
