# Functions for tone mapping
# Put all your related functions for section 1-2~1-5 in this file.
import numpy as np
gamma = 2.2


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


def localTM(src, imgFilter, scale=3):
    """Local tone mapping (section 1-3)

    Args:
        src (ndarray, float32): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float, optional): scaling factor (Defaults to 3)
    """

    result = np.zeros_like(src, dtype=np.uint8)
    gamma = 2.2
    scale = 3
    result = np.zeros(np.shape(src))
    R = src[:,:,0]
    G = src[:,:,1]
    B = src[:,:,2]
    I = (R + G + B)/3
    C_R = R / I
    C_G = G / I
    C_B = B / I
    L = np.log2(I)
    L_B = imgFilter(src, window_size, sigma_s, sigma_r)
    L_D = L - L_B
    L_min = np.min(L_B)
    L_max = np.max(L_B)
    L_B_p = (L_B - L_max) * scale / (L_max - L_min)
    I_p = 2 ** (L_B_p + L_D)
    result[:,:,0] = (C_R*I_p)**(1.0/gamma)
    result[:,:,1] = (C_G*I_p)**(1.0/gamma)
    result[:,:,2] = (C_B*I_p)**(1.0/gamma)
    result[result > 1] = 1
    result[result < 0] = 0
    result = np.round(result) * 255).astype('uint8')

    return result


def gaussianFilter(src, N=35, sigma_s=100):
    """Gaussian filter (section 1-3)

    Args:
        src (ndarray): source image
        N (int, optional): window size of the filter (Defaults to 35)
            filter indices span [-N/2, N/2]
        sigma_s (float, optional): standard deviation of Gaussian filter (Defaults to 100)
    """
    # Window size should be odd
    assert N % 2
    dtype = np.float32
    result = np.zeros_like(src, dtype=dtype)

    R = src[:,:,0]
    G = src[:,:,1]
    B = src[:,:,2]
    I = (R + G + B)/3
    L = np.log2(I)
    rows, cols, __ = np.shape(src)
    half_window_size = (N-1)//2
    L_pad = np.pad(L, half_window_size, 'symmetric')

    windows = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            windows[i,j] = np.exp(-float((i-half_window_size)**2+(j-half_window_size)**2) / (2.0*sigma_s**2))

    for i in range(rows):
        for j in range(cols):
            result[i, j] = np.sum(np.multiply(L_pad[i:i+N, j:j+N], windows))
    result = result / np.sum(windows)

    return result


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
    # HDRIMG -> src ; LB -> result ; window_size -> N
    assert N % 2
    dtype = np.float32
    result = np.zeros_like(src, dtype=dtype)
    half_window_size = (N-1)//2
    rows, cols, __ = np.shape(src)
    R = src[:,:,0]
    G = src[:,:,1]
    B = src[:,:,2]
    I = (R + G + B)/3
    L = np.log2(I)
    L_pad = np.pad(L, half_window_size, 'symmetric')
    
    gaussian_windows = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            gaussian_windows[i,j] = np.exp(-((i-half_window_size)**2+(j-half_window_size)**2) / (2.0*sigma_s**2))

    for i in range(rows):
        for j in range(cols):
            windows = np.exp(-(L_pad[i:i+N, j:j+N] - L_pad[i+half_window_size, j+half_window_size])**2) / (2.0*sigma_r**2) * gaussian_windows
            result[i, j] = np.sum(np.multiply(L_pad[i:i+N, j:j+N], windows)) / np.sum(windows)

    return result


def whiteBalance(src, y_range, x_range):
    """White balance based on Known to be White(KTBW) region

    Args:
        src (ndarray): source image
        y_range (tuple of 2): location range in y-dimension
        x_range (tuple of 2): location range in x-dimension
    """
    result = np.zeros_like(src)
    return result


if __name__ == '__main__':
    """
    list your develop log or experiments for tone mapping here
    """
    print('tone mapping')





    print('start ')
    HDRIMG = np.load('../ref/p4_imgpatch.npy')
    img_joint_bilateral = local_tone_mapping(HDRIMG,Filter=bilateralFilter,N=35,sigma_s=100,sigma_r=0.8)
    golden = cv.imread('../ref/p4_ltm_patch.png')
    psnr_p4 = cv.PSNR(golden, img_joint_bilateral)
    print('psnr_p4', psnr_p4)

