import unittest
import numpy as np
from functools import partial
import cv2 as cv
from cr_calibration import estimateResponse, constructRadiance, loadExposures
from tm import globalTM, localTM, gaussianFilter, bilateralFilter, whiteBalance
import math

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



### global_tone_mapping

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


radiance = cv.imread('../TestImg/memorial.hdr', -1)
golden = cv.imread('../ref/p2_gtm.png')
x_max = np.max(radiance, axis = 2)

scale = 1.0

x_hat = np.zeros_like(radiance, dtype=np.uint8)
count = 0
result = np.zeros_like(radiance, dtype=np.uint8)


for width in range(radiance.shape[0]):
    for height in range(radiance.shape[1]):
        for channel in range(radiance.shape[2]):
            if radiance[width][height][channel] >= 1:
                radiance[width][height][channel] = 1
            x_hat[width][height][channel]=255 * ( 2 ** (scale*(np.log2(radiance[width][height][channel])-\
                     np.log2(x_max[width][height]) + np.log2(x_max[width][height]))))** (1/2.2)
#rint('x_hat', x_hat)            
psnr = cv.PSNR(golden, x_hat)
print('psnr', psnr)


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



    for i in range(src_padded.shape[0]):
        for j in range(src_padded.shape[1]):
            result[i, j] = gfunc(i-src_padded.shape[0]//2, j-src_padded.shape[1]//2, sigma_s)
    result = result / np.sum(result)
            


    return result

"""
impulse = np.load('../ref/p3_impulse.npy')
golden = np.load('../ref/p3_gaussian.npy').astype(float)
print('golden', golden)
N = 5
pad_a = np.pad(impulse, (N//2, N//2), 'symmetric')
print(pad_a.shape)
print(pad_a)
"""

"""
sigma_s =15
dtype = np.float32
gaussian_kernel = np.zeros_like(impulse, dtype=dtype)
for i in range(N//2 + 1, pad_a.shape[0] - N//2):
    for j in range(N//2 + 1, pad_a.shape[1] - N//2):
        #print(pad_a[i][j])
        gaussian_kernel[i-1, j-1] = np.exp(-(np.power(i - N//2, 2) + np.power(j - N//2, 2) / 2 / np.power(2, sigma_s)))
        print('gaussian', gaussian_kernel[i-1, j-1], i-1, j-1)
print('gaussian', gaussian_kernel)
test = gaussianFilter(impulse, 5, 15).astype(float)
print('test', test)
psnr = cv.PSNR(golden, test)
"""

##### https://blog.csdn.net/qq_16013649/article/details/78784791
def gaussian_2d_kernel(kernel_size = 5,sigma = 15):
    
    kernel = np.zeros([kernel_size,kernel_size])
    center = kernel_size//2
    
    if sigma == 0:
        sigma = ((kernel_size-1)*0.5 - 1)*0.3 + 0.8
    
    s = 2*(sigma**2)
    sum_val = 0
    for i in range(0,kernel_size):
        for j in range(0,kernel_size):
            x = i-center
            y = j-center
            kernel[i,j] = np.exp(-(x**2+y**2) / s)
            sum_val += kernel[i,j]
            #/(np.pi * s)
    sum_val = 1/sum_val
    return kernel*sum_val

#print(gaussian_2d_kernel)


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

"""
step = np.load('../ref/p4_step.npy')
golden = np.load('../ref/p4_bilateral.npy').astype(float)
print('step', step)
print('gold', golden)
#test = bilateralFilter(step, 9, 50, 10).astype(float)
#psnr = cv.PSNR(golden, test)
"""

#####   https://github.com/nuwandda/Bilateral-Filter/blob/master/bilateral_filter.py
def gaussian(x,sigma):
    return (-np.abs(-(x**2) / (2*(sigma**2))))
    #return (1.0/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return np.sqrt(np.abs((x1-x2)**2-(y1-y2)**2) )

def bilateral_filter(image, diameter, sigma_s, sigma_r):
    new_image = np.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x = row - (diameter/2 - k)
                    n_y = col - (diameter/2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    gs = gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_s)
                    gr = gaussian(distance(n_x, n_y, row, col), sigma_r)
                    #wp = gs * gr
                    wp = np.exp(-gs - gr)
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = filtered_image
    return new_image

"""
test = bilateral_filter(step, 9, 50, 10).astype(float)
print('test', test)
psnr = cv.PSNR(golden, test)
print('psnr', psnr)
"""
##### ends


####### https://github.com/DuJunda/BilateralFilter/blob/master/BilateralFilter.py
"""
def bilateral_filter_image(image_matrix, window_length=9,sigma_color=50,sigma_space=10,mask_image_matrix = None):
    mask_image_matrix = np.zeros(
        (image_matrix.shape[0], image_matrix.shape[1])) if mask_image_matrix is None else mask_image_matrix#default: filtering the entire image
    image_matrix = image_matrix.astype(np.int32)#transfer the image_matrix to type int32，for uint cann't represent the negative number afterward
    
    def limit(x):
        x = 0 if x < 0 else x
        x = 255 if x > 255 else x
        return x
    limit_ufun = np.vectorize(limit, otypes=[np.uint8])
    def look_for_gaussion_table(delta):
        return delta_gaussion_dict[delta]
    def generate_bilateral_filter_distance_matrix(window_length,sigma):
        distance_matrix = np.zeros((window_length,window_length,3))
        left_bias = int(math.floor(-(window_length - 1) / 2))
        right_bias = int(math.floor((window_length - 1) / 2))
        for i in range(left_bias,right_bias+1):
            for j in range(left_bias,right_bias+1):
                distance_matrix[i-left_bias][j-left_bias] = math.exp(-(i**2+j**2)/(2*(sigma**2)))
        return distance_matrix
    delta_gaussion_dict = {i: math.exp(-i ** 2 / (2 *(sigma_color**2))) for i in range(256)}
    look_for_gaussion_table_ufun = np.vectorize(look_for_gaussion_table, otypes=[np.float64])#to accelerate the process of get the gaussion matrix about color.key:color difference，value:gaussion weight
    bilateral_filter_distance_matrix = generate_bilateral_filter_distance_matrix(window_length,sigma_space)#get the gaussion weight about distance directly

    margin = int(window_length / 2)
    left_bias = math.floor(-(window_length - 1) / 2)
    right_bias = math.floor((window_length - 1) / 2)
    filter_image_matrix = image_matrix.astype(np.float64)

    for i in range(0 + margin, image_matrix.shape[0] - margin):
        for j in range(0 + margin, image_matrix.shape[1] - margin):
            if mask_image_matrix[i][j]==0:
                filter_input = image_matrix[i + left_bias:i + right_bias + 1,
                               j + left_bias:j + right_bias + 1]#get the input window
                bilateral_filter_value_matrix = look_for_gaussion_table_ufun(np.abs(filter_input-image_matrix[i][j]))#get the gaussion weight about color
                bilateral_filter_matrix = np.multiply(bilateral_filter_value_matrix, bilateral_filter_distance_matrix)#multiply color gaussion weight  by distane gaussion weight to get the no-norm weigth matrix
                bilateral_filter_matrix = bilateral_filter_matrix/np.sum(bilateral_filter_matrix,keepdims=False,axis=(0,1))#normalize the weigth matrix
                filter_output = np.sum(np.multiply(bilateral_filter_matrix,filter_input),axis=(0,1)) #multiply the input window by the weigth matrix，then get the sum of channels seperately
                filter_image_matrix[i][j] = filter_output
    filter_image_matrix = limit_ufun(filter_image_matrix)#limit the range
    return filter_image_matrix


step = np.load('../ref/p4_step.npy')
golden = np.load('../ref/p4_bilateral.npy').astype(float)
bilateral_filtered = bilateral_filter_image(step)
"""
########################################
######## White balance
# step1. Find the BGR_avg of the zone(to be white)
# step2. Use the formula to calculate G' = G * (R_avg / G_avg) , G and G' stand for the all img

def whiteBalance(src, y_range, x_range):
    """White balance based on Known to be White(KTBW) region
    Args:
        src (ndarray): source image
        y_range (tuple of 2): location range in y-dimension
        x_range (tuple of 2): location range in x-dimension
    """
    result = np.zeros_like(src)
    return result

"""
img = np.random.rand(30, 30, 3)
ktbw = (slice(0, 15), slice(0, 15))
w_avg = img[0:15, 0:15, 2].mean()
#print(ktbw[0])
### test section:
#for width in range(img.shape[0]):
#    print(width)







wb_result = whiteBalance(img, (0, 15), (0, 15))     # the x, y range is known to be white 
result_avg = wb_result[ktbw].mean(axis=(0, 1))
#assertAlmostEqual(result_avg[0], w_avg)
#assertAlmostEqual(result_avg[1], w_avg)
#print('img', img)
#print('ktbw', ktbw)
"""
