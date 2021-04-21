"""
provided psnr function
"""

import numpy as np

def PSNR_UCHAR3(input_1, input_2, peak=255):
    [row,col,channel] = input_1.shape
    if input_1.shape != input_2.shape:
        print ("Warning!! Two image have different shape!!")
        return 0
    mse = ((input_1 - input_2)**2).sum() / float(row * col * channel)
    
    return 20*np.log10(peak) - 10*np.log10(mse)

if __name__ == '__main__':
    test = cv.imread('./lr.png')
    test_2 = cv.imread('../data/blurred_image/curiosity_small.png')
    golden = cv.imread('../ref_ans/curiosity_small/rl_deblur25.png')
    diff = golden - test
    lr_result_1 = np.load('../lr_result_1.npy')
    print(diff)
    print(PSNR_UCHAR3(test_2, golden))
