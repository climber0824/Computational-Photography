"""
Solve problems by ProxImaL
"""

# Proximal
import sys

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *

#import cvxpy as cvx
import numpy as np
from scipy import signal

from PIL import Image
#import cv2

import scipy.misc
import time

############################################################

#%% Load image
img = Image.open('../data/blurred_image/edgetaper/blur_edgetaper.png')  # opens the file using Pillow - it's not an array yet
b = np.asfortranarray(im2nparray(img))

# Kernel
K = Image.open('../data/kernel/kernel_medium.png')  # opens the file using Pillow - it's not an array yet
K = np.asfortranarray(im2nparray(K))
K /= np.sum(K)

K_rgb = np.zeros((K.shape[0], K.shape[1], 3))
K_rgb[:,:,0] = K
K_rgb[:,:,1] = K
K_rgb[:,:,2] = K
K = K_rgb

#%% Now test the solver with some sparse gradient deconvolution
'''
parameters (you are encouraged to test some different parameter sets)
'''
lamb = 0.01
eps_abs_rel = 1e-3
test_solver = 'pc'
max_iters = 1000

tstart = time.time()

#%% rgb channels
x = Variable(b.shape)

# model the problem by proximal
prob = Problem(poisson_norm(conv(K,x, dims=2), b) + lamb * group_norm1( grad(x, dims = 2), [3] ) + nonneg(x)) # formulate problem

# solve the problem
result = prob.solve(verbose=True,solver=test_solver,x0=b,eps_abs=eps_abs_rel, eps_rel=eps_abs_rel,max_iters=max_iters) # solve problem
x = x.value

# record time
t_int = time.time() - tstart
print( "Elapsed time: %f seconds.\n" %t_int )

# output color image
scipy.misc.toimage(x, cmin=0.0, cmax=1.0).save('../result/deblur_edgetaper_%s_la%.4f_eps%.1e_TVpossion.png' %(test_solver,lamb,eps_abs_rel))
