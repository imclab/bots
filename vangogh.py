import random

import numpy
import scipy.misc
import skimage.transform

SCALE = 8
SAMPLES = 50000

img_high = scipy.misc.imread('VG_DT1567.jpg')
res_low = (img_high.shape[0]/8, img_high.shape[1]/8) 
img_low = skimage.transform.downscale_local_mean(img_high, (SCALE,SCALE,1))


a_in = numpy.zeros((SAMPLES, 8 * 8 * 3), dtype=numpy.float32)
a_out = numpy.zeros((SAMPLES, 16 * 16 * 3), dtype=numpy.float32)

samples = []
for i in range(SAMPLES):
    yh, xh = random.randint(8*SCALE, img_high.shape[0]-16*SCALE), random.randint(8*SCALE, img_high.shape[1]-16*SCALE)
    yl, xl = yh / SCALE, xh / SCALE

    a_in[i] = img_low[yl-4:yl+4,xl-4:xl+4].flatten() / 255.0 - 0.5
    a_out[i] = img_high[yh-8:yh+8,xh-8:xh+8].flatten() / 255.0 - 0.5
    
    samples.append((yh, xh))

# scipy.misc.imsave('VG_DT1567.low.png', img_low)

import sys
import logging
logging.basicConfig(format="%(message)s", level=logging.DEBUG, stream=sys.stdout)

from sknn.mlp import Regressor, Layer

print a_in.min(), a_in.max(), a_out.min(), a_out.max()

nn = Regressor(
        layers=[
            Layer("Rectifier", units=1024, pieces=2),
            Layer("Linear"),
        ],
        learning_rate=0.001,
        n_iter=5,
        verbose=1,
        valid_size=0.2)

nn.fit(a_in, a_out)

"""
# RECONSTRUCTION.
samples = []
for i in range(SAMPLES):
    yh, xh = random.randint(8*SCALE, img_high.shape[0]-16*SCALE), random.randint(8*SCALE, img_high.shape[1]-16*SCALE)
    yl, xl = yh / SCALE, xh / SCALE
    a_in[i] = img_low[yl-4:yl+4,xl-4:xl+4].flatten() / 255.0
    samples.append((yh, xh))
"""

a_test = nn.predict(a_in)

img_test = numpy.zeros(img_high.shape, dtype=numpy.float32)
for i, (yh, xh) in enumerate(samples):
    img_test[yh-8:yh+8,xh-8:xh+8] = (a_test[i].reshape(16,16,3) + 0.5) * 255.0

print "Reconstructing..."
scipy.misc.imsave('VG_DT1567.test.png', img_test)
