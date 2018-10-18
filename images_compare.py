# This code is to compare two images
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy
import math
from math import sqrt

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	return err

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# load the images -- the original, the original + contrast,
# and the original + photoshop
original = cv2.imread("images/1.png")
contrast = cv2.imread("images/4.png")
img1 = cv2.resize(original,(200,200))
img2 = cv2.resize(contrast,(200,200))

# convert the images to grayscale
original = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
m = mse(original, contrast)
s = ssim(original, contrast)
p = psnr(original, contrast)
print(s)
print(m)
print(p)
