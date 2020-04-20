import skimage
from skimage.io import imread, imsave
from skimage.measure import compare_psnr
import numpy as np
from numpy.linalg import norm
import sys
from scipy.signal import convolve2d as cnv
from tqdm import tqdm

from utils import progress_bar
from skimage.restoration import denoise_bilateral

def S(x, y, z):
    width = max(abs(x), abs(y)) + 1
    padded = np.pad(z, [(width, width), (width, width)], mode='reflect')
    return np.sum(np.abs(padded[width+y:-width+y, width+x:-width+x] - z))

def sgn(z):
    ret = z.copy()
    ret[ret > 0] = 1
    ret[ret == 0] = 0
    ret[ret < 0] = -1
    return ret

def df(z, kernel, u):
    ret = cnv(z, kernel, mode='same', boundary='symm')
    ret = 2 * cnv(ret, kernel.T, mode='same', boundary='symm')
    ret -= cnv(u, kernel.T, mode='same', boundary='symm')

    Q = ((1, 0), (0, 1), (-1, 1), (1, -1), (1, 1), (-1, -1), (0, -1), (-1, 0))
    s = np.zeros(u.shape)
    for x, y in Q:
        d = sgn(S(x, y, z) - z)
        s += 1 / norm([x, y]) * (S(-x, -y, d) - d)

    return ret + 0.015*s

def deconvolve(u, kernel, noise_level, gt, baseline=0):
    z = u.copy()
    beta = 0.1
    noise_level = float(noise_level) / 255

    iters = 100
    for i in range(iters):
        subgrad = df(z, kernel, u)
        z = np.clip(z - beta*subgrad, 0, 1)

        progress_bar(i, iters, f'PSNR: {compare_psnr(z, gt)} | Baseline: {baseline}')
        imsave('subgrad.png', subgrad)
        break

    return z

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python main.py input_image kernel output_image noise_level')
        sys.exit(0)

    u = skimage.img_as_float(imread(sys.argv[1], as_gray=True))
    kernel = skimage.img_as_float(imread(sys.argv[2], as_gray=True))
    kernel /= np.sum(kernel)
    print(np.sum(kernel))
    print(u.shape, kernel.shape)
    noise_level = int(sys.argv[4])

    gt = skimage.img_as_float(imread('a.png', as_gray=True))
    #baseline = skimage.img_as_float(imread('data/my_result.png', as_gray=True))

    #baseline = compare_psnr(baseline, gt)

    z = deconvolve(u, kernel, noise_level, gt)

    imsave(sys.argv[3], z)



