from os import path
import numpy as np
import sys
sys.path.append('usr/local/lib/python2.7/site-packages')
import cv2
from math import sqrt

class Aindane(object):

    _EPS = 1e-6

    def __init__(self, path_to_img):

        self.img_bgr = cv2.imread(path_to_img)
        if self.img_bgr is None:
            raise Exception("cv2.imread error")

        self.img_gray = cv2.cvtColor(
            self.img_bgr,
            cv2.COLOR_BGR2GRAY
        )

        
  	self.z = None
        self.c = None
        self.p = None

    def _ale(self):

        In = self.img_gray / 255.0

        cdf = cv2.calcHist([self.img_gray], [0], None, [256], [0, 256]).cumsum()
        L = np.searchsorted(cdf, 0.1 * self.img_gray.shape[0] * self.img_gray.shape[1], side='right')
        L_as_array = np.array([L])  # L as array, for np.piecewise
        z_as_array = np.piecewise(L_as_array,
                         [L_as_array <= 50,
                          50 < L_as_array <= 150,
                          L_as_array > 150
                          ],
                         [0, (L-50) / 100.0, 1]
                         )
        z = z_as_array[0]

        self.z = z

        In_prime = 0.5 * (In**(0.75*z+0.25) + (1-In)*0.4*(1-z) + In**(2-z))
        return In_prime

    def _ace(self, In_prime, c=5):
                
   	img_freq = np.fft.fft2(self.img_gray)
        img_freq_shift = np.fft.fftshift(img_freq)

        sigma = sqrt(c**2 / 2)
        _gaussian_x = cv2.getGaussianKernel(
            int(round(sigma*3)),  
            int(round(sigma))
        )
        gaussian = (_gaussian_x * _gaussian_x.T) / np.sum(_gaussian_x * _gaussian_x.T)
        gaussian_freq_shift = np.fft.fftshift(
            np.fft.fft2(gaussian, self.img_gray.shape)
        )
       
        image_fm = img_freq_shift * gaussian_freq_shift
        I_conv = np.real(np.fft.ifft2(np.fft.ifftshift(image_fm)))

        sigma_I = np.array([np.std(self.img_gray)])
        P = np.piecewise(sigma_I,
                         [sigma_I <= 3,
                          3 < sigma_I < 10,
                          sigma_I >= 10
                          ],
                         [3, 1.0 * (27 - 2 * sigma_I) / 7, 1]
                         )[0]

        self.c = c
        self.p = P

        E = ((I_conv + self._EPS) / (self.img_gray + self._EPS)) ** P
        S = 255 * np.power(In_prime, E)
        return S

    def _color_restoration(self, S, lambdaa=[1, 1, 1]):
        S_restore = np.zeros(self.img_bgr.shape)
        for j in xrange(3):  
            S_restore[..., j] = S * (1.0 * self.img_bgr[..., j] / (self.img_gray + self._EPS)) * lambdaa[j]

        return np.clip(S_restore, 0, 255).astype('uint8')

    def aindane(self):
        In_prime = self._ale()
        S = self._ace(In_prime, c=240)
        return self._color_restoration(S, lambdaa=[1, 1, 1])

    def plot(self):
        print ("z, c, p: {}, {}, {}".format(self.z, self.c, self.p))
        from matplotlib import pyplot as plt
	img1 = cv2.cvtColor(np.hstack([self.img_bgr, self.aindane()]), cv2.COLOR_BGR2RGB)
	img2 = cv2.cvtColor(self.aindane(), cv2.COLOR_BGR2RGB)
        plt.imshow(img1)
        plt.show()
	cv2.imwrite('images/output/indane.png',img2)


#################################################
def main():
    IMG_DIR = './images/input/'
    IMG_NAME = 'sign1.png'

    aindane = Aindane(path.join(IMG_DIR, IMG_NAME))
    aindane.plot()

if __name__ == '__main__':
    main()
