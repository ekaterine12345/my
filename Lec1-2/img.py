import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import cv2
import matplotlib.pyplot as plt
data =cv2.imread('1.png')
def color_correction(img, ccm):
    '''
    Input:
        img: H*W*3 numpy array, input image
        ccm: 3*3 numpy array, color correction matrix
    Output:
        output: H*W*3 numpy array, output image after color correction
    '''

    img2 = img.reshape((img.shape[0] * img.shape[1], 3))
    output = np.matmul(img2, ccm)
    return output.reshape(img.shape).astype(img.dtype)


ccm = np.array([[1.0234, -0.2969, -0.2266],
                [-0.5625,  1.6328, -0.0469],
                [-0.0703,  0.2188,  0.6406]])
output = color_correction(data, ccm)
plt.subplot(1, 2, 1)
plt.imshow(data)
plt.subplot(1, 2, 2)
plt.imshow(output)
plt.show()
