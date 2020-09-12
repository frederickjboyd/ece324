import matplotlib.pyplot as plt
import numpy as np

# 1
img = plt.imread('pink_lake.png')

# 2
plt.imshow(img)
plt.show()

# 3
img_add = np.add(img, 0.25)
np.clip(img_add, 0, 1, img_add)
plt.imsave('img_add.png', img_add)
plt.imshow(img_add)
plt.show()

# 4
img_chan_0 = np.zeros(img.shape, dtype=img.dtype)  # R
img_chan_1 = np.zeros(img.shape, dtype=img.dtype)  # G
img_chan_2 = np.zeros(img.shape, dtype=img.dtype)  # B
for rowIndex in range(len(img)):
    for pixelIndex in range(len(img[rowIndex])):
        img_chan_0[rowIndex][pixelIndex][0] = img[rowIndex][pixelIndex][0]
        img_chan_1[rowIndex][pixelIndex][1] = img[rowIndex][pixelIndex][1]
        img_chan_2[rowIndex][pixelIndex][2] = img[rowIndex][pixelIndex][2]
plt.imshow(img_chan_0)
plt.show()
plt.imsave('img_chan_0.png', img_chan_0)
plt.imshow(img_chan_1)
plt.show()
plt.imsave('img_chan_1.png', img_chan_1)
plt.imshow(img_chan_2)
plt.show()
plt.imsave('img_chan_2.png', img_chan_2)

# 5
img_gray = np.zeros(img.shape, dtype=img.dtype)
for rowIndex in range(len(img)):
    for pixelIndex in range(len(img[rowIndex])):
        pixel = img[rowIndex][pixelIndex]
        grayPixel = img_gray[rowIndex][pixelIndex]
        grayPixelValue = (0.299 * pixel[0]) + (0.587 * pixel[1]) + (0.144 * pixel[2])
        grayPixel[0] = grayPixelValue
        grayPixel[1] = grayPixelValue
        grayPixel[2] = grayPixelValue
np.clip(img_gray, 0, 1, img_gray)
plt.imshow(img_gray)
plt.show()
plt.imsave('img_gray.png', img_gray)

# 6
cropped_height = int(img.shape[0] / 2)
img_crop = img[:cropped_height]
plt.imshow(img_crop)
plt.show()
plt.imsave('img_crop.png', img_crop)

# 7
img_flip_vert = np.flip(img, 0)
plt.imshow(img_flip_vert)
plt.show()
plt.imsave('img_flip_vert.png', img_flip_vert)
