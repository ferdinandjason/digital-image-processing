import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def histogram(image):
    m, n = image.shape
    histo = [0.0] * 256
    for i in range(m):
        for j in range(n):
            histo[image[i,j]]+=1
    return np.array(histo)/(m*n)

def cdf(histogram):
    return np.array([sum(histogram[:i+1]) for i in range(len(histogram))])

def histogram_equalization(image):
    histo = histogram(image)
    histo_cdf = cdf(histo)

    histo_equalize = np.uint8(255 * histo_cdf)

    m, n = image.shape
    new_image = np.zeros_like(image)

    for i in range(m):
        for j in range(n):
            new_image[i,j] = histo_equalize[image[i,j]]
    
    histo_new = histogram(new_image)

    return new_image, histo, histo_new


img = np.uint8(mpimg.imread('image.png')*255.0)
if len(img.shape) == 3 :
    img = np.uint8((0.2126* img[:,:,0]) +
      	  np.uint8(0.7152 * img[:,:,1]) +
    	  np.uint8(0.0722 * img[:,:,2]))

new_img , h , new_h = histogram_equalization(img)

fig = plt.figure(figsize=(9,13))

fig.add_subplot(221)
plt.imshow(img)
plt.title('original image')
plt.set_cmap('gray')

fig.add_subplot(222)
plt.imshow(new_img)
plt.title('hist. equalized image')
plt.set_cmap('gray')

fig.add_subplot(223)
plt.plot(h)
plt.title('Original histogram')

fig.add_subplot(224)
plt.plot(new_h)
plt.title('New histogram')

plt.show()
