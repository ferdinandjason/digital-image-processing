import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def filter_image(image, filter):
    m, n = image.shape

    iimage = np.zeros(shape=(m+2, n+2))
    iimage[1:m+1, 1:n+1] = image

    new_image = np.zeros_like(image)

    for i in range(1,m+1):
        for j in range(1,n+1):
            new_image[i-1,j-1] = int(np.vdot(iimage[i-1:i+2,j-1:j+2],filter)/9)

    return image, new_image


img = np.uint8(mpimg.imread('image.png')*255.0)
if len(img.shape) == 3 :
    img = np.uint8((0.2126* img[:,:,0]) +
      	  np.uint8(0.7152 * img[:,:,1]) +
    	  np.uint8(0.0722 * img[:,:,2]))

image , new_image = filter_image(img, np.array([[1,1,1],[1,1,1],[1,1,1]]))
new_image , new_image_2 = filter_image(new_image, np.array([[1,1,1],[1,1,1],[1,1,1]]))
new_image_2, new_image_3 = filter_image(new_image_2, np.array([[1,1,1],[1,1,1],[1,1,1]]))
new_image_3, new_image_4 = filter_image(new_image_3, np.array([[1,1,1],[1,1,1],[1,1,1]]))
new_image_4, new_image_5 = filter_image(new_image_4, np.array([[1,1,1],[1,1,1],[1,1,1]]))

fig = plt.figure(figsize=(17,13))

fig.add_subplot(231)
plt.imshow(image)
plt.title('original image')
plt.set_cmap('gray')

fig.add_subplot(232)
plt.imshow(new_image)
plt.title('blur image x1')
plt.set_cmap('gray')

fig.add_subplot(233)
plt.imshow(new_image_2)
plt.title('blur image x2')
plt.set_cmap('gray')

fig.add_subplot(234)
plt.imshow(new_image_3)
plt.title('blur image x3')
plt.set_cmap('gray')

fig.add_subplot(235)
plt.imshow(new_image_4)
plt.title('blur image x4')
plt.set_cmap('gray')

fig.add_subplot(236)
plt.imshow(new_image_5)

plt.title('blur image x5')
plt.set_cmap('gray')

plt.show()