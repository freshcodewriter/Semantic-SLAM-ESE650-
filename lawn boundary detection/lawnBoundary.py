import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import skimage.measure
import time

start = time.time()
def lawnBoundary(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #green mask
    lower_green = np.array( [40,40,40], dtype = "uint8")
    upper_green = np.array( [90,255,255], dtype = "uint8")
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    kernel_size = 3
    cv2.imwrite('output_2.jpg', mask_green)

    # #dilation
    kernel = np.ones((kernel_size*2,kernel_size*2),np.uint8)
    dilation_image = cv2.dilate(mask_green, kernel, iterations=1)

    # #morph close
    closing = cv2.morphologyEx(dilation_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('output_5.jpg', closing)

    #remove small blobs
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=4)
    #connectedComponentswithStats yields every separated component with information on each of them, such as size
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    min_size = 1000  #num pixels

    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    slice1Copy = np.uint8(img2)
    _, threshold = cv2.threshold(slice1Copy, 110, 255, cv2.THRESH_BINARY) 
    contours = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    return contours[1]

image = cv2.imread('./image/new_edge.jpg')
image = cv2.resize(image,(240,180))
contour = lawnBoundary(image)
cv2.drawContours(image,contour,-1,(0,255,0),3)
print(time.time()-start)
plt.imshow(image)
cv2.imwrite('output_6.jpg',image)