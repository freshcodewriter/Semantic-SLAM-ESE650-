import numpy as np
import cv2

class Stitcher:
    def __init__(self, right_img, left_img):
        self.right_img = right_img
        self.left_img = left_img
    def stitch(self):
        img1 = cv2.imread(self.left_img)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(self.right_img)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img3 = np.zeros((img1.shape[0], img1.shape[1] + 15))
        img3[:, :490] = img1
        img3[:, 490:490 + 15] = img2[:, 490 - 15:490]
        cv2.imwrite("./result/res.jpg", img3)
        return img3

s = Stitcher('./img/right1.jpg', './img/left1.jpg')
img3 = s.stitch()

