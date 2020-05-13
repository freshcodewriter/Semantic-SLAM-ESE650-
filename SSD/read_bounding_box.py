import numpy as np

import tensorflow as tf
from google.protobuf import text_format


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

box = np.load('box.npy')
label_class = np.load("class.npy")
label_score = np.load("score.npy")



print(box[0])
print(label_class[0])
print(label_score[0])

print(box.shape)
print(label_class.shape)
print(label_score.shape)

frame_no = 450
cur_label = label_score[frame_no]
cur_label[cur_label<0.6] = 0
effect_index = np.count_nonzero(cur_label)
start_idx = frame_no * 20
end_idx = start_idx + effect_index

image = cv2.imread('./frame/frame450.jpg')

for i in range(effect_index):
    if label_class[frame_no, i] == 1:
        cv2.rectangle(image,
                    (box[start_idx+i, 0], box[start_idx+i, 1]),
                    (box[start_idx+i, 2], box[start_idx+i, 3]),
                    (0, 255, 0), thickness=3)
    elif label_class[frame_no, i] == 2:
        cv2.rectangle(image,
                    (box[start_idx+i, 0], box[start_idx+i, 1]),
                    (box[start_idx+i, 2], box[start_idx+i, 3]),
                    (255, 0, 0), thickness=3)
    elif label_class[frame_no, i] == 3:
        cv2.rectangle(image,
                    (box[start_idx+i, 0], box[start_idx+i, 1]),
                    (box[start_idx+i, 2], box[start_idx+i, 3]),
                    (0, 0, 255), thickness=3)
    elif label_class[frame_no, i] == 4:
        cv2.rectangle(image,
                    (box[start_idx+i, 0], box[start_idx+i, 1]),
                    (box[start_idx+i, 2], box[start_idx+i, 3]),
                    (255, 255, 0), thickness=3)

plt.imshow(image)
plt.show()





# Saving the image
# cv2.imwrite("output.png",image)

#Display
# cv2.imshow("output", image)
#
# # Create figure and axes
# fig,ax = plt.subplots(1)
#
# # Display the image
# ax.imshow(im)
#
# # Create a Rectangle patch
# rect = patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')
#
# # Add the patch to the Axes
# ax.add_patch(rect)



#
# tf.train.write_graph(gdef, '/tmp', 'myfile.pb', as_text=False)