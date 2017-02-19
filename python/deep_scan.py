import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy
from keras.models import load_model
import math
from scipy.misc import imresize
import json
import base64
import sys
import os

filepath = os.path.dirname(os.path.realpath(__file__))
paper = cv2.imread(filepath + '/input.png')

def abs_thresh(img, orient='x', thresh=(0, 255), ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    else:
        sobel = None
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def dir_thresh(img, thresh=(0.7, 1.3), ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    arc_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # Apply threshold
    binary_output = np.zeros_like(arc_sobel)
    binary_output[(arc_sobel >= thresh[0]) & (arc_sobel <= thresh[1])] = 1
    return binary_output

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

xsize = paper.shape[1]
ysize = paper.shape[0]

left_bottom = (0, 11 / 50 * ysize)
right_bottom = (xsize, 11 / 50 * ysize)
left_top = (0, 0)
right_top = (xsize, 0)
vertices = np.array([[left_bottom, right_bottom, right_top, left_top]], dtype=np.int32)
head = region_of_interest(paper, vertices)

left_bottom = (0, ysize)
right_bottom = (xsize, ysize)
left_top = (0, 11 / 50 * ysize)
right_top = (xsize, 11 / 50 * ysize)
vertices = np.array([[left_bottom, right_bottom, right_top, left_top]], dtype=np.int32)
body = region_of_interest(paper, vertices)

head_vert_binary = abs_thresh(head, thresh=(30, 255), ksize=3)
head_vert_hist = np.sum(head_vert_binary, axis=0)
head_horiz_binary = abs_thresh(head, orient='y', thresh=(30, 255), ksize=3)
head_horiz_hist = np.sum(head_horiz_binary, axis=1)

tmp = []
max_ = max(head_horiz_hist)
for i in range(len(head_horiz_hist)):
    if head_horiz_hist[i] < max_ / 2:
        continue
    tmp.append(i)
head_horiz_lines = []
sum_ = tmp[0]
count = 1
for i in range(1, len(tmp)):
    if tmp[i] - tmp[i - 1] < 3:
        sum_ += tmp[i]
        count += 1
    else:
        head_horiz_lines.append(sum_ / count)
        sum_ = tmp[i]
        count = 1

# vertical lines
tmp = []
max_ = max(head_vert_hist)
for i in range(len(head_vert_hist)):
    if head_vert_hist[i] < max_ / 2:
        continue
    tmp.append(i)
head_vert_lines = []
sum_ = tmp[0]
count = 1
for i in range(1, len(tmp)):
    if tmp[i] - tmp[i - 1] < 3:
        sum_ += tmp[i]
        count += 1
    else:
        head_vert_lines.append(sum_ / count)
        sum_ = tmp[i]
        count = 1
head_vert_lines.append(sum_ / count)

body_vert_binary = abs_thresh(body, thresh=(30, 255), ksize=3)
body_vert_hist = np.sum(body_vert_binary, axis=0)
body_horiz_binary = abs_thresh(body, orient='y', thresh=(30, 255), ksize=3)
body_horiz_hist = np.sum(body_horiz_binary, axis=1)

# find lines in body

# horizontal lines
tmp = []
max_ = max(body_horiz_hist)
for i in range(len(body_horiz_hist)):
    if body_horiz_hist[i] < max_ / 2:
        continue
    tmp.append(i)
body_horiz_lines = []
sum_ = tmp[0]
count = 1
for i in range(1, len(tmp)):
    if tmp[i] - tmp[i - 1] < 3:
        sum_ += tmp[i]
        count += 1
    else:
        body_horiz_lines.append(sum_ / count)
        sum_ = tmp[i]
        count = 1
body_horiz_lines.append(sum_ / count)
body_horiz_lines.pop(0)

# vertical lines
tmp = []
max_ = max(body_vert_hist)
for i in range(len(body_vert_hist)):
    if body_vert_hist[i] < max_ / 2:
        continue
    tmp.append(i)
body_vert_lines = []
sum_ = tmp[0]
count = 1
for i in range(1, len(tmp)):
    if tmp[i] - tmp[i - 1] < 3:
        sum_ += tmp[i]
        count += 1
    else:
        body_vert_lines.append(sum_ / count)
        sum_ = tmp[i]
        count = 1
body_vert_lines.append(sum_ / count)

def preprocess_image(rgb_image, thickness=10):
    res = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    # print(res.shape)
    # res = cv2.resize(res, (28, 28), interpolation=cv2.INTER_AREA)
    # print(res.shape)
    for i in range(res.shape[0]):
        for j in range(thickness):
            res[i][j] = 255
    for i in range(thickness):
        for j in range(res.shape[0]):
            res[i][j] = 255
    for i in range(res.shape[0] - 1, res.shape[0] - thickness, -1):
        for j in range(res.shape[0]):
            res[i][j] = 255
    for i in range(res.shape[0]):
        for j in range(res.shape[0] - 1, res.shape[0] - thickness, -1):
            res[i][j] = 255
#     for i in range(28):
#         for j in range(28):
#             if res[i][j] <= 100:
#                 res[i][j] = 0
    return res

def reshape_images(images):
    res = []
    for i in range(len(images)):
        res.append(images[i].reshape((images[i].shape[0], images[i].shape[1], 1)))
    return np.array(res)

def center(image):
    res = np.zeros(image.shape)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i][j] = 255
    d = {}
    max_x, max_y = -1, -1
    min_x, min_y = float('inf'), float('inf')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 255:
                continue
            min_x = min(min_x, j)
            min_y = min(min_y, i)
            max_x = max(max_x, j)
            max_y = max(max_y, i)
            d[(i, j)] = image[i][j]
    if len(d) < 40:
        return res
    center = (33, 33)
    centroid = (int((max_x + min_x) / 2), int((max_y + min_y) / 2))
#     print(centroid)
    shift = (center[0] - centroid[0], center[1] - centroid[1])
    # print(center)
#     print(shift)
#     print()
    for key in d:
        x = key[0] + shift[1]
        y = key[1] + shift[0]
        if x >= 0 and x < image.shape[0] and y >= 0 and y < image.shape[1]:
            res[x][y] = d[key]
#     plt.figure()
#     plt.imshow(res, cmap='gray')
    return res

def scale_digit(image):
    min_y = float('inf')
    max_y = -1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 255:
                continue
            min_y = min(min_y, i)
            max_y = max(max_y, i)
#     print(max_y)
#     print(min_y)
#     print()
    diff = 50 / (max_y - min_y)
    res = imresize(image, (int(image.shape[0] * diff), int(image.shape[1] * diff)))
    center = (int(res.shape[0] / 2), int(res.shape[1] / 2))
    return res[int(center[0] - image.shape[0] / 2): int(center[0] + image.shape[0] / 2), int(center[1] - image.shape[1] / 2): int(center[1] + image.shape[1] / 2)]

first_name, last_name = [], []

for i in range(0, len(head_horiz_lines) - 1):
    for j in range(1, len(head_vert_lines) - 1):
        side_len = math.ceil(head_vert_lines[j + 1] - head_vert_lines[j])
        region = paper[int(head_horiz_lines[i]): int(head_horiz_lines[i] + side_len), int(head_vert_lines[j]): int(head_vert_lines[j] + side_len)]
        res = center(preprocess_image(region, thickness=8))
        is_zero = True
        for k in range(res.shape[0]):
            for l in range(res.shape[1]):
                if res[k][l] != 255:
                    is_zero = False
                    break
        if is_zero:
            continue
        if i == 0:
            first_name.append(res)
        else:
            last_name.append(res)

for i in range(len(first_name)):
    first_name[i] = scale_digit(first_name[i])
    # plt.figure()
    # plt.imshow(first_name[i], cmap='gray')
for i in range(len(last_name)):
    last_name[i] = scale_digit(last_name[i])
    # plt.figure()
    # plt.imshow(last_name[i], cmap='gray')

first_name, last_name = np.array(first_name), np.array(last_name)
first_name, last_name = reshape_images(first_name), reshape_images(last_name)

model = load_model(filepath + '/c_model.h5')

first_pred = model.predict(first_name)
last_pred = model.predict(last_name)

first_name_val = []
for n in first_pred:
    first_name_val.append(int(np.argmax(n)))

last_name_val = []
for n in last_pred:
    last_name_val.append(int(np.argmax(n)))

answers = []

for i in range(1, len(body_vert_lines), 2):
    for j in range(0, len(body_horiz_lines) - 1):
        side_len = 68
        region = paper[int(body_horiz_lines[j]): int(body_horiz_lines[j] + side_len), int(body_vert_lines[i]): int(body_vert_lines[i] + side_len)]
        res = center(preprocess_image(region, thickness=8))
        is_zero = True
        for k in range(res.shape[0]):
            for l in range(res.shape[1]):
                if res[k][l] != 255:
                    is_zero = False
                    break
        if is_zero:
            continue
        answers.append(res)

for i in range(len(answers)):
    # print(i)
    answers[i] = scale_digit(answers[i])
    # plt.figure()
    # plt.imshow(answers[i], cmap='gray')
answers = np.array(answers)
answers = reshape_images(answers)

answer_pred = model.predict(answers)

answers_val = []
for n in answer_pred:
    answers_val.append(int(np.argmax(n)))

d = {}
d['first_name'] = first_name_val
d['last_name'] = last_name_val
d['answers'] = answers_val

print(d)

with open('data.json', 'w') as fp:
    json.dump(d, fp)
