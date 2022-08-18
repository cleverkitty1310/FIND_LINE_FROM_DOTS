from cgi import test
import math
import os
from tkinter import OFF
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

OFFSET = 50
IMG_SIZE = 500

WIDTH = 7
CROPPED_IMG_WIDTH = IMG_SIZE - 2 * OFFSET
MINIMUM_POINT_COUNT = 30
ANG_DIVISION_COUNT = 100
ALPHA = 1.2
original = cv.imread('1.png')
img = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
src = cv.resize(img, (IMG_SIZE, IMG_SIZE))


for ang in range(10):
    # This is for TEST ONLY
    rows,cols = src.shape
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), ang, 1)
    img = cv.warpAffine(src, M, (cols, rows))
    # endregion

    img = img[OFFSET:-OFFSET, OFFSET:-OFFSET]

    img = cv.adaptiveThreshold(img, np.max(img), cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 19, 5)

    plt.imshow(img)
    plt.show()

    print('Processing points...')
    points_count_dictionary = {}

    for i in range(CROPPED_IMG_WIDTH):
        for j in range(CROPPED_IMG_WIDTH):
            if img[i][j] == 0:
                key = i // WIDTH * 100 + j // WIDTH
                if points_count_dictionary.get(key) != None:
                    points_count_dictionary[key] += 1
                else:
                    points_count_dictionary[key] = 1
    
    flag_dictionary = {}
    for i in points_count_dictionary:
        if flag_dictionary.get(i) != None:
            continue
        row_number = i // 100
        col_number = i % 100
        max_point = points_count_dictionary[i]
        max_key = i
        count_sum = 0
        for k in range(row_number - 1, row_number + 2):
            for l in range(col_number - 1, col_number + 2):
                if points_count_dictionary.get(k * 100 + l) != None:
                    count_sum += points_count_dictionary[k * 100 + l]
                    if i == k * 100 + l:
                        continue
                    if max_point <= points_count_dictionary[k * 100 + l]:
                        max_point = points_count_dictionary[k * 100 + l]
                        flag_dictionary[max_key] = False
                        max_key = k * 100 + l
                    elif max_point > points_count_dictionary[k * 100 + l]:
                        flag_dictionary[k * 100 + l] = False
                        
        if count_sum < MINIMUM_POINT_COUNT:
            flag_dictionary[max_key] = False

    test_image = img.copy()
    for i in points_count_dictionary:
        if flag_dictionary.get(i) == None:
            row = i // 100
            col = i % 100
            test_image = cv.rectangle(test_image, (col * WIDTH, row * WIDTH), (col * WIDTH + WIDTH, row * WIDTH + WIDTH), 0, 3)
    plt.imshow(test_image)
    plt.show()

    rows = CROPPED_IMG_WIDTH // WIDTH
    center_point = None
    center_length = 0
    for i in range(rows // 2 - 5, rows // 2 + 6):
        for j in range(rows // 2 - 5, rows // 2 + 6):
            if points_count_dictionary.get(i * 100 + j) != None and flag_dictionary.get(i * 100 + j) == None:
                length = (i - rows // 2)**2 + (j - rows // 2)**2
                if center_point == None:
                    center_length = length
                    center_point = i * 100 + j
                else:
                    if center_length > length:
                        center_length = length
                        center_point = i * 100 + j

    n_list = np.zeros(ANG_DIVISION_COUNT)
    average_list = np.zeros(ANG_DIVISION_COUNT)

    if center_point == None:
        print('Cannot find center point.')
    else:
        center_row = center_point // 100
        center_col = center_point % 100
        test_image = img.copy()
        test_image = cv.rectangle(test_image, (center_col * WIDTH, center_row * WIDTH), (center_col * WIDTH + WIDTH, center_row * WIDTH + WIDTH), 0, 2)
        plt.imshow(test_image)
        plt.show()
        for i in points_count_dictionary:
            if flag_dictionary.get(i) == None:
                i_row = i // 100
                i_col = i % 100
                length = math.sqrt((i_row - center_row)**2 + (i_col - center_col)**2)
                if i_row < center_row + 2 and i_col > center_col and length < (CROPPED_IMG_WIDTH // WIDTH - center_col):
                    i_ang = math.acos((i_col - center_col) / length)
                    for k in range(ANG_DIVISION_COUNT):
                        
                        if math.fabs(math.sin(k * math.pi / 2 / ANG_DIVISION_COUNT - i_ang) * length) <= ALPHA:
                            average_list[k] = average_list[k] * n_list[k] + i_ang
                            n_list[k] += 1
                            average_list[k] /= n_list[k]
                    
    print(n_list)
    print(average_list)
    print(np.argmax(n_list))
    rotation = average_list[np.argmax(n_list)] * 180 / math.pi
    
    rows, cols = src.shape
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -rotation, 1)
    result_image = cv.warpAffine(src, M, (cols, rows))
    plt.imshow(result_image)
    plt.show()

    # print(points_count_dictionary.__len__())
    # cv.waitKey(0)