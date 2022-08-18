import math
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

OFFSET = 50
IMG_SIZE = (500, 500)

WIDTH = 20
DISTANCE = 5
STEP = 25

img = cv.imread('1.png', 0)
src = cv.resize(img, IMG_SIZE)

for ang in range(15):
    rows,cols = src.shape
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),ang,1)
    img = cv.warpAffine(src,M,(cols,rows))

    img = img[50:-50, 50:-50]

    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 7)

    deltas = []
    for u in range(30):
        i = 5 * u
        horizontal_crops_0 = 255 - img[i:i + WIDTH, :]
        horizontal_crops_1 = 255 - img[i + WIDTH + DISTANCE:i + 2 * WIDTH + DISTANCE, :]

        density_0 = []
        density_1 = []
        x = []
        for k in range(400 // STEP):
            density_0.append(np.mean(horizontal_crops_0[:, k * STEP:(k + 1) * STEP - 1]))
            density_1.append(np.mean(horizontal_crops_1[:, k * STEP:(k + 1) * STEP - 1]))
            x.append(k)

        for j in x:
            if density_0[j] > 20:
                x_0 = j
                break
        for j in x:
            if density_1[j] > 20:
                x_1 = j
                break

        delta = math.fabs(x_0 - x_1) * STEP
        deltas.append(delta)

    deltas = np.array(deltas)
    delta = np.mean(deltas)

    hh = 200 - 400 * (WIDTH + DISTANCE) / delta
    print(hh)

    img_final = cv.line(img, (0, 200), (400, int(hh)), (0, 0, 0), 2)

    cv.imshow('final', img_final)
    cv.waitKey(0)