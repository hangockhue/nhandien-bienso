import cv2
import numpy as np
import math


# Các biến

Gaussian_smooth_filter_size = (5, 5)
Adaptive_thresh_block_size = 19
Adaptive_thresh_weight = 9


def preprocess(img):

    img_grayscale = extract_value(img)

    img_maxcontrast_grayscale = maximize_contrast(img_grayscale)

    height, width = img_grayscale.shape

    img_blurred = np.zeros((height, width), np.uint8)

    img_blurred = cv2.GaussianBlur(img_maxcontrast_grayscale, Gaussian_smooth_filter_size, 0)

    img_thresh = cv2.adaptiveThreshold(img_blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV
                                       , Adaptive_thresh_block_size, Adaptive_thresh_weight)

    return img_grayscale, img_thresh


def extract_value(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # chuyển hình ảnh sang kênh màu HSV

    img_hue, img_saturation, img_value = cv2.split(img_hsv)

    return img_value


def maximize_contrast(img_grayscale):

    height, width = img_grayscale.shape

    img_blackhat = np.zeros((height, width, 1), np.uint8)

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_tophat = cv2.morphologyEx(img_grayscale, cv2.MORPH_TOPHAT, structuring_element)

    img_grayscale_plus_tophat = cv2.add(img_grayscale, img_tophat)
    img_grayscale_plus_tophat_minus_blackhat = cv2.subtract(img_grayscale_plus_tophat, img_blackhat)

    return img_grayscale_plus_tophat_minus_blackhat
