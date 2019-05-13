import os
import cv2
import numpy as np
import math

import Preprocess
import Char
import Plate
import FindChars


Plate_width_padding_factor = 1.3
Plate_height_padding_factor = 1.5


def find_plates_in_scene(img_scene):  # Danh sách tất cả các biển

    list_of_possible_plates = []

    height, width, num_channels = img_scene.shape

    img_grayscale_scene = np.zeros((height, width, 1), np.uint8)
    img_thresh_scene = np.zeros((height, width, 1), np.uint8)
    img_contours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    img_grayscale_scene, img_thresh_scene = Preprocess.preprocess(img_scene)

    list_of_possible_chars_in_scene = find_possible_chars_in_scene(img_thresh_scene)

    list_of_lists_of_matching_chars_in_scene = \
        FindChars.find_list_of_lists_of_matching_chars(list_of_possible_chars_in_scene)

    for list_of_matching_chars in list_of_lists_of_matching_chars_in_scene:
        possible_plate = extract_plate(img_scene, list_of_matching_chars)

        if possible_plate.img_plate is not None:
            list_of_possible_plates.append(possible_plate)

    print("\n" + str(len(list_of_possible_plates)) + " bien tim duoc")

    return list_of_possible_plates


def find_possible_chars_in_scene(img_thresh):  # Tìm những ký tự trong hình ảnh được đưa vào

    list_of_possible_chars = []

    int_count_of_possible_chars = 0

    img_thresh_copy = img_thresh.copy()

    contours, npa_hierachy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)
    height, width = img_thresh.shape
    img_contours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):

        possible_char = Char.Char(contours[i])

        if FindChars.check_if_possible_char(possible_char):
            int_count_of_possible_chars = int_count_of_possible_chars + 1
            list_of_possible_chars.append(possible_char)
    print(int_count_of_possible_chars)
    return list_of_possible_chars


def extract_plate(img, list_of_matching_chars):  # Đống khung biển

    possible_plate = Plate.Plate()

    list_of_matching_chars.sort(key=lambda matching_char: matching_char.int_center_x)

    flt_plate_center_x = (list_of_matching_chars[0].int_center_x +
                          list_of_matching_chars[len(list_of_matching_chars) - 1].int_center_x) / 2.0
    flt_plate_center_y = (list_of_matching_chars[0].int_center_y +
                          list_of_matching_chars[len(list_of_matching_chars) - 1].int_center_y) / 2.0

    pt_plate_center = flt_plate_center_x, flt_plate_center_y

    int_plate_width = int((list_of_matching_chars[len(list_of_matching_chars) - 1].int_bounding_of_x +
                           list_of_matching_chars[len(list_of_matching_chars) - 1].int_bounding_of_width -
                           list_of_matching_chars[0].int_bounding_of_x) *
                          Plate_width_padding_factor)
    int_total_of_char_heights = 0

    for matching_char in list_of_matching_chars:
        int_total_of_char_heights = int_total_of_char_heights + matching_char.int_bounding_of_height
    # end for

    flt_average_char_height = int_total_of_char_heights / len(list_of_matching_chars)

    int_plate_height = int(flt_average_char_height * Plate_height_padding_factor)

    flt_opposite = (list_of_matching_chars[len(list_of_matching_chars) - 1].int_center_y -
                    list_of_matching_chars[0].int_center_y)
    flt_hypotenuse = FindChars.distance_between_chars(list_of_matching_chars[0],
                                                      list_of_matching_chars[len(list_of_matching_chars) - 1])
    flt_correction_angle_in_rad = math.asin(flt_opposite / flt_hypotenuse)
    flt_correction_angle_in_deg = flt_correction_angle_in_rad * (180.0 / math.pi)

    possible_plate.rr_location_of_plate_in_scene = (tuple(pt_plate_center),
                                                    (int_plate_width, int_plate_height),
                                                    flt_correction_angle_in_deg)

    rotation_matrix = cv2.getRotationMatrix2D(tuple(pt_plate_center), flt_correction_angle_in_deg, 1.0)

    height, width, num_channels = img.shape

    img_rotated = cv2.warpAffine(img, rotation_matrix, (width, height))

    img_cropped = cv2.getRectSubPix(img_rotated, (int_plate_width, int_plate_height),
                                    tuple(pt_plate_center))
    possible_plate.img_plate = img_cropped

    return possible_plate
