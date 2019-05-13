import os
import cv2
import numpy as np
import math
import random

import Main
import Preprocess
import Char

kNearest = cv2.ml.KNearest_create()

Min_pixel_width = 2
Min_pixel_height = 8

Min_aspect_ratio = 0.25
Max_aspect_ration = 1.0

Min_pixel_area = 80

Min_diag_size_multiple_away = 0.3
Max_diag_size_multiple_away = 5.0

Max_change_in_area = 0.5

Max_change_in_width = 0.8
Max_change_in_height = 0.2

Max_angle_between_chars = 12.0

Min_number_of_maching_chars = 3

Resized_char_image_width = 20
Resized_char_image_height = 30

Min_contour_area = 100


def load_KNN_data_and_train_KNN():

    try:
        load_classification = np.loadtxt("classifications.txt", np.float32)
        print(load_classification)
    except:
        print('Lỗi , không thể mở file classifications.txt')
        os.system('pause')
        return False

    try:
        load_flattened_image = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print("Lỗi, không thể mở file flattened_image.txt", np.float32)
        os.system("pause")
        return False

    load_classification = load_classification.reshape((load_classification.size), 1)

    kNearest.setDefaultK(1)

    kNearest.train(load_flattened_image, cv2.ml.ROW_SAMPLE, load_classification)

    return True     # Train thành công


def find_chars_in_plates(list_of_possible_plates):
    int_plate_counter = 0
    img_contours = None
    contours = []

    if len(list_of_possible_plates) == 0:
        return list_of_possible_plates

    for possible_plate in list_of_possible_plates:

        possible_plate.img_grayscale, possible_plate.img_thresh = Preprocess.preprocess(possible_plate.img_plate)

        possible_plate.img_thresh = cv2.resize(possible_plate.img_thresh, (0,0), fx=1.6, fy=1.6)

        thresh_shold_value, possible_plate.img_thresh = cv2.threshold(possible_plate.img_thresh, 0.0, 255.0,
                                                                      cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        list_of_possible_chars_in_plates = find_possible_chars_in_plate(possible_plate.img_grayscale, possible_plate.img_thresh)

        list_of_lists_of_matching_chars_in_plate = find_list_of_lists_of_matching_chars(list_of_possible_chars_in_plates)

        if len(list_of_lists_of_matching_chars_in_plate) == 0:

            possible_plate.str_chars = ""

            continue

        for i in range(0, len(list_of_lists_of_matching_chars_in_plate)):
            list_of_lists_of_matching_chars_in_plate[i].sort(key=lambda matching_char: matching_char.int_center_x)
            list_of_lists_of_matching_chars_in_plate[i] =\
                remove_inner_overlapping_chars(list_of_lists_of_matching_chars_in_plate[i])

        int_len_of_longest_list_of_chars = 0
        int_index_of_longest_list_of_chars = 0

        for i in range(0, len(list_of_lists_of_matching_chars_in_plate)):
            if len(list_of_lists_of_matching_chars_in_plate[i]) > int_len_of_longest_list_of_chars:
                int_len_of_longest_list_of_chars = len(list_of_lists_of_matching_chars_in_plate[i])
                int_index_of_longest_list_of_chars = i

        longest_list_of_matching_chars_in_plate =\
            list_of_lists_of_matching_chars_in_plate[int_index_of_longest_list_of_chars]

        possible_plate.str_chars = recognize_chars_in_plate(possible_plate.img_thresh,
                                                            longest_list_of_matching_chars_in_plate)

    return list_of_possible_plates


def find_possible_chars_in_plate(img_grayscale, img_thresh):  # Tìm tất cả các contours có trong bức hình

    list_of_possible_chars = []
    contours = []
    img_thresh_copy = img_thresh.copy()
    # Tìm tất cả các contour trong hình ảnh img_thresh
    contours, npa_hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        possible_char = Char.Char(contour)

        if check_if_possible_char(possible_char):
            list_of_possible_chars.append(possible_char)
        # end if
    # end for
    return list_of_possible_chars


def check_if_possible_char(possible_char):  # Kiểm tra mỗi countours tìm được

    if (possible_char.int_bounding_of_area > Min_pixel_area and
        possible_char.int_bounding_of_width > Min_pixel_width and
        possible_char.int_bounding_of_height > Min_pixel_height and
            Min_aspect_ratio < possible_char.flt_aspect_ratio < Max_aspect_ration):

        return True
    else:
        return False


def find_list_of_lists_of_matching_chars(list_of_possible_chars):  # Gom tất cả cả contours gần nhau lại thành một list

    list_of_lists_of_matching_chars = []

    for possible_char in list_of_possible_chars:
        list_of_matching_chars = find_list_of_matching_chars(possible_char, list_of_possible_chars)

        list_of_matching_chars.append(possible_char)

        if len(list_of_matching_chars) < Min_number_of_maching_chars:
            continue

        # end if

        list_of_lists_of_matching_chars.append(list_of_matching_chars)
        list_of_possible_chars_with_current_matches_removed = list(set(list_of_possible_chars) - set(list_of_matching_chars))

        recursive_list_of_lists_of_matching_chars = \
            find_list_of_lists_of_matching_chars(list_of_possible_chars_with_current_matches_removed)

        for recursive_list_of_matching_chars in recursive_list_of_lists_of_matching_chars:
            list_of_lists_of_matching_chars.append(recursive_list_of_matching_chars)
        break

    return list_of_lists_of_matching_chars


def find_list_of_matching_chars(possible_char, list_of_chars):  # Tìm các ký tự ở gần nhau
    list_of_matching_chars = []

    for possible_matching_char in list_of_chars:
        if possible_matching_char == possible_char:
            continue
        flt_distance_between_chars = distance_between_chars(possible_char, possible_matching_char)
        flt_angle_between_chars = angle_between_chars(possible_char, possible_matching_char)

        flt_change_in_area = float(abs(possible_matching_char.int_bounding_of_area
                                       - possible_char.int_bounding_of_area)) / float(possible_char.int_bounding_of_area)
        flt_change_in_width = float(abs(possible_matching_char.int_bounding_of_width
                                        - possible_char.int_bounding_of_width)) / float(possible_char.int_bounding_of_width)
        flt_change_in_height = float(abs(possible_matching_char.int_bounding_of_height
                                         - possible_char.int_bounding_of_height)) / float(possible_char.int_bounding_of_height)

        if (flt_distance_between_chars < (possible_char.flt_diagonal_size * Max_diag_size_multiple_away)
                and flt_angle_between_chars < Max_angle_between_chars
                and flt_change_in_area < Max_change_in_area
                and flt_change_in_width < Max_change_in_width
                and flt_change_in_height < Max_change_in_height):

            list_of_matching_chars.append(possible_matching_char)
    print(len(list_of_matching_chars))
    return list_of_matching_chars


def distance_between_chars(first_char, second_char):  # Tính khoảng cách giữa 2 ký tự

    int_x = abs(first_char.int_center_x - second_char.int_center_x)
    int_y = abs(first_char.int_center_y - second_char.int_center_y)

    return math.sqrt((int_x ** 2) + (int_y ** 2))


def angle_between_chars(first_char, second_char):  # Tìm góc giữa 2 ký tự tính theo độ

    flt_adj = float(abs(first_char.int_center_x - second_char.int_center_x))
    flt_opp = float(abs(first_char.int_center_y - second_char.int_center_y))

    if flt_adj != 0.0:
        flt_angle_in_rad = math.atan(flt_opp / flt_adj)  # Tìm góc khi có tan() 
    else:
        flt_angle_in_rad = 1.5708  # Đường thằng
    # end if

    flt_angle_in_deg = flt_angle_in_rad * (180 / math.pi)
    return flt_angle_in_deg


def remove_inner_overlapping_chars(list_of_matching_chars):  # Loại bỏ những ký tự chồng chéo về kích thước

    list_of_matching_chars_with_inner_char_removed = list(list_of_matching_chars)

    for current_char in list_of_matching_chars:
        for other_char in list_of_matching_chars:
            if current_char != other_char:
                if distance_between_chars(current_char, other_char) <\
                        (current_char.flt_diagonal_size * Min_diag_size_multiple_away):
                    if current_char.int_bounding_of_area < other_char.int_bounding_of_area:
                        if current_char in list_of_matching_chars_with_inner_char_removed:
                            list_of_matching_chars_with_inner_char_removed.remove(current_char)
                    else:
                        if other_char in list_of_matching_chars_with_inner_char_removed:
                            list_of_matching_chars_with_inner_char_removed.remove(other_char)
    return list_of_matching_chars_with_inner_char_removed


def recognize_chars_in_plate(img_thresh, list_of_matching_chars):  # Nhận diện ký tự trong biển

    str_chars = ""

    height, width = img_thresh.shape

    img_thresh_color = np.zeros((height, width, 3), np.uint8)

    list_of_matching_chars.sort(key=lambda matching_char: matching_char.int_center_x)

    cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR, img_thresh_color)

    for current_char in list_of_matching_chars:
        pt1 = (current_char.int_bounding_of_x, current_char.int_bounding_of_y)
        pt2 = ((current_char.int_bounding_of_x + current_char.int_bounding_of_width),
               (current_char.int_bounding_of_y + current_char.int_bounding_of_height))

        cv2.rectangle(img_thresh_color, pt1, pt2, Main.Scalar_green, 2)

        img_roi = img_thresh[current_char.int_bounding_of_y:
                             current_char.int_bounding_of_y + current_char.int_bounding_of_height,
                             current_char.int_bounding_of_x:
                             current_char.int_bounding_of_x + current_char.int_bounding_of_width]
        img_roi_resized = cv2.resize(img_roi, (Resized_char_image_width, Resized_char_image_height))

        npa_roi_resized = img_roi_resized.reshape((1, Resized_char_image_width * Resized_char_image_height))

        npa_roi_resized = np.float32(npa_roi_resized)
        retval, npa_results, neigh_nesp, dists = kNearest.findNearest(npa_roi_resized, k=1)

        str_current_char = str(chr(int(npa_results[0][0])))

        str_chars = str_chars + str_current_char

    return str_chars
