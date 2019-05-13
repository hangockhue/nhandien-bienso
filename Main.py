import cv2
import os

import FindChars
import FindPlates
import Plate

Scalar_black = (0.0, 0.0, 0.0)
Scalar_white = (255.0, 255.0, 255.0)
Scalar_yellow = (0.0, 255.0, 255.0)
Scalar_green = (0.0, 255.0, 0.0)
Scalar_red = (0.0, 0.0, 255.0)


def main():

    KNN_trainning = FindChars.load_KNN_data_and_train_KNN()

    if KNN_trainning == False:

        print("Lỗi Trainning")
        return

    img_scene = cv2.imread("4.jpg")

    if img_scene is None:
        print(" Không có image ")
        os.system("pause")
        return

    list_of_possible_plate = FindPlates.find_plates_in_scene(img_scene)

    list_of_possible_plate = FindChars.find_chars_in_plates(list_of_possible_plate)

    cv2.imshow("Image", img_scene)

    if len(list_of_possible_plate) == 0:
        print("Không tìm thấy được biển nào")
    else:
        list_of_possible_plate.sort(key=lambda possible_plate: len(possible_plate.str_chars), reverse=True)

        lic_plate = list_of_possible_plate[0]

        cv2.imshow("Image Plate", lic_plate.img_plate)
        cv2.imshow("Image Thresh", lic_plate.img_thresh)

        if len(lic_plate.str_chars) == 0:
            print(" Không tìm thấy ký tự trong biển ")
            return
        draw_red_of_angle_around_plate(img_scene, lic_plate)
        print("Ký tự ở biển là = " + lic_plate.str_chars + "\n")
        print("-----------------------------------------------")

        write_license_on_image(img_scene, lic_plate)

        cv2.imshow("Image", img_scene)

        cv2.imwrite("ImageResults.png", img_scene)

    cv2.waitKey(0)

    return


def draw_red_of_angle_around_plate(img_scene, lic_plate):  # Vẽ khung cho biển tìm được

    p2f_of_point = cv2.boxPoints(lic_plate.rr_location_of_plate_in_scene)

    cv2.line(img_scene, tuple(p2f_of_point[0]), tuple(p2f_of_point[1]),
             Scalar_red, 2)
    cv2.line(img_scene, tuple(p2f_of_point[1]), tuple(p2f_of_point[2]),
             Scalar_red, 2)
    cv2.line(img_scene, tuple(p2f_of_point[2]), tuple(p2f_of_point[3]),
             Scalar_red, 2)
    cv2.line(img_scene, tuple(p2f_of_point[3]), tuple(p2f_of_point[0]),
             Scalar_red, 2)


def write_license_on_image(img_scene, lic_plate):  # Viết ký tự vào trong hình

    scene_height, scene_width, scene_numchannels = img_scene.shape
    plate_height, plate_width, plate_numchannels = lic_plate.img_plate.shape

    int_font_face = cv2.FONT_HERSHEY_SIMPLEX
    flt_font_scale = float(plate_height) / 30.0
    int_font_thickness = int(round(flt_font_scale * 1.5))

    text_size, base_line = cv2.getTextSize(lic_plate.str_chars, int_font_face,
                                           flt_font_scale, int_font_thickness)
    ((int_plate_center_x, int_plate_center_y), (int_plate_width, int_plate_height), flt_correction_angle_in_deg) =\
        lic_plate.rr_location_of_plate_in_scene

    int_plate_center_x = int(int_plate_center_x)
    int_plate_center_y = int(int_plate_center_y)

    pt_center_of_text_area_x = int(int_plate_center_x)

    if int_plate_center_y < (scene_height * 0.75):
        pt_center_of_text_area_y = int(round(int_plate_center_y)) + int(round(plate_height * 1.6))
    else:
        pt_center_of_text_area_y = int(round(int_plate_center_y)) - int(round(plate_height * 1.6))

    text_size_width, text_size_height = text_size

    pt_lower_left_text_origin_x = int(pt_center_of_text_area_x - (text_size_width / 2))
    pt_lower_left_text_origin_y = int(pt_center_of_text_area_y + (text_size_height / 2))

    cv2.putText(img_scene, lic_plate.str_chars, (pt_lower_left_text_origin_x, pt_lower_left_text_origin_y),
                int_font_face, flt_font_scale, Scalar_yellow, int_font_thickness)


if __name__ == "__main__":
    main()
