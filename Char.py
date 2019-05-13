import cv2
import math


class Char:

    def __init__(self, _contour):
        self.contour = _contour

        self.bounding_rect = cv2.boundingRect(self.contour)

        [int_x, int_y, int_width, int_height] = self.bounding_rect

        self.int_bounding_of_x = int_x
        self.int_bounding_of_y = int_y
        self.int_bounding_of_width = int_width
        self.int_bounding_of_height = int_height

        self.int_bounding_of_area = (self.int_bounding_of_width * self.int_bounding_of_height)

        self.int_center_x = (self.int_bounding_of_x + self.int_bounding_of_x + self.int_bounding_of_width) / 2
        self.int_center_y = (self.int_bounding_of_y + self.int_bounding_of_y + self.int_bounding_of_height) / 2

        self.flt_diagonal_size = math.sqrt((self.int_bounding_of_width ** 2)
                                           + (self.int_bounding_of_height ** 2))
        self.flt_aspect_ratio = float(self.int_bounding_of_width) / float(self.int_bounding_of_height)

    # end class
