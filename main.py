import os
import cv2
import numpy as np
import math
from PIL import Image


from ConfigReader import PATH_TO_IMAGES, PATH_TO_RESULTS, PATH_TO_HAARCASCADE


class PassportDetector(object):
    def __init__(self):
        ...

    def execute(self):
        __raw_imgs: list = self.__read_images(PATH_TO_IMGS=PATH_TO_IMAGES)
        __resized_img_list: list = self.__resize_images(img_list=__raw_imgs)
        __blurred_img_list: list = self.__blur_images(img_list=__resized_img_list)
        __gray_img_list: list = self.__gray_images(img_list=__blurred_img_list)
        __thresh_img_list: list = self.__thresh_images(img_list=__gray_img_list)
        __contours_list: list = self.__get_contours(img_list=__thresh_img_list)
        __largest_contours: list = self.__get_largest_contour(img_list=__resized_img_list, contours_list=__contours_list)
        __processed_data: list = self.__process_images(image_list=__resized_img_list, contour_list=__largest_contours)
        __rotated_img_list, __rotated_point_list = self.__rotate(image_list=__resized_img_list, processed_data=__processed_data)
        __moved_img_list: list = self.__move_images(image_list=__rotated_img_list, processed_data=__processed_data)
        __resized_processed_img_list: list = self.__resize_processed_img(image_list=__moved_img_list, processed_data=__rotated_point_list)
        __unflipped_img_list: list = self.__unflip_check(image_list=__resized_processed_img_list)
        self.__save_results(image_list=__unflipped_img_list)

    @staticmethod
    def __read_images(PATH_TO_IMGS: str) -> list:
        img_list: list = []
        file_list: list = os.listdir(PATH_TO_IMGS)

        for FILE in file_list:
            img = cv2.imread(PATH_TO_IMGS + FILE)
            img_list.append(img)

        return img_list

    @staticmethod
    def __resize_images(img_list: list) -> list:
        resized_img_list: list = []
        for img in img_list:
            ORIGINAL_WIDTH: int = np.shape(img)[0]
            ORIGINAL_HEIGHT: int = np.shape(img)[1]

            SCALE_COEFF: float = ORIGINAL_WIDTH / 640

            SCALED_WIDTH: int = int(ORIGINAL_WIDTH / SCALE_COEFF)
            SCALED_HEIGHT: int = int(ORIGINAL_HEIGHT / SCALE_COEFF)

            resized_image = cv2.resize(img, (SCALED_HEIGHT, SCALED_WIDTH))
            resized_img_list.append(resized_image)

        return resized_img_list

    @staticmethod
    def __blur_images(img_list: list, COEFF: int = 3) -> list:
        blurred_img_list: list = []
        for img in img_list:
            blured_image = cv2.medianBlur(img, COEFF)
            blurred_img_list.append(blured_image)

        return blurred_img_list

    @staticmethod
    def __gray_images(img_list: list) -> list:
        gray_img_list: list = []
        for img in img_list:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img_list.append(gray_img)

        return gray_img_list

    @staticmethod
    def __thresh_images(img_list: list, LOWER: int = 2, UP: int = 25) -> list:
        thresh_img_list: list = []
        for img in img_list:
            thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, UP, LOWER)
            thresh_img_list.append(thresh_img)

        return thresh_img_list

    @staticmethod
    def __get_contours(img_list: list) -> list:
        contours_list: list = []
        for img in img_list:
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_list.append(contours)

        return contours_list

    @staticmethod
    def __get_largest_contour(img_list: list, contours_list: list):
        largest_contours: list = []
        for contours, img in zip(contours_list, img_list):
            LARGEST_CONTOUR_AREA: float = .0
            for contour in contours:
                if cv2.contourArea(contour) > LARGEST_CONTOUR_AREA:
                    LARGEST_CONTOUR_AREA: float = cv2.contourArea(contour)
                    largest_contour = contour
            largest_contours.append(largest_contour)

        return largest_contours

    @staticmethod
    def __rotate(
            image_list: list,
            processed_data: list
        ) -> tuple:

        rotated_image_list: list = []
        rotated_point_list: list = []
        for image, data in zip(image_list, processed_data):
            rotation_point: list = data["rotation_point"]
            side_point: list = data["side_point"]

            LENGTH_1: float = ((rotation_point[1] - side_point[1]) ** 2) ** 0.5
            LENGTH_2: float = ((side_point[0] - rotation_point[0]) ** 2) ** 0.5

            if data["case"] == 1:
                DEGREES = math.degrees(math.tan(LENGTH_1 / LENGTH_2)) - 90
            elif data["case"] == 2:
                DEGREES = -math.degrees(math.tan(LENGTH_1 / LENGTH_2)) - 90
            elif data["case"] == 3:
                DEGREES = math.degrees(math.tan(LENGTH_2 / LENGTH_1))
            elif data["case"] == 4:
                DEGREES = -math.degrees(math.tan(LENGTH_2 / LENGTH_1))
            elif data["case"] == 5:
                DEGREES = 0
            elif data["case"] == 6:
                DEGREES = 0

            rotation_matrix = cv2.getRotationMatrix2D((int(rotation_point[0]), int(rotation_point[1])), DEGREES, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1]*2, image.shape[0]*2))

            SHIFT_X, SHIFT_Y = data["rotation_point"][0], data["rotation_point"][1]

            moved_rotation_point = np.asarray((data["rotation_point"][0] - SHIFT_X, data["rotation_point"][1] - SHIFT_Y))
            moved_side_point = np.asarray((data["side_point"][0] - SHIFT_X, data["side_point"][1] - SHIFT_Y))
            moved_antirotation_point = np.asarray((data["antirotation_point"][0] - SHIFT_X, data["antirotation_point"][1] - SHIFT_Y))

            custom_rotation_matrix: np.ndarray = np.asarray(
                [
                    [math.cos(np.radians(DEGREES)), math.sin(np.radians(DEGREES))],
                    [-math.sin(np.radians(DEGREES)), math.cos(np.radians(DEGREES))]
                ]
            )

            moved_shifted_rotation_point: np.ndarray = custom_rotation_matrix @ moved_rotation_point
            moved_shifted_side_point: np.ndarray = custom_rotation_matrix @ moved_side_point
            moved_shifted_antirotation_point: np.ndarray = custom_rotation_matrix @ moved_antirotation_point

            moved_shifted_rotation_point = moved_shifted_rotation_point.astype(int)
            moved_shifted_side_point = moved_shifted_side_point.astype(int)
            moved_shifted_antirotation_point = moved_shifted_antirotation_point.astype(int)

            points: dict = {
                "rotated_rotation_point": moved_shifted_rotation_point,
                "rotated_side_point": moved_shifted_side_point,
                "rotated_antirotation_point": moved_shifted_antirotation_point,
            }

            rotated_image_list.append(rotated_image)
            rotated_point_list.append(points)

        return rotated_image_list, rotated_point_list

    @staticmethod
    def __move_images(image_list: list, processed_data: list) -> list:
        moved_image_list: list = []

        for image, data in zip(image_list, processed_data):

            rotation_point: np.ndarray = data["rotation_point"]
            wrap_x, wrap_y = -rotation_point[0], -rotation_point[1]
            transition_matrix: np.ndarray = np.float32([[1, 0, wrap_x], [0, 1, wrap_y]])
            img_trans = cv2.warpAffine(image, transition_matrix, (image.shape[1], image.shape[0]))
            moved_image_list.append(img_trans)

        return moved_image_list

    def __process_images(self, image_list: list, contour_list: list) -> list:
        processed_data: list = []
        for image, contour in zip(image_list, contour_list):
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            index_list: list = list(range(len(box)))

            y_list: list = []
            for i in index_list:
                y_list.append(box[i][1])

            UPPERMOST_POINT_Y_COORD: int = min(y_list)
            if y_list.count(UPPERMOST_POINT_Y_COORD) == 1:
                UPPERMOST_POINT_INDEX: int = y_list.index(UPPERMOST_POINT_Y_COORD)

                index_list.pop(UPPERMOST_POINT_INDEX)
                y_dist: list = []
                LOWEST_POINT_Y_COORD: int = 0
                for index in index_list:
                    if LOWEST_POINT_Y_COORD < box[index][1]:
                        LOWEST_POINT_Y_COORD: int = int(box[index][1])
                    y_dist.append(int(box[index][1]))

                index_list.pop(y_dist.index(LOWEST_POINT_Y_COORD))

                point1: tuple = box[index_list[0]]
                point2: tuple = box[index_list[1]]

                # Определяем какая из двух отсавшихся точек выше (по длиной стороне или по короткой)
                if point1[1] > point2[1]:
                    lower: tuple = point1
                    upper: tuple = point2
                else:
                    upper: tuple = point1
                    lower: tuple = point2

                # фиксируем координаты верхней точки
                X: int = box[UPPERMOST_POINT_INDEX][0]
                Y: int = box[UPPERMOST_POINT_INDEX][1]

                # фиксируем координаты верхней и нижней точек
                X1, Y1 = lower[0], lower[1]
                X2, Y2 = upper[0], upper[1]

                # считаем длины катетов
                CATHETUS_1_LEHGTH: float = ((X1 - X) ** 2 + (Y1 - Y) ** 2) ** 0.5
                CATHETUS_2_LEHGTH: float = ((X2 - X) ** 2 + (Y2 - Y) ** 2) ** 0.5

                if CATHETUS_1_LEHGTH < CATHETUS_2_LEHGTH:
                    if X > X1:
                        point_data: dict = {
                            "case": 1,
                            "side_point": box[3],
                            "rotation_point": box[0],
                            "antirotation_point": box[2]
                        }
                    else:
                        point_data: dict = {
                            "case": 2,
                            "side_point": box[2],
                            "rotation_point": box[3],
                            "antirotation_point": box[1]
                        }
                else:
                    if X < X2:
                        point_data: dict = {
                            "case": 3,
                            "side_point": box[0],
                            "rotation_point": box[1],
                            "antirotation_point": box[3]
                        }
                    else:
                        point_data: dict = {
                            "case": 4,
                            "side_point": box[3],
                            "rotation_point": box[0],
                            "antirotation_point": box[2]
                        }
            else:
                if box[2][0] > box[2][1]:
                    point_data: dict = {
                        "case": 5,
                        "side_point": box[3],
                        "rotation_point": box[0],
                        "antirotation_point": box[2]
                    }
                else:
                    point_data: dict = {
                        "case": 6,
                        "side_point": box[3],
                        "rotation_point": box[0],
                        "antirotation_point": box[2]
                    }
            processed_data.append(point_data)

        return processed_data

    @staticmethod
    def __resize_processed_img(image_list: list, processed_data: list) -> list:
        resized_processed_image_list: list = []
        for img, point in zip(image_list, processed_data):

            crop_img = img[:point["rotated_antirotation_point"][1], :point["rotated_antirotation_point"][0]]

            resized_image: np.ndarray = cv2.resize(crop_img, (480, 640))

            resized_processed_image_list.append(resized_image)

        return resized_processed_image_list

    @staticmethod
    def __unflip_check(image_list: list) -> list:
        flipped_img_list: list = []
        for image in image_list:
            face_cascade = cv2.CascadeClassifier(PATH_TO_HAARCASCADE)

            x, y = image.shape[:2]
            x_rotation = y // 2
            y_rotation = x // 2

            rotation_matrix: np.ndarray = cv2.getRotationMatrix2D((x_rotation, y_rotation), 180, 1)
            flipped_image: np.ndarray = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

            gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            flipped_gray_scale = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.5, minNeighbors=5)
            flipped_faces = face_cascade.detectMultiScale(flipped_gray_scale, scaleFactor=1.5, minNeighbors=5)

            if len(faces) == 0 and len(flipped_faces) > 0:
                flipped_img_list.append(flipped_image)

            elif len(faces) > 0 and len(flipped_faces) == 0:
                flipped_img_list.append(image)

        return flipped_img_list

    @staticmethod
    def __save_results(image_list: list) -> None:
        for img, NAMES in zip(image_list, os.listdir(PATH_TO_IMAGES)):
            color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image: Image = Image.fromarray(color_coverted)
            image.save(PATH_TO_RESULTS + NAMES)


if __name__ == "__main__":
    passports_detector: PassportDetector = PassportDetector()
    passports_detector.execute()
