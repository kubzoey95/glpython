import pyrr
from typing import Tuple
from numpy import float32


class GameObject:
    def __init__(self, translation: Tuple[float] = (0, 0, 0), rotation: Tuple[float] = (0, 0, 0),
                 scale: Tuple[float] = (1, 1, 1)):
        self.__translation = translation
        self.__rotation = rotation
        self.__scale = scale

        self.__trans_matrix = pyrr.matrix44.create_from_translation(translation)
        self.__rotation_matrix = pyrr.matrix44.create_from_eulers(rotation)
        self.__scale_matrix = pyrr.matrix44.create_from_scale(scale)

        self.__matrix = self.__trans_matrix @ self.__rotation_matrix @ self.__scale_matrix

        self.__matrix_valid = True

    @property
    def matrix(self):
        if self.__matrix_valid:
            return self.__matrix
        else:
            self.__matrix = self.__trans_matrix @ self.__rotation_matrix @ self.__scale_matrix
            self.__matrix_valid = True
            return self.__matrix

    @property
    def translation(self):
        return self.__translation

    @translation.setter
    def translation(self, val):
        self.__translation = val
        self.__trans_matrix = pyrr.matrix44.create_from_translation(val)
        self.__matrix_valid = False

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, val):
        self.__scale = val
        self.__scale_matrix = pyrr.matrix44.create_from_scale(val)
        self.__matrix_valid = False

    @property
    def rotation(self):
        return self.__rotation

    @rotation.setter
    def rotation(self, val):
        self.__rotation = val
        self.__rotation_matrix = pyrr.matrix44.create_from_eulers(val)
        self.__matrix_valid = False
