from my_engine.component import Component
import pyrr
import numpy as np

DEFAULT_RATIO = 1280/720


class Camera(Component):
    def __init__(self, obj,
                 projection: np.ndarray = pyrr.matrix44.create_perspective_projection_matrix(45, DEFAULT_RATIO, 0.1, 1000)):
        super().__init__('Camera', obj)
        self.__projection = projection

    @property
    def projection(self):
        return self.__projection

    @projection.setter
    def projection(self, val):
        self.__projection = val

    @property
    def matrix(self):
        return self.__obj.inverse_matrix @ self.__projection
