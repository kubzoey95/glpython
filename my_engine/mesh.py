from my_engine.component import Component
import numpy as np
import meshio
from typing import Union, NamedTuple, Dict
from collections import namedtuple
import re


class Mesh(Component):
    def __init__(self, vertices: np.ndarray = np.array([]), indices: np.ndarray = np.array([]),
                 point_data: Union[Dict[str, np.ndarray], None] = None, obj=None):
        super().__init__('Mesh', obj)
        self.__vertices: np.ndarray = vertices
        self.__indices: np.ndarray = indices
        self.__point_data: Union[NamedTuple[np.ndarray], None] = None
        if point_data is not None:
            self.__point_data = namedtuple('PointData', point_data.keys())(*point_data.values())

    def load_from_file(self, file):
        mesh_obj = meshio.read(file)
        self.__vertices = mesh_obj.points
        self.__indices = mesh_obj.cells_dict['triangle']
        self.__point_data = namedtuple('PointData', [re.sub('[^0-9a-zA-Z_]', '', key) for key in mesh_obj.point_data.keys()])(*mesh_obj.point_data.values())

    @property
    def vertices(self):
        return self.__vertices

    @property
    def indices(self):
        return self.__indices

    @property
    def point_data(self):
        return self.__point_data
