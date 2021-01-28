from my_engine.component import Component
import numpy as np
import meshio
from typing import Union, NamedTuple, Dict
from collections import namedtuple
import re


class Mesh(Component):
    def __init__(self, vertices: np.ndarray = np.array([]), indices: np.ndarray = np.array([]),
                 point_data: Union[Dict[str, np.ndarray], None] = None,
                 uniform_data: Union[Dict[str, np.ndarray], None] = None, obj=None,
                 vertices_mapping='vertices', indices_mapping='indices',
                 point_data_mapping: Union[Dict[str, str], None] = None,
                 uniform_data_mapping: Union[Dict[str, str], None] = None):
        super().__init__('Mesh', obj)
        self.__vertices: np.ndarray = vertices
        self.__vertices_mapping = vertices_mapping
        self.__indices: np.ndarray = indices
        self.__indices_mapping = indices_mapping
        self.__point_data: Union[Dict[str, np.ndarray], None] = None
        self.__point_data_mapping = point_data_mapping
        if point_data is not None:
            self.__point_data = point_data

        self.__uniform_data = uniform_data

        self.__uniform_data_mapping = uniform_data_mapping

    def load_from_file(self, file):
        mesh_obj = meshio.read(file)
        self.__vertices = mesh_obj.points
        self.__indices = mesh_obj.cells_dict['triangle']
        self.__point_data = {re.sub('[^0-9a-zA-Z_]', '', key): val for key, val in mesh_obj.point_data.items()}
        self.__point_data_mapping = {key: key for key in mesh_obj.point_data.keys()}

    @property
    def vertices(self):
        return self.__vertices

    @property
    def indices(self):
        return self.__indices

    @property
    def point_data(self):
        return self.__point_data

    @property
    def vertices_mapping(self):
        return self.__vertices_mapping

    @vertices_mapping.setter
    def vertices_mapping(self, val):
        self.__vertices_mapping = val

    @property
    def indices_mapping(self):
        return self.__indices_mapping

    @indices_mapping.setter
    def indices_mapping(self, val):
        self.__indices_mapping = val

    @property
    def point_data_mapping(self):
        return self.__point_data_mapping
