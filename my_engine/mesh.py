from my_engine.component import Component
from my_engine.material import Material
import numpy as np
import meshio
from typing import Union, NamedTuple, Dict
from collections import namedtuple
import re
from math import inf


class Mesh(Component):
    def __init__(self, vertices: np.ndarray = np.array([]), indices: np.ndarray = np.array([]),
                 point_data: Union[Dict[str, np.ndarray], None] = None,
                 uniform_data: Union[Dict[str, np.ndarray], None] = None, obj=None,
                 vertices_mapping='vertices', indices_mapping='indices',
                 point_data_mapping: Union[Dict[str, str], None] = None,
                 uniform_data_mapping: Union[Dict[str, str], None] = None,
                 instanced_point_data: Union[Dict[str, np.ndarray], None] = None):
        super().__init__('Mesh', obj)
        self.__vertices: np.ndarray = vertices
        self.__vertices_mapping = vertices_mapping
        self.__indices: np.ndarray = indices
        self.__indices_mapping = indices_mapping
        self.__point_data: Union[Dict[str, np.ndarray], None] = None
        self.__point_data_mapping = point_data_mapping
        if point_data is not None:
            self.__point_data = point_data
        else:
            self.__point_data = {}

        if point_data_mapping is None:
            self.__point_data_mapping = {}

        if uniform_data is None:
            self.__uniform_data = {}

        if instanced_point_data is None:
            self.__instanced_point_data = {}

        self.__uniform_data_mapping = uniform_data_mapping

    def get_data_positioned_to_material(self, material: Material):
        # vertices_pos = material.attributes[self.__vertices_mapping]
        point_data_pos = [[material.attributes[self.__point_data_mapping[point_data]], self.__point_data[point_data]] for point_data in self.__point_data]
        point_data_pos.append([material.attributes[self.__vertices_mapping], self.__vertices])
        point_data_pos.sort(key=lambda x: x[0])

        # point_data_copy = dict(self.__point_data)
        # max_pos = [point_data[1] for point_data in point_data_pos]
        # max_pos.append(vertices_pos)
        # max_pos = max(max_pos)
        pos_array = np.zeros((len(self.__vertices), len(self.__vertices[0]) + sum((len(val[0]) for key, val in self.__point_data.items()))))
        # pos_array[vertices_pos] = self.__vertices
        pointer = 0
        for key, val in point_data_pos:
            pos_array[:, pointer:pointer+val.shape[1]] = val
            pointer = val.shape[1]

        return pos_array.flatten()

    def get_instanced_data_positioned_to_material(self, material: Material, flattened=True):
        point_data_pos = []
        min_length = inf
        data2length = {}
        for key, val in self.__instanced_point_data.items():
            if callable(val):
                value = val()
            else:
                value = val
            if len(value) < min_length:
                min_length = len(value)
            data2length[key] = len(value[0])
            point_data_pos.append([material.attributes[self.__point_data_mapping[key]], value])

        point_data_pos.sort(key=lambda x: x[0])

        pos_array = np.zeros((min_length, sum((val for key, val in data2length.items()))))
        pointer = 0
        for key, val in point_data_pos:
            pos_array[:, pointer:pointer + val.shape[1]] = val
            pointer = val.shape[1]

        if flattened:
            return pos_array.flatten()
        return pos_array

    def load_from_file(self, file):
        verts, uvs, normals, indices = self.load_obj_file(file)
        self.__vertices = np.array(verts)
        self.__indices = np.array(indices).reshape((len(indices) // 3, 3))
        self.__point_data['uvs'] = np.array(uvs)
        self.__point_data['normals'] = np.array(normals)
        self.__point_data_mapping['uvs'] = 'uvs'
        self.__point_data_mapping['normals'] = 'normals'

    @staticmethod
    def load_obj_file(path):
        vertices = []
        indices = {}
        normals = []
        uvs = []
        new_vertices = []
        new_uvs = []
        new_normals = []
        new_indices = []

        pos = 0
        with open(path) as f:
            line = f.readline()
            while line:
                values = line.split()
                if values:
                    if values[0] == 'v':
                        vertices.append(tuple(float(num) for num in values[1:]))
                    elif values[0] == 'vt':
                        uvs.append(tuple(float(num) for num in values[1:]))
                    elif values[0] == 'vn':
                        normals.append(tuple(float(num) for num in values[1:]))
                    elif values[0] == 'f':
                        for value in values[1:]:
                            index = [None, None, None]
                            for i, val in enumerate(value.split('/')):
                                if val.isdigit():
                                    index[i] = int(val) - 1

                            index = tuple(index)

                            if index not in indices:
                                new_indices.append(pos)
                                new_vertices.append(vertices[index[0]])
                                if index[1] is not None:
                                    new_uvs.append(uvs[index[1]])
                                if index[2] is not None:
                                    new_normals.append(normals[index[2]])
                                indices[index] = pos
                                pos += 1
                            else:
                                new_indices.append(indices[index])
                line = f.readline()

        return new_vertices, new_uvs, new_normals, new_indices

    @property
    def vertices(self):
        return self.__vertices

    @property
    def indices(self):
        return self.__indices

    @property
    def uniform_data(self):
        return self.__uniform_data

    @property
    def uniform_data_mapping(self):
        return self.__uniform_data_mapping

    @property
    def instanced_point_data(self):
        return self.__instanced_point_data

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
