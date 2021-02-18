from my_engine.component import Component
import pyrr
import numpy as np
from typing import Tuple, Union
from collections import defaultdict
from OpenGL.GL import glBindFramebuffer, GL_FRAMEBUFFER, glBindRenderbuffer, glClearColor, GL_DEPTH_TEST, glEnable, \
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, glBlendFunc

DEFAULT_RATIO = 1280/720


class Camera(Component):
    def __init__(self, obj,
                 projection: np.ndarray = pyrr.matrix44.create_perspective_projection_matrix(45, DEFAULT_RATIO, 0.1, 1000),
                 frame_buffer=0, depth_buffer=None, clear_color: Tuple = (0.1, 0.1, 0.1, 1), enable: Tuple = (GL_DEPTH_TEST,),
                 blend_settings: Union[Tuple, None] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)):
        super().__init__('Camera', obj)
        self.__projection = projection
        self.__frame_buffer = frame_buffer
        self.__depth_buffer = depth_buffer
        self.__clear_color = clear_color
        self.__enable = enable
        self.__blend_settings = blend_settings
        self.__all_objects = []
        self.__data_for_render = {}
        self.__vertices = np.array([])
        self.__indices = np.array([])
        self.objects = []

    @property
    def projection(self):
        return self.__projection

    @property
    def vertices(self):
        return self.__vertices

    @property
    def indices(self):
        return self.__indices

    @projection.setter
    def projection(self, val):
        self.__projection = val

    @property
    def matrix(self):
        return self.__obj.inverse_matrix @ self.__projection

    def get_all_objects(self):
        descendants = {child: child.matrix for child in self.objects}
        unchecked_children = [child1 for child in self.objects for child1 in child.children]
        while unchecked_children:
            child = unchecked_children.pop()
            descendants[child] = child.matrix @ descendants[child.parent]
            unchecked_children.extend(child.children)
        return descendants

    def start(self):
        self.__all_objects = [(key, val) for key, val in self.get_all_objects().items() if 'Material' in key.components and 'Mesh' in key.components]

        vertices = []
        indices = []

        shaders = defaultdict(list)
        verts_len = 0
        for obj in self.__all_objects:
            material = obj[0].components['Material']
            mesh = obj[0].components['Mesh']
            vertices.extend(list(mesh.get_data_positioned_to_material(material)))
            object_data = {'pointer': len(vertices), 'length': mesh.vertices.size, 'matrix': obj[1], 'object': obj[0],
                           'indices': mesh.indices.flatten() + verts_len}
            # indices.extend(list(mesh.indices.flatten() + verts_len))
            verts_len += len(mesh.vertices)
            if 'Texture' in obj[0].components:
                object_data['texture'] = obj[0].components['Texture']
            shaders[material].append(object_data)

        self.__data_for_render = shaders

        self.__vertices = np.array(vertices, dtype=np.float32)
        self.__indices = np.array(indices, dtype=np.uint32)

    def get_data_for_render(self):
        return self.__data_for_render

    def bind_frame_buffer(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.__frame_buffer)
        glClearColor(*self.__clear_color)
        for setting in self.__enable:
            glEnable(setting)
        if self.__blend_settings:
            glEnable(GL_BLEND)
            glBlendFunc(*self.__blend_settings)
