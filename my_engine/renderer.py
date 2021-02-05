from my_engine.component import Component
from my_engine.material import Material, TYPE_TO_LENGTH
from my_engine.mesh import Mesh
from my_engine.texture import Texture
from my_engine.camera import Camera
from typing import Tuple, Union
from collections import defaultdict
from OpenGL.GL import glGenBuffers, glBindBuffer, glBufferData, glClearColor, glEnable, glBlendFunc, GL_DEPTH_TEST, \
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, \
    glEnableVertexAttribArray, glVertexAttribPointer, GL_FALSE, GL_FLOAT, GL_SAMPLER_2D, glUniformMatrix4fv, glDrawElements, \
    GL_TRIANGLES, GL_UNSIGNED_INT, glDrawRangeElements
import numpy as np
import ctypes


class Renderer(Component):
    def __init__(self, obj, active_camera: Camera=None, enable: Tuple = (GL_DEPTH_TEST,),
                 blend_settings: Union[Tuple, None] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
                 clear_color: Tuple = (0.1, 0.1, 0.1, 1)):
        super().__init__('Renderer', obj)
        self.__active_camera = active_camera
        self.__enable = enable
        self.__blend_settings = blend_settings
        self.__clear_color = clear_color
        self.__VBO = None
        # glBindBuffer(GL_ARRAY_BUFFER, VBO)
        # glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.__EBO = None
        # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        # glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        self.__all_objects = {}
        self.__data_for_render = {}
        self.__vertices = np.array([])
        self.__indices = np.array([])
        self.__texture_buffer = None

    @property
    def active_camera(self):
        return self.__active_camera

    @active_camera.setter
    def active_camera(self, val):
        self.__active_camera = val

    def get_all_objects(self):
        descendants = {child: child.matrix for child in self.obj.children}
        unchecked_children = [child1 for child in self.obj.children for child1 in child.children]
        while unchecked_children:
            child = unchecked_children.pop()
            descendants[child] = descendants[child.parent] @ child.matrix
            unchecked_children.extend(child.children)
        return descendants

    def start(self):
        glClearColor(*self.__clear_color)
        for setting in self.__enable:
            glEnable(setting)
        if self.__blend_settings:
            glEnable(GL_BLEND)
            glBlendFunc(*self.__blend_settings)

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

        self.__VBO = glGenBuffers(1)
        self.__EBO = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.__VBO)
        glBufferData(GL_ARRAY_BUFFER, self.__vertices.nbytes, self.__vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.__EBO)

        self.__texture_buffer = Texture.get_buffer()

    def update(self):
        for material, data in self.__data_for_render.items():
            num_of_items = sum(TYPE_TO_LENGTH[type_] for attrib, type_ in material.attributes_types.items())
            pointer = 0
            for attrib, index in sorted(material.attributes.items(), key=lambda x: x[1]):
                length = TYPE_TO_LENGTH[material.attributes_types[attrib]]
                glEnableVertexAttribArray(index)
                glVertexAttribPointer(index, length, GL_FLOAT, GL_FALSE, self.__vertices.itemsize * num_of_items, ctypes.c_void_p(pointer * 4))
                pointer += length
            material.use_program()
            glUniformMatrix4fv(material.uniforms[material.view_matrix_name], 1, GL_FALSE, self.__active_camera.obj.inverse_matrix)
            glUniformMatrix4fv(material.uniforms[material.projection_matrix_name], 1, GL_FALSE, self.__active_camera.projection)
            for dat in data:
                for uniform, index in material.uniforms.items():
                    if material.uniforms_types[uniform] == GL_SAMPLER_2D:
                        if 'Texture' in dat['object'].components:
                            texture = dat['object'].components['Texture']

                            texture.load_settings()
                            texture.send_texture()
                    elif material.model_matrix_name == uniform:
                        glUniformMatrix4fv(index, 1, GL_FALSE, dat['matrix'])
                    elif material.view_matrix_name == uniform:
                        pass
                    elif material.projection_matrix_name == uniform:
                        pass
                # glDrawElements(GL_TRIANGLES, dat['length'], GL_UNSIGNED_INT, ctypes.c_void_p(dat['pointer'] * 0))
                flatten_indices = dat['indices']
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, flatten_indices.nbytes, flatten_indices, GL_STATIC_DRAW)
                # glDrawElements(GL_TRIANGLES, len(flatten_indices), GL_UNSIGNED_INT, None)
                glDrawRangeElements(GL_TRIANGLES, dat['pointer'], dat['pointer'] + dat['length'], len(flatten_indices), GL_UNSIGNED_INT, None)
                # glDrawRangeElements(GL_TRIANGLES, len(self.__indices), GL_UNSIGNED_INT, None)