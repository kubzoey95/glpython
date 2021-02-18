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
    GL_TRIANGLES, GL_UNSIGNED_INT, glDrawRangeElements, glUniform1f, glDrawElementsInstanced, glVertexAttribDivisor, glUseProgram, \
    glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
import numpy as np
import ctypes
from typing import List


class Renderer(Component):
    def __init__(self, obj, active_camera: List[Camera]=None):
        super().__init__('Renderer', obj)
        self.__active_camera = active_camera

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
            descendants[child] = child.matrix @ descendants[child.parent]
            unchecked_children.extend(child.children)
        return descendants

    def start(self):
        self.__VBO = glGenBuffers(1)
        self.__EBO = glGenBuffers(1)
        self.__instanceVBO = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.__VBO)
        glBufferData(GL_ARRAY_BUFFER, self.__vertices.nbytes, self.__vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.__EBO)

        self.__texture_buffer = Texture.get_default_texture()
        return
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

    def update(self):
        for camera in self.__active_camera:
            self.__vertices = camera.vertices
            self.__indices = camera.indices
            self.__data_for_render = camera.get_data_for_render()
            glBindBuffer(GL_ARRAY_BUFFER, self.__VBO)
            glBufferData(GL_ARRAY_BUFFER, self.__vertices.nbytes, self.__vertices, GL_STATIC_DRAW)
            camera.bind_frame_buffer()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            for material, data in self.__data_for_render.items():
                material.use_program()
                num_of_items = sum(TYPE_TO_LENGTH[type_] for attrib, type_ in material.attributes_types.items() if attrib not in material.instanced_attribs)
                pointer = 0
                glBindBuffer(GL_ARRAY_BUFFER, self.__VBO)
                for attrib, index in sorted(material.attributes.items(), key=lambda x: x[1]):
                    if attrib not in material.instanced_attribs:
                        length = TYPE_TO_LENGTH[material.attributes_types[attrib]]
                        glEnableVertexAttribArray(index)
                        glVertexAttribPointer(index, length, GL_FLOAT, GL_FALSE, self.__vertices.itemsize * num_of_items, ctypes.c_void_p(pointer * 4))
                        pointer += length
                if material.view_matrix_name in material.uniforms:
                    glUniformMatrix4fv(material.uniforms[material.view_matrix_name], 1, GL_FALSE, camera.obj.inverse_matrix)
                if material.projection_matrix_name in material.uniforms:
                    glUniformMatrix4fv(material.uniforms[material.projection_matrix_name], 1, GL_FALSE, camera.projection)
                for dat in data:
                    for uniform, index in material.uniforms.items():
                        if material.uniforms_types[uniform] == GL_SAMPLER_2D:
                            if 'Texture' in dat['object'].components:
                                texture = dat['object'].components['Texture']
                                texture.bind_texture()

                                texture.load_settings()
                                if texture.default_texture == texture.texture:
                                    texture.send_texture()
                                # texture.bind_default_texture()
                        elif material.model_matrix_name == uniform:
                            glUniformMatrix4fv(index, 1, GL_FALSE, dat['object'].transformation_matrix)
                        elif material.view_matrix_name == uniform:
                            pass
                        elif material.projection_matrix_name == uniform:
                            pass
                        elif uniform in dat['object'].components['Mesh'].uniform_data:
                            uniform_data = dat['object'].components['Mesh'].uniform_data[uniform]
                            if callable(uniform_data):
                                uniform_data = uniform_data()
                                if type(uniform_data) == int or float:
                                    glUniform1f(index, float(uniform_data))
                            elif type(uniform_data) == int or float:
                                glUniform1f(index, float(uniform_data))

                    # glDrawElements(GL_TRIANGLES, dat['length'], GL_UNSIGNED_INT, ctypes.c_void_p(dat['pointer'] * 0))
                    flatten_indices = dat['indices']
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.__EBO)
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER, flatten_indices.nbytes, flatten_indices, GL_STATIC_DRAW)
                    # glDrawElements(GL_TRIANGLES, len(flatten_indices), GL_UNSIGNED_INT, None)
                    mesh = dat['object'].components['Mesh']
                    if len(mesh.instanced_point_data):
                        instance_data = mesh.get_instanced_data_positioned_to_material(material, flattened=False)
                        instance_cnt = len(instance_data)
                        instance_data = instance_data.flatten()
                        glBindBuffer(GL_ARRAY_BUFFER, self.__instanceVBO)
                        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_STATIC_DRAW)
                        num_of_items = sum(TYPE_TO_LENGTH[material.attributes_types[attrib]] for attrib in material.instanced_attribs)
                        ptr = 0
                        for attrib, index in sorted(material.attributes.items(), key=lambda x: x[1]):
                            if attrib in material.instanced_attribs:
                                length = TYPE_TO_LENGTH[material.attributes_types[attrib]]
                                glEnableVertexAttribArray(index)
                                glVertexAttribPointer(index, length, GL_FLOAT, GL_FALSE, instance_data.itemsize * num_of_items, ctypes.c_void_p(ptr * 4))
                                glVertexAttribDivisor(index, 1)
                                ptr += length

                        glDrawElementsInstanced(GL_TRIANGLES, len(flatten_indices), GL_UNSIGNED_INT, None, instance_cnt)
                    else:
                        glDrawRangeElements(GL_TRIANGLES, dat['pointer'], dat['pointer'] + dat['length'], len(flatten_indices), GL_UNSIGNED_INT, None)
                    # glUseProgram(0)
                    # glDrawRangeElements(GL_TRIANGLES, len(self.__indices), GL_UNSIGNED_INT, None)