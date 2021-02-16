from my_engine.component import Component
from typing import Union, Dict, NamedTuple
from collections import namedtuple
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_ACTIVE_ATTRIBUTES, \
    glGetProgramiv, glGetActiveAttrib, GLuint, GLenum, GLint, GLsizei, GLchar, glGetActiveUniform, GL_ACTIVE_UNIFORMS, \
    glGetUniformLocation, glGetAttribLocation, GL_GEOMETRY_SHADER, glUseProgram, GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, \
    GL_FLOAT_VEC4, GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4, glUniformMatrix4fv, GL_FLOAT_MAT4


TYPE_TO_LENGTH = {
    GL_FLOAT: 1,
    GL_FLOAT_VEC2: 2,
    GL_FLOAT_VEC3: 3,
    GL_FLOAT_VEC4: 4,
    GL_INT: 1,
    GL_INT_VEC2: 2,
    GL_INT_VEC3: 3,
    GL_INT_VEC4: 4
}

UNIFORM_TYPE_TO_FUNC = {
    GL_FLOAT_MAT4: glUniformMatrix4fv,

}


class Material(Component):
    def __init__(self, vertex_shader: Union[str, None] = None, fragment_shader: Union[str, None] = None,
                 geometry_shader: Union[str, None] = None, view_matrix_name: str = 'view_matrix',
                 projection_matrix_name: str = 'projection_matrix', model_matrix_name: str = 'model_matrix',
                 instanced_attribs=(), obj=None):
        super().__init__('Material', obj)
        self.__vertex_shader = vertex_shader
        self.__fragment_shader = fragment_shader
        self.__geometry_shader = geometry_shader
        self.__attributes: Dict[str, int] = {}
        self.__uniforms: Dict[str, int] = {}
        self.view_matrix_name = view_matrix_name
        self.projection_matrix_name = projection_matrix_name
        self.model_matrix_name = model_matrix_name

        self.__instanced_attribs = set(instanced_attribs)

        self.__attributes_values: Union[Dict[str, np.ndarray]] = {}
        self.__uniforms_values: Union[Dict[str, np.ndarray]] = {}
        self.__attributes_types = {}
        self.__uniforms_types = {}

        if self.__vertex_shader and self.__fragment_shader:
            program = compileProgram(
                *[compileShader(shader, const) for shader, const in [(self.__vertex_shader, GL_VERTEX_SHADER),
                                                                     (self.__fragment_shader, GL_FRAGMENT_SHADER),
                                                                     (self.__geometry_shader, GL_GEOMETRY_SHADER)] if shader]
            )
            self.__program = program

            num_active_attribs = glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES)
            for u in range(num_active_attribs):
                name, size, type_ = glGetActiveAttrib(program, u)
                location = glGetAttribLocation(program, name)
                name = name.decode("utf-8")
                self.__attributes_types[name] = type_
                self.__attributes[name] = location
                self.__attributes_values[name] = np.array([])

            num_active_uniforms = glGetProgramiv(program, GL_ACTIVE_UNIFORMS)
            for u in range(num_active_uniforms):
                name, size, type_ = glGetActiveUniform(program, u)
                location = glGetUniformLocation(program, name)
                name = name.decode("utf-8")
                self.__uniforms_types[name] = type_
                self.__uniforms[name] = location
                self.__uniforms_values[name] = np.array([])

    @property
    def instanced_attribs(self):
        return self.__instanced_attribs

    @property
    def attributes(self):
        return self.__attributes

    @property
    def attributes_types(self):
        return self.__attributes_types

    @property
    def uniforms_types(self):
        return self.__uniforms_types

    @property
    def uniforms(self):
        return self.__uniforms

    def use_program(self):
        glUseProgram(self.__program)
