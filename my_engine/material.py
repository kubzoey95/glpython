from my_engine.component import Component
from typing import Union, Dict, NamedTuple
from collections import namedtuple
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_ACTIVE_ATTRIBUTES, \
    glGetProgramiv, glGetActiveAttrib, GLuint, GLenum, GLint, GLsizei, GLchar, glGetActiveUniform, GL_ACTIVE_UNIFORMS, \
    glGetUniformLocation, glGetAttribLocation, GL_GEOMETRY_SHADER


class Material(Component):
    def __init__(self, vertex_shader: Union[str, None] = None, fragment_shader: Union[str, None] = None,
                 geometry_shader: Union[str, None] = None, obj=None):
        super().__init__('Material', obj)
        self.__vertex_shader = vertex_shader
        self.__fragment_shader = fragment_shader
        self.__geometry_shader = geometry_shader
        self.__attributes: Dict[str, int] = {}
        self.__uniforms: Dict[str, int] = {}

        self.__attributes_values: Union[NamedTuple[np.ndarray], None] = None
        self.__uniforms_values: Union[NamedTuple[np.ndarray], None] = None

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
                self.__attributes[name.decode("utf-8")] = location

            self.__attributes_values = namedtuple('ShaderAttributes', self.__attributes.keys())(*[None for _ in range(len(self.__attributes))])

            num_active_uniforms = glGetProgramiv(program, GL_ACTIVE_UNIFORMS)
            for u in range(num_active_uniforms):
                name, size, type_ = glGetActiveUniform(program, u)
                location = glGetUniformLocation(program, name)
                self.__uniforms[name.decode("utf-8")] = location

            self.__uniforms_values = namedtuple('ShaderUniforms', self.__uniforms.keys())(*[None for _ in range(len(self.__uniforms))])

    @property
    def attributes(self):
        return self.__attributes_values

    @property
    def uniforms(self):
        return self.__uniforms_values
