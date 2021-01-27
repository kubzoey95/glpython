from my_engine.component import Component
from typing import Union
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_ACTIVE_ATTRIBUTES,\
    glGetProgramiv, glGetActiveAttrib, GLuint, GLenum, GLint, GLsizei, GLchar, glGetActiveUniform, GL_ACTIVE_UNIFORMS


class Material(Component):
    def __init__(self, vertex_shader: Union[str, None] = None, fragment_shader: Union[str, None] = None, obj=None):
        super().__init__('Material', obj)
        self.__vertex_shader = vertex_shader
        self.__fragment_shader = fragment_shader

        if self.__vertex_shader and self.__fragment_shader:
            program = compileProgram(compileShader(vertex_shader, GL_VERTEX_SHADER), compileShader(fragment_shader, GL_FRAGMENT_SHADER))
            self.__program = program

            num_active_attribs = glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES)
            for u in range(num_active_attribs):
                name, size, type_ = glGetActiveAttrib(program, u)
                print(name, size, type_)

            num_active_uniforms = glGetProgramiv(program, GL_ACTIVE_UNIFORMS)
            for u in range(num_active_uniforms):
                name, size, type_ = glGetActiveUniform(program, u)
                print(name, size, type_)
