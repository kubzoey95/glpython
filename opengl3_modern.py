import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import sys

VERTEX_SH = """
# version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

out vec3 v_color;

void main(){
    gl_Position = vec4(a_position, 1.0);
    v_color = a_color;
}
"""

FRAGMENT_SH = """
# version 330 core

out vec4 out_color;
in vec3 v_color;

void main(){
    out_color = vec4(v_color, 1.0);
}
"""


def window_resize(wind, width, height):
    glViewport(0, 0, width, height)


if not glfw.init():
    raise Exception('AAAAAA')

window = glfw.create_window(1024, 768, 'XXXXXDDDD', None, None)

if not window:
    glfw.terminate()
    raise Exception('AAAAA')

glfw.set_window_pos(window, 100, 100)
glfw.set_window_size_callback(window, window_resize)


glfw.make_context_current(window)

glClearColor(0, 0, 0, 0)

vertices = np.array([-0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
                     0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
                     -0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
                     0.5, 0.5, 0.0, 1.0, 1.0, 1.0
                     # colors in one array strided
                     ], dtype=np.float32)



shader = compileProgram(compileShader(VERTEX_SH, GL_VERTEX_SHADER), compileShader(FRAGMENT_SH, GL_FRAGMENT_SHADER))

glUseProgram(shader)

VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)  # 0 - position layout location
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
# 0 - position layout location

glEnableVertexAttribArray(1)  # 1 - color layout location
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT)
    ct = glfw.get_time()

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    glfw.swap_buffers(window)

glfw.terminate()
