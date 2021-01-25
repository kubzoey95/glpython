import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import sys
import pyrr

VERTEX_SH = """
# version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

uniform mat4 rotation;

out vec3 v_color;

void main(){
    gl_Position = rotation * vec4(a_position, 1.0);
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

vertices = np.array([-0.5, -0.5, 0.5, 1.0, 0.0, 0.0,
                     0.5, -0.5, 0.5, 0.0, 1.0, 0.0,
                     0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                     -0.5, 0.5, 0.5, 1.0, 1.0, 1.0,

                     -0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
                     0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
                     0.5, 0.5, -0.5, 0.0, 0.0, 1.0,
                     -0.5, 0.5, -0.5, 1.0, 1.0, 1.0
                     # colors in one array strided
                     ], dtype=np.float32)

indices = np.array([0, 1, 2, 2, 3, 0,
                    4, 5, 6, 6, 7, 4,
                    4, 5, 1, 1, 0, 4,
                    6, 7, 3, 3, 2, 6,
                    5, 6, 2, 2, 1, 5,
                    7, 4, 0, 0, 3, 7], dtype=np.uint32)

shader = compileProgram(compileShader(VERTEX_SH, GL_VERTEX_SHADER), compileShader(FRAGMENT_SH, GL_FRAGMENT_SHADER))

glUseProgram(shader)
glClearColor(0, 0, 0, 0)
glEnable(GL_DEPTH_TEST)

VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# index buffer
EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)  # 0 - position layout location
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
# 0 - position layout location

rotation_location = glGetUniformLocation(shader, 'rotation')

glEnableVertexAttribArray(1)  # 1 - color layout location
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    ct = glfw.get_time()

    rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
    rot_y = pyrr.Matrix44.from_y_rotation(0.7 * glfw.get_time())

    glUniformMatrix4fv(rotation_location, 1, GL_FALSE, rot_x * rot_y)

    # glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    glfw.swap_buffers(window)

glfw.terminate()
