import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import sys
import pyrr
from PIL import Image

VERTEX_SH = """
# version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec2 a_texture;

uniform mat4 rotation;

out vec3 v_color;
out vec2 v_texture;

void main(){
    gl_Position = rotation * vec4(a_position, 1.0);
    v_color = a_color;
    v_texture = a_texture;
}
"""

FRAGMENT_SH = """
# version 330 core

out vec4 out_color;
in vec3 v_color;
in vec2 v_texture;

uniform sampler2D s_texture;

void main(){
    out_color = texture(s_texture, v_texture);
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

vertices = np.array([-0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                     0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                     -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

                     -0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                     0.5, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                     -0.5, 0.5, -0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

                     0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                     0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                     0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

                     -0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                     -0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                     -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                     -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

                     -0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                     0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                     -0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

                     0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                     -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                     -0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                     0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)

indices = np.array([0, 1, 2, 2, 3, 0,
                    4, 5, 6, 6, 7, 4,
                    8, 9, 10, 10, 11, 8,
                    12, 13, 14, 14, 15, 12,
                    16, 17, 18, 18, 19, 16,
                    20, 21, 22, 22, 23, 20], dtype=np.uint32)

shader = compileProgram(compileShader(VERTEX_SH, GL_VERTEX_SHADER), compileShader(FRAGMENT_SH, GL_FRAGMENT_SHADER))

glUseProgram(shader)
glClearColor(0, 0, 0, 0)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# vertex buffer
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# index buffer
EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

# vertex
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(0))

# color
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(12))

# uvs
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(24))

rotation_location = glGetUniformLocation(shader, 'rotation')

# tekstury
texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)

# TEXTURE WRAP
glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

# TEX FILTERING
glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# load image
img = Image.open('cat.png').transpose(Image.FLIP_TOP_BOTTOM)

img_data = img.convert('RGBA').tobytes()
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

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
