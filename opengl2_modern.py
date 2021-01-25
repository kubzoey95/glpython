import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import sys

VERTEX_SH = """
# version 330 core

in vec3 a_position;
in vec3 a_color;

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

if not glfw.init():
    raise Exception('AAAAAA')

window = glfw.create_window(1024, 768, 'XXXXXDDDD', None, None)

if not window:
    glfw.terminate()
    raise Exception('AAAAA')

glfw.set_window_pos(window, 100, 100)

glfw.make_context_current(window)

glClearColor(0, 0, 0, 0)

vertices = np.array([-0.5, -0.5, 0.0,
                     0.5, -0.5, 0.0,
                     0.0, 0.5, 0.0,
                     # colors in one array
                     1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0], dtype=np.float32)


shader = compileProgram(compileShader(VERTEX_SH, GL_VERTEX_SHADER), compileShader(FRAGMENT_SH, GL_FRAGMENT_SHADER))

glUseProgram(shader)

VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

position = glGetAttribLocation(shader, 'a_position')
glEnableVertexAttribArray(position)
glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))


color = glGetAttribLocation(shader, 'a_color')
glEnableVertexAttribArray(color)
glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(36))

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT)
    ct = glfw.get_time()
    glLoadIdentity()
    glScale(abs(math.sin(ct)), abs(math.cos(ct)), 1)
    glRotatef(ct, 0, ct / 2.0, 0)
    glTranslatef(math.sin(ct), math.cos(ct), 1)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    glfw.swap_buffers(window)

glfw.terminate()
