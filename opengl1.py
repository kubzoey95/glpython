import glfw
from OpenGL.GL import *
import numpy as np
import math

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
                     0.0, 0.5, 0.0], dtype=np.float32)

colors = np.array([1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0], dtype=np.float32)

glEnableClientState(GL_VERTEX_ARRAY)
glVertexPointer(3, GL_FLOAT, 0, vertices)

glEnableClientState(GL_COLOR_ARRAY)
glColorPointer(3, GL_FLOAT, 0, colors)

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
