from my_engine.component import Component
from my_engine.material import Material
from my_engine.mesh import Mesh
from my_engine.camera import Camera
from typing import Tuple, Union
from collections import defaultdict
from OpenGL.GL import glGenBuffers, glClearColor, glEnable, glBlendFunc, GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA


class Renderer(Component):
    def __init__(self, obj, active_camera: Camera=None, enable: Tuple = (GL_DEPTH_TEST,),
                 blend_settings: Union[Tuple[2], None] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
                 clear_color: Tuple[4] = (0.1, 0.1, 0.1, 1)):
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

    @property
    def active_camera(self):
        return self.__active_camera

    @active_camera.setter
    def active_camera(self, val):
        self.__active_camera = val

    def get_all_objects(self):
        descendants = {child: child.matrix for child in self.__obj.children}
        unchecked_children = [child1 for child in self.__obj.children for child1 in child.children]
        while unchecked_children:
            child = unchecked_children.pop()
            descendants[child] = descendants[child.parent] * child.matrix
            unchecked_children.extend(child.children)
        return descendants

    def start(self):
        self.__VBO = glGenBuffers(1)
        self.__EBO = glGenBuffers(1)
        glClearColor(*self.__clear_color)
        for setting in self.__enable:
            glEnable(setting)
        if self.__blend_settings:
            glEnable(GL_BLEND)
            glBlendFunc(*self.__blend_settings)

        self.__all_objects = [(key, val) for key, val in self.get_all_objects().items() if 'Material' in key.components and 'Mesh' in key.components]

        shaders = defaultdict(list)

        for obj in self.__all_objects:
            material = obj[0].components['Material']
            mesh = obj[0].components['Mesh']
            shaders[material].append((mesh.get_data_positioned_to_material(material), mesh.indices, obj[1]))
