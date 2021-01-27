from my_engine.component import Component
import numpy as np
import meshio


class Mesh(Component):
    def __init__(self, vertices=np.array([]), obj=None):
        super().__init__('Mesh', obj)
        self.vertices: np.ndarray = vertices

    def load_from_file(self, file):
        mesh_obj = meshio.read(file)
