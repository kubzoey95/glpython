from my_engine.game_object import GameObject
from typing import Tuple, Union, Set
import numpy as np


class Mesh(GameObject):
    def __init__(self, translation: Tuple[float] = (0, 0, 0), rotation: Tuple[float] = (0, 0, 0),
                 scale: Tuple[float] = (1, 1, 1), name: str = "Untitled"):
        super().__init__(translation, rotation, scale, name)

    @staticmethod
    def get_vertices():
        return np.array([])
