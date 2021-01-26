import pyrr
from typing import Tuple, Union, Set


class GameObject:
    def __init__(self, translation: Tuple[float] = (0, 0, 0), rotation: Tuple[float] = (0, 0, 0),
                 scale: Tuple[float] = (1, 1, 1), name: str = "Untitled"):
        self.__translation = translation
        self.__rotation = rotation
        self.__scale = scale

        self.__trans_matrix = pyrr.matrix44.create_from_translation(translation)
        self.__trans_matrix_valid = True
        self.__rotation_matrix = pyrr.matrix44.create_from_eulers(rotation)
        self.__rotation_matrix_valid = True
        self.__scale_matrix = pyrr.matrix44.create_from_scale(scale)
        self.__scale_matrix_valid = True

        self.__matrix = self.__trans_matrix @ self.__rotation_matrix @ self.__scale_matrix

        self.__children: Set[GameObject] = set()
        self.__parent: Union[GameObject, None] = None
        self.name = name

    def __update_matrices(self):
        if not self.__trans_matrix_valid:
            self.__trans_matrix = pyrr.matrix44.create_from_translation(self.__translation)
            self.__trans_matrix_valid = True
        if not self.__rotation_matrix_valid:
            self.__rotation_matrix = pyrr.matrix44.create_from_eulers(self.__rotation)
            self.__rotation_matrix_valid = True
        if not self.__scale_matrix_valid:
            self.__scale_matrix = pyrr.matrix44.create_from_scale(self.__scale)
            self.__scale_matrix_valid = True

    @property
    def matrix(self):
        if self.__trans_matrix_valid and self.__rotation_matrix_valid and self.__scale_matrix_valid:
            return self.__matrix
        else:
            self.__update_matrices()
            self.__matrix = self.__trans_matrix @ self.__rotation_matrix @ self.__scale_matrix
            return self.__matrix

    @property
    def transformation_matrix(self):
        if self.__parent is None:
            return self.matrix
        else:
            return self.__parent.matrix @ self.matrix

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, obj):
        last_parent = self.__parent
        self.__parent = obj
        if last_parent is not None:
            last_parent.remove_child(self)
        if self.__parent is not None:
            self.__parent.add_child(self)

    @property
    def children(self):
        return self.__children

    def add_child(self, child):
        self.__children.add(child)
        if child.parent != self:
            child.parent = self

    def remove_child(self, child):
        child in self.__children and self.__children.remove(child)
        if child.parent == self:
            child.parent = None

    def get_descendants(self):
        descendants = list()
        unchecked_children = list(self.children)
        while unchecked_children:
            child = unchecked_children.pop()
            descendants.append(child)
            unchecked_children.extend(child.children)
        return set(descendants)

    @property
    def translation(self):
        return self.__translation

    @translation.setter
    def translation(self, val):
        self.__translation = val
        self.__trans_matrix_valid = False

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, val):
        self.__scale = val
        self.__scale_matrix_valid = False

    @property
    def rotation(self):
        return self.__rotation

    @rotation.setter
    def rotation(self, val):
        self.__rotation = val
        self.__rotation_matrix_valid = False
