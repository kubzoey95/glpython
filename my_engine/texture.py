from my_engine.component import Component
import numpy as np
import pygame
from OpenGL.GL import glTexImage2D, glGenTextures, glBindTexture, glTexParameteri, GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_WRAP_S, GL_REPEAT, GL_TEXTURE_WRAP_T, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR


class Texture(Component):
    def __init__(self, obj=None, image=None, settings=((GL_TEXTURE_WRAP_S, GL_REPEAT),
                                                       (GL_TEXTURE_WRAP_T, GL_REPEAT),
                                                       (GL_TEXTURE_MIN_FILTER, GL_LINEAR),
                                                       (GL_TEXTURE_MAG_FILTER, GL_LINEAR))):
        super().__init__('Texture', obj)
        self.__settings = settings
        self.__image = image
        self.__image_width, self.__image_height = image.get_rect().size
        self.__img_data = pygame.image.tostring(image, "RGBA")

    def load_from_file(self, path):
        image = pygame.image.load(path)
        image = pygame.transform.flip(image, False, True)
        self.__image_width, self.__image_height = image.get_rect().size
        self.__img_data = pygame.image.tostring(image, "RGBA")

    def load_settings(self):
        for setting in self.__settings:
            glTexParameteri(GL_TEXTURE_2D, *setting)

    @staticmethod
    def get_buffer():
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        return texture

    def send_texture(self):
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.__image_width, self.__image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.__img_data)