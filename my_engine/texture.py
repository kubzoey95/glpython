from my_engine.component import Component
import numpy as np
import pygame
from OpenGL.GL import glTexImage2D, glGenTextures, glBindTexture, glTexParameteri, GL_TEXTURE_2D, GL_RGBA, \
    GL_UNSIGNED_BYTE, GL_TEXTURE_WRAP_S, GL_REPEAT, GL_TEXTURE_WRAP_T, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, \
    GL_LINEAR, glBindFramebuffer, glFramebufferTexture2D, GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, glFramebufferRenderbuffer, \
    GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, glActiveTexture, GL_TEXTURE0


class Texture(Component):
    default_texture = None

    def __init__(self, obj=None, image=None, settings=((GL_TEXTURE_WRAP_S, GL_REPEAT),
                                                       (GL_TEXTURE_WRAP_T, GL_REPEAT),
                                                       (GL_TEXTURE_MIN_FILTER, GL_LINEAR),
                                                       (GL_TEXTURE_MAG_FILTER, GL_LINEAR)),
                 texture=None, dont_send=False):
        super().__init__('Texture', obj)
        self.__settings = settings
        self.__image = image
        self.__image_width, self.__image_height = image.get_rect().size if image else (None, None)
        self.__img_data = image and pygame.image.tostring(image, "RGBA")
        self.__texture = texture
        self.dont_send=dont_send

    def load_from_file(self, path):
        image = pygame.image.load(path)
        image = pygame.transform.flip(image, False, True)
        self.__image_width, self.__image_height = image.get_rect().size
        self.__img_data = pygame.image.tostring(image, "RGBA")

    def load_settings(self):
        for setting in self.__settings:
            glTexParameteri(GL_TEXTURE_2D, *setting)

    @classmethod
    def bind_default_texture(cls):
        if cls.default_texture is None:
            cls.default_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, cls.default_texture)

    @classmethod
    def get_default_texture(cls):
        if cls.default_texture is None:
            cls.default_texture = glGenTextures(1)
        cls.bind_default_texture()
        return cls.default_texture

    @property
    def texture(self):
        if self.__texture is None:
            return self.default_texture
        else:
            return self.__texture

    def bind_texture(self):
        # glActiveTexture(GL_TEXTURE0 + self.texture)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    @texture.setter
    def texture(self, val):
        self.__texture = val

    def send_texture(self):
        self.bind_texture()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.__image_width, self.__image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.__img_data)

    def send_img_data(self, img_data=None, width=1280, height=720):
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    def bind_to_frame_depth_buffer(self, frame_buffer, depth_buffer=None):
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0)
        if depth_buffer is not None:
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
