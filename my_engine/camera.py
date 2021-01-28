from my_engine.component import Component


class Camera(Component):
    def __init__(self, obj):
        super().__init__('Camera', obj)

    @property
    def matrix(self):
        return