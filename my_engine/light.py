from my_engine.component import Component


class Light(Component):
    def __init__(self, obj=None, constant=1., linear=1., quadratic=1., ambient=(1., 1., 1.), diffuse=(1., 1., 1.), specular=(1., 1., 1.)):
        super().__init__('Light', obj)

