class Component:
    def __init__(self, name: str, obj=None):
        self.__name: str = name
        self.__obj = obj
        if self.__obj is not None:
            self.__obj.add_component(self)

    @property
    def obj(self):
        return self.__obj

    @property
    def name(self):
        return self.__name

    def start(self):
        pass

    def update(self):
        pass
