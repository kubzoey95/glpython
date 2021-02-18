from my_engine.component import Component


class Instancing(Component):
    def __init__(self, mesh=None, obj=None, instancing_data=None):
        super().__init__('Instancing', obj)
        if mesh is None:
            if 'Mesh' in obj.components:
                self.__mesh = obj.components['Mesh']
            else:
                self.__mesh = None
        else:
            self.__mesh = mesh

        if instancing_data is None:
            self.__instancing_data = {}
        else:
            self.__instancing_data = instancing_data
