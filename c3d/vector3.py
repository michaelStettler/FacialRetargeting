class Vector3(object):
    def __init__(self, newX=None, newY=None, newZ=None):
        self.__x = 0.0 or float(newX)
        self.__y = 0.0 or float(newY)
        self.__z = 0.0 or float(newZ)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def z(self):
        return self.__z

    def setX(self, newX):
        self.__x = float(newX)

    def setY(self, newY):
        self.__y = float(newY)

    def setZ(self, newZ):
        self.__z = float(newZ)