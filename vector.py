import math
eps = 10**-3

class Vector:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
        self.length = math.sqrt((self.x)**2 + (self.y)**2)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def normalized(self):
        if self.length < eps:
            self.length = 1.0
        return Vector(self.x / self.length, self.y / self.length)

    @staticmethod
    def reflect(v, normal):
        vNorm = v.normalized()
        angle = math.acos(vNorm.dot(normal))
        x = v.length * math.sin(angle)
        y = v.length * math.cos(angle)
        return Vector(x, y)