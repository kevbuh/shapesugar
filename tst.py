from typing import NewType, TypeVarTuple, Self

Shape = TypeVarTuple("Shape")

class Array[*Shape]:
    def __init__(self, shape: tuple[*Shape]): self._shape: tuple[*Shape] = shape
    def __abs__(self: Self) -> Self: return self
    def __add__(self: Self, other: Self) -> Self: return self

Height = NewType('Height', int)
Width = NewType('Width', int)

shape = (Height(480), Width(640))
x: Array[Height, Width] = Array(shape)
print(x._shape)

y = abs(x)  # Array[Height, Width]
z = x + x   # Array[Height, Width]

print(y._shape)
print(z._shape)