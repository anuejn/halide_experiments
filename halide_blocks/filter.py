# what we want
# * gauss
# * bilateral
# * guided
# * WLS

import math
from typing import List
import halide as hl

class Gauss:
    def __init__(self, size: int, std: float):
        self.func = hl.Func("gauss")
        self.x = hl.Var("x")
        self.size = size
        average = (size-1) / 2
        def gauss(x):
            return hl.f32(math.e) ** (-0.5*(hl.f32(x) - average)**2 / float(std)**2)
        
        summed = sum(gauss(i) for i in range(size))
        self.func[self.x] = gauss(self.x) / summed

class Conv1D:
    def __init__(self, input, kernel: Gauss, axis=0):
        self.kernel = kernel
        self.func = hl.Func("conv1d")    
        self.indices = [hl.Var() for _ in range(input.dimensions())]
        summed = 0
        for i in range(kernel.size):
            indices = self.indices.copy()
            indices[axis] = indices[axis] + i - kernel.size // 2
            v = hl.cast(hl.Float(32), input.__getitem__(indices))
            summed = summed + v * kernel.func[i]

        self.func.__setitem__(self.indices, summed)
