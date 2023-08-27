# what we want
# * gauss
# * bilateral
# * guided
# * WLS

import math
import halide as hl


def gauss(size: int, std: float):
    func = hl.Func("gauss")
    x = hl.Var("x")
    average = (size-1) / 2
    def gauss(x):
        return hl.f32(math.e) ** (-0.5*(hl.f32(x) - average)**2 / float(std)**2)
    
    summed = sum(gauss(i) for i in range(size))
    func[x] = gauss(x) / summed
    return func

def conv_1D(input, kernel, size, axis=0):
    func = hl.Func("conv1d")    
    indices = [hl.Var() for _ in range(input.dimensions())]
    summed = 0
    for i in range(size):
        ind = indices.copy()
        ind[axis] = indices[axis] + i - size // 2
        v = hl.cast(hl.Float(32), input.__getitem__(ind))
        summed = summed + v * kernel[i]

    func.__setitem__(indices, summed)
    return func
