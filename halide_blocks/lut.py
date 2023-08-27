import halide as hl

class GammaLut:
    func = hl.Func("gamma_lut")
    i = hl.Var("i")

    def __init__(self, gamma=0.5, bits=12):
        max_value = 2**bits - 1
        self.func[self.i] = hl.u16(hl.clamp(hl.pow(self.i / float(max_value), hl.f32(gamma)) * float(max_value), 0, max_value))


class ApplyLut:
    func = hl.Func("gamma_lut")
    indices: List[hl.Var]

    def __init__(self, input, lut):
        self.indices = [hl.Var() for _ in range(input.dimensions())]
        value = input.__getitem__(self.indices)
        value = lut[value]
        self.func.__setitem__(self.indices, value)
