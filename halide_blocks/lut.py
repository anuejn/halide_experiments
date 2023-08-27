import halide as hl


def gamma_lut(gamma=0.5, bits=12):
    func = hl.Func("gamma_lut")
    i = hl.Var("i")
    max_value = 2**bits - 1
    func[i] = hl.u16(hl.clamp(hl.pow(i / float(max_value), hl.f32(gamma)) * float(max_value), 0, max_value))
    return func

def apply_lut_1D(input, lut):
    func = hl.Func("gamma_lut")
    indices = [hl.Var() for _ in range(input.dimensions())]
    value = input.__getitem__(indices)
    value = lut[value]
    func.__setitem__(indices, value)
    return func
