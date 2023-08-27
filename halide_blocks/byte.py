import halide as hl


def unpack(input):
    func = hl.Func("unpack")
    x, y = hl.Var("x"), hl.Var("y")
    a = hl.cast(hl.UInt(16), input[x / 2 * 3 + 0, y])
    b = hl.cast(hl.UInt(16), input[x / 2 * 3 + 1, y])
    c = hl.cast(hl.UInt(16), input[x / 2 * 3 + 2, y])
    func[x, y] = hl.select(
        x % 2 == 0, 
        (a << 4) | (b & 0xf0),
        ((b & 0x0f) << 8) | c,
    )
    return func

def to_8bit(input):
    func = hl.Func("to_8bit")
    indices = [hl.Var() for _ in range(input.dimensions())]
    value = hl.cast(hl.UInt(16), input.__getitem__(indices)) >> 4
    value = hl.min(value, 255.0)
    value = hl.cast(hl.UInt(8), value)
    func.__setitem__(indices, value)
    return func
