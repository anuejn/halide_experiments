from typing import List
import halide as hl

class Unpack:
    func = hl.Func("unpack")
    i = hl.Var("i")

    def __init__(self, input):
        i = hl.max(0, self.i)
        a = hl.cast(hl.UInt(16), input[i / 2 * 3 + 0])
        b = hl.cast(hl.UInt(16), input[i / 2 * 3 + 1])
        c = hl.cast(hl.UInt(16), input[i / 2 * 3 + 2])
        self.func[self.i] = hl.select(
            i % 2 == 0, 
            (a << 4) | (b & 0xf0),
            ((b & 0x0f) << 8) | c,
        )


class Reshape:
    func = hl.Func("resape")
    x, y = hl.Var("x"), hl.Var("y")

    def __init__(self, input):
        self.func[self.x, self.y] = input[
            hl.max(0, hl.min(self.y, 3071)) * 4096 + 
            hl.max(0, hl.min(self.x, 4095)) % 4096
        ]

class To8Bit:
    func = hl.Func("to_8bit")
    indices: List[hl.Var]

    def __init__(self, input):
        self.indices = [hl.Var() for _ in range(input.dimensions())]
        value = hl.cast(hl.UInt(16), input.__getitem__(self.indices)) >> 4
        value = hl.min(value, 255.0)
        value = hl.cast(hl.UInt(8), value)
        self.func.__setitem__(self.indices, value)

