from typing import List
import halide as hl

class Unpack:
    def __init__(self, input):
        self.func = hl.Func("unpack")
        self.x, self.y = hl.Var("x"), hl.Var("y")
        a = hl.cast(hl.UInt(16), input[self.x / 2 * 3 + 0, self.y])
        b = hl.cast(hl.UInt(16), input[self.x / 2 * 3 + 1, self.y])
        c = hl.cast(hl.UInt(16), input[self.x / 2 * 3 + 2, self.y])
        self.func[self.x, self.y] = hl.select(
            self.x % 2 == 0, 
            (a << 4) | (b & 0xf0),
            ((b & 0x0f) << 8) | c,
        )


class To8Bit:
    func = hl.Func("to_8bit")
    indices: List[hl.Var]

    def __init__(self, input):
        self.indices = [hl.Var() for _ in range(input.dimensions())]
        value = hl.cast(hl.UInt(16), input.__getitem__(self.indices)) >> 4
        value = hl.min(value, 255.0)
        value = hl.cast(hl.UInt(8), value)
        self.func.__setitem__(self.indices, value)

