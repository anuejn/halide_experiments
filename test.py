from pathlib import Path
from typing import Any
import halide as hl
import halide.imageio
import numpy as np
from timeit import repeat

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
    vars = []

    def __init__(self, input):
        self.vars = [hl.Var() for _ in range(input.dimensions())]
        value = input.__getitem__(self.vars) >> 4
        value = hl.min(value, 255.0)
        value = hl.cast(hl.UInt(8), value)
        self.func.__setitem__(self.vars, value)

class Debayer:
    func = hl.Func("debayer")
    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")

    def __init__(self, input, cfa="RGBG", output_order="RGB"):
        self.func[self.x, self.y, self.c] = hl.cast(hl.UInt(16), 0)
        for i, c in enumerate(output_order):
            idx = cfa.index(c)
            color_x = idx % 2
            color_y = idx // 2
            x_pos = self.x % 2
            y_pos = self.y % 2
            self.func[self.x, self.y, i] = input[self.x + color_x - x_pos, self.y + color_y - y_pos]

image_bytes = np.fromfile(Path("~/data/axiom_raw/Darkbox-Timelapse-Clock-Sequence/tl-00.raw12").expanduser(), dtype='uint8')
input_buffer = hl.Buffer(image_bytes)
output = hl.Buffer(hl.UInt(8), [4096, 3072, 3])


unpacked = Unpack(input_buffer)
reshaped = Reshape(unpacked.func)
debayered = Debayer(reshaped.func, cfa="RGGB")
in_8bit = To8Bit(debayered.func)

### schedule
x_outer, x_inner, y_outer, y_inner = hl.Var("x_outer"), hl.Var("x_inner"), hl.Var("y_outer"), hl.Var("y_inner")

unpacked.func.store_at(in_8bit.func, y_outer)
unpacked.func.compute_at(in_8bit.func, y_outer)
unpacked.func.vectorize(unpacked.i, 128)
in_8bit.func.tile(in_8bit.vars[0], in_8bit.vars[1], x_outer, y_outer, x_inner, y_inner, 256, 8)
in_8bit.func.vectorize(x_inner)
in_8bit.func.parallel(y_outer)



in_8bit.func.realize(output)
print(1 / (min(repeat(lambda: in_8bit.func.realize(output), number=10, repeat=100)) / 10), "fps")

halide.imageio.imwrite("output.png", output)