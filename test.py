from pathlib import Path
from typing import Any, List
from filter import Conv1D, Gauss
import halide as hl
import halide.imageio
import numpy as np
from timeit import repeat
from pathlib import Path

from util import Div, Lightness

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

            self.func[self.x, self.y, i] = input[
                self.x + color_x - x_pos,
                self.y + color_y - y_pos
            ]


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


image_bytes = np.fromfile(Path("~/data/axiom_raw/Darkbox-Timelapse-Clock-Sequence/tl-00.raw12").expanduser(), dtype='uint8')
input_buffer = hl.Buffer(image_bytes)
output = hl.Buffer(hl.UInt(8), [4096, 3072, 3])


unpacked = Unpack(input_buffer)
reshaped = Reshape(unpacked.func)
debayered = Debayer(reshaped.func, cfa="RGGB")

lightness = Lightness(debayered.func)
blurred1 = Conv1D(lightness.func, Gauss(10, 10), axis=0)
blurred2 = Conv1D(blurred1.func, Gauss(10, 10), axis=1)

div = Div(debayered.func, blurred2.func)

in_8bit = To8Bit(div.func)
in_8bit.func.set_estimates([
    hl.Range(4096, 4096),
    hl.Range(3072, 3072),
    hl.Range(3, 3)
])


pipeline = hl.Pipeline(in_8bit.func)

hl.load_plugin(str(Path(hl.__file__).parent.parent.parent.parent / 'libautoschedule_adams2019.so'))
pipeline.apply_autoscheduler(hl.get_target_from_environment(), hl.AutoschedulerParams("Adams2019"))

pipeline.compile_jit()
print(1 / (min(repeat(lambda: pipeline.realize(output), number=10, repeat=10)) / 10), "fps")
halide.imageio.imwrite("output.png", output)
