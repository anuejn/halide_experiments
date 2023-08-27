from pathlib import Path
from typing import Any, List
from halide_blocks.byte import Reshape, To8Bit, Unpack
from halide_blocks.debayer import Debayer
from halide_blocks.filter import Conv1D, Gauss
import halide as hl
import halide.imageio
import numpy as np
from timeit import repeat
from pathlib import Path

from util import Div, Lightness

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
