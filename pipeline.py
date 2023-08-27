from pathlib import Path
from typing import Any, List
from halide_blocks.byte import To8Bit, Unpack
from halide_blocks.debayer import Debayer
from halide_blocks.filter import Conv1D, Gauss
import halide as hl
import halide.imageio
import numpy as np
from timeit import repeat
from pathlib import Path
from halide_util import find_gpu_target

from util import Div, Lightness

image_bytes = np.fromfile(Path("~/data/axiom_raw/Darkbox-Timelapse-Clock-Sequence/tl-00.raw12").expanduser(), dtype='uint8')
input_buffer = hl.Buffer(image_bytes.reshape((-1, 4096 * 3 // 2)))
clamped = hl.BoundaryConditions.repeat_edge(input_buffer)
output = hl.Buffer(hl.UInt(8), [4096, 3072, 3])


unpacked = Unpack(clamped)
debayered = Debayer(unpacked.func, cfa="RGGB")

#lightness = Lightness(debayered.func)
#blurred1 = Conv1D(lightness.func, Gauss(10, 10), axis=0)
#blurred2 = Conv1D(blurred1.func, Gauss(10, 10), axis=1)

#div = Div(debayered.func, blurred2.func)
in_8bit = To8Bit(debayered.func)


in_8bit.func.set_estimates([
    hl.Range(4096, 4096),
    hl.Range(3072, 3072),
    hl.Range(3, 3)
])
pipeline = hl.Pipeline(in_8bit.func)

hl.load_plugin(str(Path(hl.__file__).parent.parent.parent.parent / 'libautoschedule_anderson2021.so'))
target = find_gpu_target()
# see https://halide-lang.org/docs/struct_halide_1_1_internal_1_1_autoscheduler_1_1_adams2019_params.html for autoscheduler parameters
pipeline.apply_autoscheduler(target, hl.AutoschedulerParams("Anderson2021"))

pipeline.compile_jit(target)
print(1 / (min(repeat(lambda: pipeline.realize(output), number=10, repeat=10)) / 10), "fps")
output.copy_to_host()
halide.imageio.imwrite("output.png", output)
