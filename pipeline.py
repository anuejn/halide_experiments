from pathlib import Path
from typing import Any, List
import halide as hl
import halide.imageio
import numpy as np
from timeit import repeat
from pathlib import Path
from benchmark_util import benchmark
from halide_blocks.byte import to_8bit, unpack
from halide_blocks.debayer import debayer
from halide_blocks.filter import conv_1D, gauss
from halide_blocks.util import clamp, lightness, div
from halide_util import find_gpu_target


image_bytes = np.fromfile(Path("~/data/axiom_raw/Darkbox-Timelapse-Clock-Sequence/tl-00.raw12").expanduser(), dtype='uint8')
input_buffer = hl.Buffer(image_bytes.reshape((-1, 4096 * 3 // 2)))
input_buffer = hl.BoundaryConditions.repeat_edge(input_buffer)
output = hl.Buffer(hl.UInt(8), [4096, 3072, 3])


unpacked = unpack(input_buffer)
# unpacked = clamp(unpacked, (0, 4095), (0, 3071))
debayered = debayer(unpacked, cfa="RGGB")

gray = lightness(debayered)
kernel = gauss(10, 10)
blurred1 = conv_1D(gray, kernel, 10, axis=0)
blurred2 = conv_1D(blurred1, kernel, 10, axis=1)

div = div(debayered, blurred2)
in_8bit = to_8bit(debayered)


in_8bit.set_estimates([
    hl.Range(4096, 4096),
    hl.Range(3072, 3072),
    hl.Range(3, 3)
])
pipeline = hl.Pipeline(in_8bit)

gpu = True
scheduler = "Anderson2021" if gpu else "Adams2019"
hl.load_plugin(str(Path(hl.__file__).parent.parent.parent.parent / f'libautoschedule_{scheduler.lower()}.so'))
target = find_gpu_target()
pipeline.apply_autoscheduler(target, hl.AutoschedulerParams(scheduler))

pipeline.compile_jit(target)
benchmark(lambda: pipeline.realize(output, target))
output.copy_to_host()
halide.imageio.imwrite("output.png", output)
