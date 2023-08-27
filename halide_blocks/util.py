import halide as hl


def lightness(input):
    func = hl.Func("lightness")
    x, y = hl.Var("x"), hl.Var("y")
    if input.dimensions() == 3:
        func[x, y] = (
            input[x, y, 0] * hl.f32(0.2126)
            + input[x, y, 1] * hl.f32(0.7152)
            + input[x, y, 2] * hl.f32(0.0722)
        )
    return func

def clamp(input, *clamps):
    func = hl.Func("clamp")
    indices = [hl.Var() for _ in range(input.dimensions())]
    func[indices] = input[[hl.max(hl.min(idx, max), min) for idx, (min, max) in zip(indices, clamps)]]
    return func


def div(a, b):
    func = hl.Func("div")
    indices = [hl.Var() for _ in range(max(a.dimensions(), b.dimensions()))]
    func[indices] = a[indices[: a.dimensions()]] / (b[indices[: b.dimensions()]] / 2**11)
    return func