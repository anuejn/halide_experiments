import halide as hl


class Lightness:
    def __init__(self, input):
        self.func = hl.Func("lightness")
        self.x, self.y = hl.Var("x"), hl.Var("y")

        if input.dimensions() == 3:
            self.func[self.x, self.y] = (
                input[self.x, self.y, 0] * hl.f32(0.2126)
                + input[self.x, self.y, 1] * hl.f32(0.7152)
                + input[self.x, self.y, 2] * hl.f32(0.0722)
            )


class Div:
    def __init__(self, a, b):
        self.func = hl.Func("div")
        self.indices = [hl.Var() for _ in range(max(a.dimensions(), b.dimensions()))]

        self.func.__setitem__(
            self.indices,
            a.__getitem__(self.indices[: a.dimensions()])
            / (b.__getitem__(self.indices[: b.dimensions()]) / 2**11),
        )
