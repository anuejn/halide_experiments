import halide as hl

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

