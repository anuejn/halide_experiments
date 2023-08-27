import halide as hl


def debayer(input, cfa="RGBG", output_order="RGB"):
    func = hl.Func("debayer")
    x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")
    func[x, y, c] = hl.cast(hl.UInt(16), 0)
    for i, c in enumerate(output_order):
        idx = cfa.index(c)
        color_x = idx % 2
        color_y = idx // 2
        x_pos = x % 2
        y_pos = y % 2

        func[x, y, i] = input[
            x + color_x - x_pos,
            y + color_y - y_pos
        ]
    return func