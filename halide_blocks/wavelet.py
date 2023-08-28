import halide as hl

def wavelet_stage(input, shape, axis=0):
    func = hl.Func("wavelet")    
    indices = [hl.Var(chr(ord("x") + i)) for i in range(input.dimensions())]
    func[indices] = input[indices]

    lf_ind, ind0, ind1 = indices.copy(), indices.copy(), indices.copy()
    r = hl.RDom([(0, shape[axis] // 2)])
    ind0[axis] = r * 2
    ind1[axis] = r * 2 + 1
    lf_ind[axis] = r
    func[lf_ind] = input[ind0] + input[ind1] / 2

    hf_ind, ind0, ind1 = indices.copy(), indices.copy(), indices.copy()
    r = hl.RDom([(shape[axis] // 2, shape[axis])])
    ind0[axis] = (r - shape[axis] // 2) * 2
    ind1[axis] = (r - shape[axis] // 2) * 2 + 1
    hf_ind[axis] = r
    func[hf_ind] = input[ind0] - input[ind1]

    return func


def wavelet(input, shape, stages=3):
    for i in range(stages):
        s = [x // 2**i for x in shape]
        input = wavelet_stage(input, s, axis=0)
        input = wavelet_stage(input, s, axis=1)
    return input
