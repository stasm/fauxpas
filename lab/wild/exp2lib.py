from common import *
def multi(fracs, gain=4.0, beta=0.0, comb='max', norm=True, gains=None):
    def f(T, uv, L, ds, N):
        h0 = T.center(uv)[..., 3]; Lh, cot = geom(L)
        dexit = ds * cot * (1 - h0)
        acc = np.zeros_like(h0)
        for i, fr in enumerate(fracs):
            d = fr * dexit
            s = T.sample(uv + Lh * d[..., None], lodof(T, d * beta) if beta > 0 else 0.0)[..., 3]
            e = s - h0 - fr * (1 - h0)
            g = gain if gains is None else gains[i]
            o = sat(g * e / np.maximum(1 - h0, 0.05)) if norm else sat(g * e)
            acc = np.maximum(acc, o) if comb == 'max' else acc + o
        return 1 - sat(acc)
    return f
