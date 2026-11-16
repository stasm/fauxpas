from common import *
def cone(fracs, beta=0.5, kappa=0.25, b=0.0, comb='max', minrel=0.05):
    """Samples at fractions of the exit distance; mip footprint = beta * distance;
    occlusion ramp width proportional to distance (kappa)."""
    def f(T, uv, L, ds, N):
        h0 = T.center(uv)[..., 3]; Lh, cot = geom(L)
        H = np.maximum(1 - h0, minrel)
        dexit = ds * cot * (1 - h0)
        acc = np.zeros_like(h0)
        for fr in fracs:
            d = fr * dexit
            s = T.sample(uv + Lh * d[..., None], lodof(T, d * beta) if beta > 0 else 0.0)[..., 3]
            e = s - h0 - fr * (1 - h0)
            o = sat(b + e / (kappa * fr * H))
            acc = np.maximum(acc, o) if comb == 'max' else acc + o
        return 1 - sat(acc)
    return f
