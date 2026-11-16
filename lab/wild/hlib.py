from common import *
def horizon(a=0.35, beta=0.25, w0=0.05, kappa=0.1, comb='sum', lodmode='dist', p=1.0, lodbias=0.0):
    def f(T, uv, L, ds, N):
        h0 = T.center(uv)[..., 3]; Lh, cot = geom(L)
        H = 1 - h0
        dexit = ds * cot * H
        acc = np.zeros_like(h0)
        for i in range(1, int(N) + 1):
            fr = ((i - a) / N) ** p
            d = fr * dexit
            foot = beta * d if lodmode == 'dist' else beta * dexit / N
            lod = (lodof(T, foot) + lodbias) if beta > 0 else 0.0
            s = T.sample(uv + Lh * d[..., None], lod)[..., 3]
            e = s - h0 - fr * H
            o = sat(e / (np.maximum(H, 0.02) * (w0 + kappa * fr)))
            acc = np.maximum(acc, o) if comb == 'max' else acc + o
        return 1 - sat(acc)
    return f
