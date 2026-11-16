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
for g in (4,6,8,12):
    for fr in (0.4,0.5,0.6):
        run(f"1s f{fr} g{g}", multi([fr], g))
    run(f"1s f.5 g{g} nonorm", multi([0.5], g, norm=False))
for fr in ([0.25,0.6],[0.33,0.67],[0.2,0.5],[0.3,0.75],[0.15,0.5]):
    for g in (4,8):
        for comb in ('max','sum'):
            run(f"2s {fr} g{g} {comb}", multi(fr, g, comb=comb))
for fr in ([0.2,0.45,0.75],[0.25,0.5,0.75],[0.12,0.35,0.7]):
    for g in (4,8):
        for comb in ('max','sum'):
            run(f"3s {fr} g{g} {comb}", multi(fr, g, comb=comb))
for fr in ([0.2,0.4,0.6,0.8],[0.1,0.25,0.5,0.8]):
    for g in (4,8):
        for comb in ('max','sum'):
            run(f"4s {fr} g{g} {comb}", multi(fr, g, comb=comb))
