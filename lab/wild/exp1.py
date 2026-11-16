from common import *
# --- 0 fetch: height prior
def zero(k=1.0, p=1.0):
    def f(T, uv, L, ds, N):
        h0 = T.center(uv)[..., 3]; Lh, cot = geom(L)
        return 1 - sat(k * (1 - h0) * min(ds * cot * p, 1.0))
    return f
for k in (0.5,1.0):
    for p in (2,4,8):
        run(f"zero k{k} p{p}", zero(k,p))
# --- 1 fetch: directional difference of heights against a prefiltered sample
def dir1(frac=0.5, beta=0.5, gain=2.0, fixed=None):
    def f(T, uv, L, ds, N):
        h0 = T.center(uv)[..., 3]; Lh, cot = geom(L)
        dexit = ds * cot * (1 - h0)
        d = frac * dexit if fixed is None else np.full_like(h0, fixed*ds*min(cot,3))
        s = T.sample(uv + Lh * d[..., None], lodof(T, d * beta))[..., 3]
        e = s - h0 - d / (ds * cot)
        return 1 - sat(gain * e / np.maximum(1 - h0, 0.05)) if fixed is None else 1 - sat(gain*e)
    return f
for frac in (0.3,0.5,0.7):
    for beta in (0.0,0.5,1.0):
        for gain in (2,4):
            run(f"dir1 f{frac} b{beta} g{gain}", dir1(frac,beta,gain))
run("fxps", lab.fxps, Ns=(1,2,3,4,6))
