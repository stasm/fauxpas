from common import *
# ---- FXPS dissection
def fxps_norise(T, uv, L, ds, N):
    h0 = T.center(uv)[..., 3]; sh = np.zeros_like(h0)
    for i in range(1, N + 1):
        t = (i / N) ** 0.5
        sh = np.maximum(sh, (T.sample(uv + L[:2] * ds * t)[..., 3] - h0) / i)
    return sat(1 - sh * N)
def fxps_fixeddir(T, uv, L, ds, N):   # light forced to 30deg elevation, same azimuth: is elevation even used?
    Lh, cot = geom(L); L2 = np.array([Lh[0]*0.866, Lh[1]*0.866, 0.5])
    return lab.fxps(T, uv, L2, ds, N)
run("fxps", lab.fxps, Ns=(3,4)); run("fxps rise=0", fxps_norise, Ns=(3,4)); run("fxps elev frozen 30", fxps_fixeddir, Ns=(3,4))
per_tex("fxps", lab.fxps, 4)
# ---- unsharp / directional difference at FIXED distance (not exit-scaled)
def unsharp(c=0.5, lod=2.0, gain=4.0, rise=1.0, cotmax=3.0):
    def f(T, uv, L, ds, N):
        h0 = T.center(uv)[..., 3]; Lh, cot = geom(L)
        d = c * ds * min(cot, cotmax)
        s = T.sample(uv + Lh * d, lod)[..., 3]
        return 1 - sat(gain * (s - h0 - rise * d / (ds * cot)))
    return f
for c in (0.25, 0.5):
    for lod in (0.0, 2.0, 3.5):
        for gain in (3, 8):
            run(f"unsharp c{c} lod{lod} g{gain}", unsharp(c, lod, gain))
# ---- cavity AO only (1 fetch at centre, high lod)
def cavity(lod=4.0, gain=2.0):
    def f(T, uv, L, ds, N):
        h0 = T.center(uv)[..., 3]
        return 1 - sat(gain * (T.sample(uv, lod)[..., 3] - h0))
    return f
for lod in (3.0, 5.0):
    for g in (1, 2):
        run(f"cavity lod{lod} g{g}", cavity(lod, g))
# ---- VSM-style: mean from coarse lod, variance from (fine - coarse)^2 at the same spot (2 fetches / position)
def vsm(frac=0.5, lod=3.0, dl=2.0, vmin=1e-3, exitscaled=True):
    def f(T, uv, L, ds, N):
        h0 = T.center(uv)[..., 3]; Lh, cot = geom(L)
        d = frac * ds * cot * (1 - h0)
        p = uv + Lh * d[..., None]
        mu = T.sample(p, lod)[..., 3]; fine = T.sample(p, max(lod - dl, 0))[..., 3]
        var = (fine - mu) ** 2 + vmin
        t = h0 + frac * (1 - h0)
        pch = var / (var + np.maximum(t - mu, 0) ** 2)      # Chebyshev upper bound on P(h > t)
        return 1 - np.where(mu >= t, 1.0, pch)
    return f
for lod in (2.0, 3.0, 4.0):
    for vmin in (1e-3, 1e-2):
        run(f"vsm lod{lod} vmin{vmin}", vsm(lod=lod, vmin=vmin))
# constant-variance Chebyshev, 1 fetch
def cheb1(frac=0.5, beta=0.5, var=0.01):
    def f(T, uv, L, ds, N):
        h0 = T.center(uv)[..., 3]; Lh, cot = geom(L)
        d = frac * ds * cot * (1 - h0)
        mu = T.sample(uv + Lh * d[..., None], lodof(T, d * beta))[..., 3]
        t = h0 + frac * (1 - h0)
        return 1 - np.where(mu >= t, 1.0, var / (var + (t - mu) ** 2))
    return f
for beta in (0.25, 0.5, 1.0):
    for var in (0.003, 0.01, 0.03):
        run(f"cheb1 b{beta} var{var}", cheb1(beta=beta, var=var))
