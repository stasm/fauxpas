from synth import *
def hz2(T, uv, L, ds, N, smax=0.02, r=0.05, dist='mid', fade=1.0, fpow=2, combine='min'):
    h0 = T.center(uv)[..., 3]
    lxy = max(math.hypot(L[0], L[1]), 1e-4)
    texit = ds * (1.0 - h0) / L[2]
    tcap = N * smax / lxy if smax else 1e9
    tmax = np.minimum(texit, tcap)
    c = tmax / np.maximum(texit, 1e-6)            # 1 = ray reaches the volume exit
    c = np.where(texit < 1e-6, 1.0, c)
    k = lxy / r
    vis = np.ones_like(h0)
    for i in range(1, int(N) + 1):
        t = dist_t(i, N, dist)
        tau = tmax * t
        h = T.sample(uv + L[:2] * tau[..., None])[..., 3]
        x = (L[2] - ds * (h - h0) / np.maximum(tau, 1e-6)) * k
        v = np.clip(0.5 + 0.5 * x, 0, 1)
        w = 1.0 - fade * (1.0 - c) * t ** fpow
        vis = np.minimum(vis, 1.0 - w * (1.0 - v))
    return vis
def mk2(**kw): return lambda T, uv, L, ds, N: hz2(T, uv, L, ds, N, **kw)
def stabT(fn, N, T, elev, a0=25.0, step=0.5, n=9, ds=0.08):
    uv = T.uv_grid(); fr = []
    for j in range(n):
        L = lab.light(elev, a0 + j * step)
        fr.append(lab.ndotl(T, uv, L) * np.clip(fn(T, uv, L, ds, N), 0, 1))
    fr = np.array(fr); d2 = np.abs(np.diff(fr, n=2, axis=0))
    return d2.mean() * 1e3, np.percentile(d2, 99.9)
def synth_scores(fn, N):
    out = []
    for kind in ['posts', 'ridges']:
        T = SynTex(kind, make(kind))
        s = [score(fn, N, T, e, a) for e in (15, 30, 50) for a in (25, 140)]
        out.append('%s lit=%.3f blur=%.3f pop15=%.2f/%.3f' % ((kind, np.mean([x['lit'] for x in s]), np.mean([x['blur'] for x in s])) + stabT(fn, N, T, 15)))
    return ' | '.join(out)
def full(name, fn, Ns=(4,), detail=True, syn=True):
    for N in Ns:
        m = lab.evaluate(fn, N); print(lab.fmt('%s N=%d' % (name, N), m), flush=True)
        if detail: print('    ' + per_elev(m)); print('    ' + per_tex(m))
        if syn: print('    synth: ' + synth_scores(fn, N), flush=True)
