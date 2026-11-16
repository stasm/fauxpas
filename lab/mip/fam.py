"""Parametric family of prefiltered (mip) light-ray shadows. Exploration only."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import lab

def ign(shape):
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    return np.modf(52.9829189 * np.modf(0.06711056 * x + 0.00583715 * y)[0])[0].astype(np.float32)

def fam(T, uv, L, ds, N, sched='uni', lodmode='match', lodbias=0.0, range_cap=None, range_frac=1.0,
        wmode='rise', wk=1.0, wmin=0.0, hbias=0.0, combine='min', jitter=0.0, ratio=2.0, power=2.0,
        off=0.5, shape='lin', fade=False, lodconst=0.0, pen_soft=0.0):
    N = int(N)
    c = T.center(uv); h0 = c[..., 3]
    hl = float(np.hypot(L[0], L[1])); Lz = float(L[2])
    tau_exit = ds * (1.0 - h0) / Lz                 # per pixel
    tau_max = tau_exit * range_frac
    if range_cap is not None:                      # cap on horizontal reach, in uv units
        tau_max = np.minimum(tau_max, range_cap / max(hl, 1e-6))
    j = (ign(h0.shape) - 0.5) * jitter if jitter else 0.0
    # normalised positions t_i in (0,1] and spacing s_i
    ts, ss = [], []
    for i in range(1, N + 1):
        if sched == 'uni':   t = (i - 0.5) / N; s = 1.0 / N
        elif sched == 'uni1': t = i / N; s = 1.0 / N
        elif sched == 'geo':            # cells [r^(i-1-N), r^(i-N)], sample at cell centre; first cell from 0
            hi = ratio ** (i - N); lo = ratio ** (i - 1 - N) if i > 1 else 0.0
            t = 0.5 * (hi + lo); s = hi - lo
        elif sched == 'pow':
            hi = (i / N) ** power; lo = ((i - 1) / N) ** power
            t = 0.5 * (hi + lo); s = hi - lo
        ts.append(t); ss.append(s)
    vis = np.ones_like(h0); acc = np.zeros_like(h0)
    for t, s in zip(ts, ss):
        tj = t + j * s
        tau = tau_max * tj
        sp_tex = tau_max * s * hl * T.size          # footprint length in texels (lod-0)
        if lodmode == 'match':
            lod = np.log2(np.maximum(sp_tex, 1e-3)) + lodbias
            lod = np.maximum(lod, 0.0)
        elif lodmode == 'const':
            lod = lodconst
        else:
            lod = 0.0
        h = T.sample(uv + (L[:2] * 1.0)[None, None, :] * tau[..., None], lod)[..., 3]
        rise = Lz * tau / ds                        # ray height above h0 at sample (normalised)
        pen = h + hbias * (np.maximum(lod, 0) if not np.isscalar(lod) else max(lod, 0)) - (h0 + rise)
        if wmode == 'rise':       # half the ray rise across the footprint
            w = wk * Lz * tau_max * s / ds
        elif wmode == 'dist':     # proportional to distance (cone / area light)
            w = wk * rise
        elif wmode == 'const':
            w = wk
        w = np.maximum(w, wmin) + 1e-5
        x = np.clip(pen / w + off, 0.0, 1.0)
        if shape == 'smooth': x = x * x * (3 - 2 * x)
        if fade: x = x * (1.0 - tj)
        if combine == 'min': acc = np.maximum(acc, x)
        elif combine == 'prod': vis = vis * (1 - x)
        elif combine == 'sum': acc = acc + x
    if combine == 'prod': return vis
    return 1.0 - np.clip(acc, 0, 1)

def run(name, Ns=(4,), **kw):
    for N in Ns:
        m = lab.evaluate(lambda T, uv, L, ds, n: fam(T, uv, L, ds, n, **kw), N)
        print(lab.fmt(f"{name} N={N}", m), flush=True)
    return m
