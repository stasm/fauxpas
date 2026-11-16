"""Horizon-formulation experiments. Parameterised candidate family."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import lab

def dist_t(i, N, kind):
    """normalised sample position in (0,1], i = 1..N"""
    x = i / N
    if kind == 'uni': return x
    if kind == 'quad': return x * x
    if kind == 'sqrt': return math.sqrt(x)
    if kind == 'geo2': return 2.0 ** (i - N)
    if kind == 'geo15': return 1.5 ** (i - N)
    if kind == 'geo3': return 3.0 ** (i - N)
    if kind == 'mid':  return (i - 0.5) / N
    if kind == 'midquad': return ((i - 0.5) / N) ** 2
    raise ValueError(kind)

def horizon(T, uv, L, ds, N, dist='quad', reach='exit', dmax=0.08, r=0.05,
            ramp='lin', combine='max', fade=False, center=0.5, kexit=1.0, hard_r=None):
    """
    slope form: q_i = ds*(h_i-h0)/tau_i  (= horizon slope * |Lxy|); light: L.z
    x_i = (L.z - q_i) * |Lxy| / r   ~ (elevation margin of light centre over horizon)/r
    vis_i = ramp(center + 0.5*x_i)  (center=0.5: disc light half covered when centre on horizon)
    """
    h0 = T.center(uv)[..., 3]
    lxy = math.sqrt(L[0] ** 2 + L[1] ** 2)
    if reach == 'exit':          # per-pixel: to volume exit
        tmax = kexit * ds * (1.0 - h0) / L[2]
    elif reach == 'exitc':       # exit, clamped to dmax (uv distance)
        tmax = np.minimum(kexit * ds * (1.0 - h0) / L[2], dmax / max(lxy, 1e-4))
    elif reach == 'full':        # constant: ds / L.z
        tmax = np.full_like(h0, ds / L[2])
    elif reach == 'fullc':
        tmax = np.full_like(h0, min(ds / L[2], dmax / max(lxy, 1e-4)))
    elif reach == 'fixed':       # constant uv distance
        tmax = np.full_like(h0, dmax / max(lxy, 1e-4))
    elif reach == 'fxps':
        tmax = np.full_like(h0, ds)
    k = lxy / r
    best = None
    prod = np.ones_like(h0); ssum = np.zeros_like(h0)
    for i in range(1, int(N) + 1):
        t = dist_t(i, N, dist)
        tau = tmax * t
        h = T.sample(uv + L[:2] * tau[..., None])[..., 3]
        q = ds * (h - h0) / np.maximum(tau, 1e-6)
        x = (L[2] - q) * k
        if fade:                 # fade far samples: weight -> 0 at t=1
            w = 1.0 - t * t if fade is True else fade(t)
            x = x * 1.0 + (1 - w) * 4.0   # push toward lit
        best = x if best is None else np.minimum(best, x)
        v = np.clip(center + 0.5 * x, 0, 1)
        if ramp == 'smooth': v = v * v * (3 - 2 * v)
        prod *= v; ssum += 1 - v
    if combine == 'max':
        v = np.clip(center + 0.5 * best, 0, 1)
        if ramp == 'smooth': v = v * v * (3 - 2 * v)
        return v
    if combine == 'prod': return prod
    if combine == 'sum': return np.clip(1 - ssum, 0, 1)

def mk(**kw):
    return lambda T, uv, L, ds, N: horizon(T, uv, L, ds, N, **kw)

def per_elev(m):
    out = []
    for e in lab.ELEVS:
        rs = [r for r in m['rows'] if r['elev'] == e]
        out.append("e%g: lit=%.3f blur=%.3f bias=%+.3f fl=%.2f" % (e, np.mean([r['rmse_lit'] for r in rs]),
                   np.mean([r['rmse_blur'] for r in rs]), np.mean([r['bias'] for r in rs]), np.mean([r['flicker'] for r in rs])))
    return " | ".join(out)

def per_tex(m):
    out = []
    for t in lab.TEXTURES:
        rs = [r for r in m['rows'] if r['tex'] == t]
        out.append("%s: lit=%.3f blur=%.3f bias=%+.3f fl=%.2f" % (t, np.mean([r['rmse_lit'] for r in rs]),
                   np.mean([r['rmse_blur'] for r in rs]), np.mean([r['bias'] for r in rs]), np.mean([r['flicker'] for r in rs])))
    return " | ".join(out)

def run(name, fn, Ns=(4,), detail=False):
    for N in Ns:
        m = lab.evaluate(fn, N)
        print(lab.fmt("%s N=%d" % (name, N), m), flush=True)
        if detail:
            print("    " + per_elev(m)); print("    " + per_tex(m))
    return m
