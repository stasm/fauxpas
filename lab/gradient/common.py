import sys; sys.path.insert(0,'/home/user/fauxpas/lab')
import lab, numpy as np
K = {'bump': 8.4, 'bricks': 50.0, 'rocks': -9.9}   # measured: dh/duv = K * (-n.xy/n.z)

def summarize(name, fn, N=4, **kw):
    m = lab.evaluate(fn, N, **kw)
    print(lab.fmt(f"{name} N={N}", m))
    return m

def detail(name, fn, N=4, **kw):
    m = summarize(name, fn, N, **kw)
    r = m['rows']
    s = "   tex: " + " | ".join(f"{t} lit={np.mean([x['rmse_lit'] for x in r if x['tex']==t]):.3f} blur={np.mean([x['rmse_blur'] for x in r if x['tex']==t]):.3f} bias={np.mean([x['bias'] for x in r if x['tex']==t]):+.3f} flk={np.mean([x['flicker'] for x in r if x['tex']==t]):.2f}" for t in lab.TEXTURES)
    print(s)
    s = "   elev: " + " | ".join(f"{e:g} lit={np.mean([x['rmse_lit'] for x in r if x['elev']==e]):.3f} blur={np.mean([x['rmse_blur'] for x in r if x['elev']==e]):.3f} flk={np.mean([x['flicker'] for x in r if x['elev']==e]):.2f}" for e in lab.ELEVS)
    print(s)
    return m

def geom(T, uv, L, ds):
    c = T.center(uv); h0 = c[..., 3]
    hl = float(np.hypot(L[0], L[1])); d = L[:2] / max(hl, 1e-6)
    m = L[2] / max(hl, 1e-6) / ds          # ray rise in normalised height per unit uv
    return c, h0, d, m

def slope_along(T, s, d):
    """dh/ds along unit uv dir d from the normal of sample s (oracle per-texture K)."""
    n = s[..., :3] * 2 - 1
    return K[T.name] * (-(n[..., 0] * d[0] + n[..., 1] * d[1]) / np.maximum(n[..., 2], 0.05))

def point(T, uv, L, ds, N, p=1.0, soft=0.5, off=0.0, den=None, rel=True, cap=None, fade=0.0, grow=0.0):
    """Plain point samples, physical ray. vis from max relative horizon excess."""
    c, h0, d, m = geom(T, uv, L, ds)
    sx = (1 - h0) / m
    if cap: sx = np.minimum(sx, cap)
    den = den if den is not None else N
    occ = np.zeros_like(h0)
    for i in range(1, N + 1):
        s = sx * ((i - off) / den) ** p
        h = T.sample(uv + d * s[..., None])[..., 3]
        rise = m * s
        e = (h - h0 - rise) / np.maximum(rise, 1e-4) if rel else (h - h0 - rise) * ds / np.maximum(s, 1e-5)
        t = ((i - off) / den) ** p
        occ = np.maximum(occ, e * (1 - fade * t) / (1 + grow * t))
    return np.clip(1 - occ / soft, 0, 1)
