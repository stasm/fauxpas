import sys, math; sys.path.insert(0,'/home/user/fauxpas/lab'); import lab, numpy as np
def geom(L):
    hl = math.hypot(L[0], L[1]); Lh = np.array([L[0], L[1]]) / max(hl, 1e-6)
    return Lh, hl / max(L[2], 1e-3)          # unit uv dir toward light, cot(elevation)
def sat(x): return np.clip(x, 0.0, 1.0)
def lodof(T, d_uv): return np.log2(np.maximum(d_uv * T.size, 1.0))
def run(name, fn, Ns=(4,), **kw):
    for N in Ns:
        print(lab.fmt(f"{name} N={N}", lab.evaluate(fn, N, **kw)))
def per_tex(name, fn, N):
    m = lab.evaluate(fn, N)
    for t in lab.TEXTURES:
        for e in lab.ELEVS:
            r=[x for x in m['rows'] if x['tex']==t and x['elev']==e]
            print(f"  {name:14s} {t:6s} e={e:4.0f} lit={np.mean([x['rmse_lit'] for x in r]):.3f} blur={np.mean([x['rmse_blur'] for x in r]):.3f} bias={np.mean([x['bias'] for x in r]):+.3f} flick={np.mean([x['flicker'] for x in r]):.2f}")
