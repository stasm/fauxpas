"""Exit-clipped Horizon Shadow (XHS): numpy mirror of horizonShadow() in shaders.js."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, lab

def xhs(T, uv, L, ds, N, radius=0.05, max_cot=4.0, beta=0.0, min_rise=0.02, lod_ramp=0.0):
    """beta: mip footprint as a fraction of the sample spacing (0 = no mips).
    lod_ramp: widen the penumbra by this many normalised-height units per unit of
    (footprint relative to slice rise) -- only used in experiments."""
    h0 = T.center(uv)[..., 3]
    H = 1.0 - h0
    hl = max(float(np.hypot(L[0], L[1])), 1e-4)
    cot = min(hl / L[2], max_cot)
    span = (L[:2] / hl)[None, None, :] * (ds * cot * H)[..., None]
    lod = 0.0
    if beta > 0:
        lod = np.log2(np.maximum(beta * ds * cot * H / N * T.size, 1.0))
    ratio = np.zeros_like(h0)
    for i in range(int(N)):
        f = (i + 0.5) / N
        h = T.sample(uv + span * f, lod)[..., 3]
        ratio = np.maximum(ratio, (h - h0) / np.maximum(f * H, min_rise))
    sin_cos = cot / (1.0 + cot * cot)
    return np.clip(0.5 - (ratio - 1.0) * sin_cos / (2.0 * radius), 0.0, 1.0)

def dho(T, uv, L, ds, N, reach=0.06, radius=0.25, beta=0.5, min_rise=0.02):
    """Directional Horizon Occlusion: numpy mirror of directionalOcclusion() in shaders.js."""
    h0 = T.center(uv)[..., 3]
    hl = max(float(np.hypot(L[0], L[1])), 1e-4)
    cot = hl / max(L[2], 1e-3)
    ln = np.minimum(ds * cot * (1.0 - h0), reach)
    rise = ln / (ds * cot)
    span = (L[:2] / hl)[None, None, :] * ln[..., None]
    lod = np.log2(np.maximum(beta * ln / N * T.size, 1.0))
    ratio = np.zeros_like(h0)
    for i in range(int(N)):
        f = (i + 0.5) / N
        h = T.sample(uv + span * f, lod)[..., 3]
        ratio = np.maximum(ratio, (h - h0) / np.maximum(f * rise, min_rise))
    tan_diff = cot * (1.0 - ratio) / (cot * cot + ratio)
    return np.clip(0.5 + 0.5 * tan_diff / radius, 0.0, 1.0)

def variant(**kw):
    return lambda T, uv, L, ds, N: xhs(T, uv, L, ds, N, **kw)

if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    cands = {
        'nomip': variant(),
        'nomip nocap': variant(max_cot=1e3),
        'beta .5': variant(beta=0.5),
        'beta 1': variant(beta=1.0),
        'beta 1 r.1': variant(beta=1.0, radius=0.1),
        'nomip r.1': variant(radius=0.1),
    }
    for N in (2, 4, 6):
        print('--- N =', N)
        for k, f in cands.items():
            print(lab.fmt(k, lab.evaluate(f, N)))
    sel = {k: cands[k] for k in ('nomip', 'beta .5', 'beta 1', 'beta 1 r.1')}
    sel['fxps'] = lab.fxps
    for tex, e in (('bump', 15), ('bump', 30), ('bricks', 15), ('rocks', 30)):
        lab.save_sheet(f'{out}/mip_{tex}_e{e}.png', sel, tex=tex, elev=e, azim=25, N=4)
