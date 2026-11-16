from hz2 import *
def hz3(T, uv, L, ds, N, smax=0.03, r=0.05, dist='mid', fade=1.0, fpow=2, beta=0.0, onesided=False):
    """beta: penumbra half-width floor = beta/2 * (relative slope gap to the previous sample's ray point)"""
    h0 = T.center(uv)[..., 3]
    lxy = max(math.hypot(L[0], L[1]), 1e-4)
    texit = ds * (1.0 - h0) / L[2]
    tcap = N * smax / lxy if smax else 1e9
    tmax = np.minimum(texit, tcap)
    c = np.where(texit < 1e-6, 1.0, tmax / np.maximum(texit, 1e-6))
    wr = r / (L[2] * lxy)                 # physical half-width, relative-slope units (q/L.z)
    vis = np.ones_like(h0); tprev = 0.0
    for i in range(1, int(N) + 1):
        t = dist_t(i, N, dist)
        tau = tmax * t
        h = T.sample(uv + L[:2] * tau[..., None])[..., 3]
        rel = 1.0 - ds * (h - h0) / (L[2] * np.maximum(tau, 1e-6))    # 1 - slope/lightslope ; <0 occluded
        ws = 0.5 * beta * (1.0 - tprev / t)
        if onesided: v = np.clip(np.minimum(0.5 + 0.5 * rel / wr, rel / max(2 * ws, 1e-6) if beta else 1), 0, 1)
        else: v = np.clip(0.5 + 0.5 * rel / max(wr, ws), 0, 1)
        w = 1.0 - fade * (1.0 - c) * t ** fpow
        vis = np.minimum(vis, 1.0 - w * (1.0 - v))
        tprev = t
    return vis
def mk3(**kw): return lambda T, uv, L, ds, N: hz3(T, uv, L, ds, N, **kw)
