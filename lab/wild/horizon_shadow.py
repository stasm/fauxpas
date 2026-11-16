"""Exit-clipped horizon shadow ("XHS") -- best candidate of lab/wild.

For each of N samples, placed at the midpoints of N equal slices of the segment
of the physical light ray that lies inside the height volume, compare a
mip-prefiltered height (footprint = BETA * slice length) with the ray height and
accumulate a linear occlusion ramp.  See REPORT in the agent hand-back.
"""
import sys, math
sys.path.insert(0, '/home/user/fauxpas/lab')
import numpy as np
import lab

BETA = 0.5     # mip footprint / sample spacing (1 = gap-free prefilter, 0 = point samples)
W_REL = 0.10   # penumbra ramp width as a fraction of the head-room H = 1 - h0
W_ABS = 0.02   # absolute floor of the ramp width (normalised height) -- anti-pop fudge


def horizon_shadow(T, uv, L, ds, N, beta=BETA, w_rel=W_REL, w_abs=W_ABS):
    N = int(N)
    h0 = T.center(uv)[..., 3]                       # free
    hl = math.hypot(L[0], L[1])
    Lh = np.array([L[0], L[1]]) / max(hl, 1e-5)     # unit uv direction toward the light
    cot = hl / max(L[2], 1e-3)                      # uv travelled per unit of real height gained
    H = 1.0 - h0                                    # head-room above the shaded point
    d_exit = ds * cot * H                           # uv distance at which the ray leaves the volume
    lod = np.log2(np.maximum(beta * d_exit * T.size / N, 1.0))   # footprint = beta * spacing
    inv_w = 1.0 / (w_rel * H + w_abs)
    occ = np.zeros_like(h0)
    for i in range(N):
        f = (i + 0.5) / N                           # fraction of the in-volume ray
        h = T.sample(uv + Lh * (f * d_exit)[..., None], lod)[..., 3]
        occ += np.clip((h - h0 - f * H) * inv_w, 0.0, 1.0)      # ray height at f is h0 + f*H
    return 1.0 - np.clip(occ, 0.0, 1.0)


def variant(**kw):
    return lambda T, uv, L, ds, N: horizon_shadow(T, uv, L, ds, N, **kw)


if __name__ == "__main__":
    for N in (1, 2, 3, 4, 6):
        print(lab.fmt(f"xhs N={N}", lab.evaluate(horizon_shadow, N)))
        print(lab.fmt(f"fxps N={N}", lab.evaluate(lab.fxps, N)))
