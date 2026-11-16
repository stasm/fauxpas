"""Best candidate from lab/gradient: relative-horizon shadow ("RHS").

rhs_shadow          k-free, N fetches, uses only heights.            <- recommended
rhs_hermite_shadow  same + tangent-line virtual samples from each fetched normal; needs the
                    signed per-texture gain K (dh/duv = K * -n.xy/n.z), see calib.py.
"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, lab

LIGHT_RADIUS = 0.05     # angular radius of the light (rad): the only look parameter
RISE_EPS = 0.02         # min ray rise (normalised height) used in the ratio; ~5 LSB of an 8-bit height
K = {'bump': 8.4, 'bricks': 50.0, 'rocks': -9.9}

def rhs_shadow(T, uv, L, ds, N, hermite=0.0):
    c = T.center(uv); h0 = c[..., 3]
    hl = max(float(np.hypot(L[0], L[1])), 1e-4)
    d = L[:2] / hl
    m = L[2] / (hl * ds)                    # ray rise, normalised height per unit uv
    sx = (1.0 - h0) / m                     # uv distance at which the ray leaves the volume
    W = 2.0 * LIGHT_RADIUS / max(L[2] * hl, 1e-3)   # light disc width in units of tan(elevation)
    ratio = np.zeros_like(h0)               # max over samples of tan(horizon)/tan(light)
    ds_ = sx / N
    for i in range(1, int(N) + 1):
        s = ds_ * (i - 0.5)                 # midpoints of N equal strata of [0, exit]
        t = T.sample(uv + d * s[..., None])
        dh = t[..., 3] - h0
        ratio = np.maximum(ratio, dh / np.maximum(m * s, RISE_EPS))
        if hermite > 0:                     # tangent line at the sample, evaluated +-w along the ray
            n = t[..., :3] * 2 - 1
            g = K[T.name] * (-(n[..., 0] * d[0] + n[..., 1] * d[1]) / np.maximum(n[..., 2], 0.05))
            w = hermite * ds_
            ratio = np.maximum(ratio, (dh - g * w) / np.maximum(m * (s - w), RISE_EPS))
            ratio = np.maximum(ratio, (dh + g * w) / np.maximum(m * (s + w), RISE_EPS))
    return np.clip(0.5 - (ratio - 1.0) / W, 0.0, 1.0)

def rhs_hermite_shadow(T, uv, L, ds, N):
    return rhs_shadow(T, uv, L, ds, N, hermite=0.2)

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    from common import detail
    for N in (2, 3, 4, 6):
        detail("rhs", rhs_shadow, N)
        detail("rhs_hermite(oracle K)", rhs_hermite_shadow, N)
    fns = {'rhs': rhs_shadow, 'rhs_hermite': rhs_hermite_shadow, 'fxps': lab.fxps}
    for tex, e, a in (('bump', 15, 265), ('rocks', 15, 25), ('bricks', 30, 140), ('rocks', 50, 140)):
        lab.save_sheet(os.path.join(here, f'final_{tex}_e{e}.png'), fns, tex=tex, elev=e, azim=a, N=4, scale=2)
