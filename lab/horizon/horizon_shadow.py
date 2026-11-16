"""
horizonShadow -- few-sample horizon-angle shadow for height fields (best candidate of lab/horizon).

    Q    = max_i  ds*(h_i - h0) / tau_i          tau_i = tmax * (i - 0.5)/N
           (= horizon slope along the light azimuth, times |L.xy|;  Q >= 0: flat ground is the floor)
    tmax = min( ds*(1-h0)/L.z ,  N*MAX_SPACING/|L.xy| )
           (ray ends where it leaves the height volume, or earlier so that samples are
            never further apart than MAX_SPACING in uv)
    vis  = clamp( 0.5 + 0.5 * tan(elev_light - elev_horizon) / LIGHT_RADIUS )
           tan(a-b) = |L.xy| (L.z - Q) / (|L.xy|^2 + L.z Q)

Usage:  import horizon_shadow; lab.evaluate(horizon_shadow.horizon_shadow, 4)
"""
import math
import numpy as np

LIGHT_RADIUS = 0.07    # angular radius of the light (rad). Reference uses 0.05; larger = softer/more stable
MAX_SPACING = 0.03     # largest uv distance between consecutive samples (tuned: ~ smallest occluder width that must not ghost)


def horizon_shadow(T, uv, L, ds, N, r=LIGHT_RADIUS, smax=MAX_SPACING):
    N = int(N)
    h0 = T.center(uv)[..., 3]                                   # free
    if L[2] <= 0.0:
        return np.zeros_like(h0)
    lxy = max(math.sqrt(L[0] * L[0] + L[1] * L[1]), 1e-4)
    tmax = np.minimum(ds * (1.0 - h0) / L[2], N * smax / lxy)   # per pixel, outside the loop
    m = np.zeros_like(h0)                                       # max of (h_i - h0)/t_i, floor 0
    for i in range(N):
        t = (i + 0.5) / N
        h = T.sample(uv + L[:2] * (tmax * t)[..., None])[..., 3]
        m = np.maximum(m, (h - h0) * (1.0 / t))                 # 1/t is a compile-time-ish constant per i
    Q = ds * m / np.maximum(tmax, 1e-6)
    return np.clip(0.5 + 0.5 * lxy * (L[2] - Q) / (r * (lxy * lxy + L[2] * Q)), 0.0, 1.0)


def variant(**kw):
    return lambda T, uv, L, ds, N: horizon_shadow(T, uv, L, ds, N, **kw)
