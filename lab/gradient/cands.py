from common import *

def positions(sx, i, N, p, off):
    return sx * ((i - off) / N) ** p

def herm(T, uv, L, ds, N, p=1.5, soft=0.25, off=0.5, origin=True, mid=True, points=True,
         kscale=1.0, lin=0.0, first_mid=True, robust=False):
    """Point samples + Hermite midpoint virtual samples (value+slope from normals)."""
    c, h0, d, m = geom(T, uv, L, ds)
    sx = (1 - h0) / m
    g0 = slope_along(T, c, d) * kscale
    occ = np.zeros_like(h0)
    if origin:                       # limit of excess as s->0: tangent plane vs ray
        occ = np.maximum(occ, g0 / m - 1)
    sa = np.zeros_like(h0); ca = np.zeros_like(h0); da = g0 - m
    for i in range(1, N + 1):
        s = positions(sx, i, N, p, off)
        t = T.sample(uv + d * s[..., None])
        cb = t[..., 3] - h0 - m * s
        db = slope_along(T, t, d) * kscale - m
        if points:
            occ = np.maximum(occ, cb / np.maximum(m * s, 1e-4))
        if mid and (i > 1 or first_mid):
            dl = s - sa
            cm = 0.5 * (ca + cb) + dl * (da - db) * 0.125
            if robust:   # never let the virtual sample exceed what the endpoint slopes alone allow
                cm = np.minimum(cm, np.maximum(ca, cb) + 0.25 * dl * np.maximum(np.abs(da), np.abs(db)))
            occ = np.maximum(occ, cm / np.maximum(m * 0.5 * (sa + s), 1e-4))
        if lin > 0:                  # linear extrapolation either side of the sample
            w = lin * (s - sa)
            occ = np.maximum(occ, (cb - db * w) / np.maximum(m * (s - w), 1e-4))
            occ = np.maximum(occ, (cb + db * w) / np.maximum(m * (s + w), 1e-4))
        sa, ca, da = s, cb, db
    return np.clip(1 - occ / soft, 0, 1)

def origin_only(T, uv, L, ds, N, soft=0.25, kscale=1.0):
    c, h0, d, m = geom(T, uv, L, ds)
    g0 = slope_along(T, c, d) * kscale
    return np.clip(1 - (g0 / m - 1) / soft, 0, 1)

def adaptive(T, uv, L, ds, N, soft=0.25, alpha=1.0, use_nz=True, first=0.125, smin=0.05, newton=False):
    """Premecz-style: step forward by the horizontal distance over which the ray gains the
    current clearance (times n.z), floored. Soft estimate = max relative horizon excess."""
    c, h0, d, m = geom(T, uv, L, ds)
    sx = (1 - h0) / m
    s = sx * first
    occ = np.zeros_like(h0)
    for i in range(N):
        t = T.sample(uv + d * s[..., None])
        cb = t[..., 3] - h0 - m * s
        occ = np.maximum(occ, cb / np.maximum(m * s, 1e-4))
        nz = np.clip(t[..., 2] * 2 - 1, 0, 1) if use_nz else 1.0
        if newton:
            g = slope_along(T, t, d)
            step = np.where(g > m, -cb / np.maximum(g - m, 1e-3), sx)   # tangent line meets ray
            step = np.clip(step * alpha, smin * sx, 0.5 * sx)
        else:
            step = np.maximum(alpha * np.maximum(-cb, 0) / m * nz, smin * sx)
        s = s + step
    return np.clip(1 - occ / soft, 0, 1)
