from common import *

def gs(T, uv, L, ds, N, p=1.5, off=0.5, W=None, radius=0.05, centre=True, soft=0.25,
       secant=0.0, lin=0.0, kscale=1.0, origin=False, h0bias=0.0, wmin=0.0, eps=0.02):
    c, h0, d, m = geom(T, uv, L, ds)
    h0 = h0 + h0bias
    sx = np.maximum(1 - h0, 0) / m
    hl = float(np.hypot(L[0], L[1]))
    if W is None: W = max(2 * radius / max(L[2] * hl, 1e-3), wmin)   # relative-tangent width of the light disc
    occ = np.full_like(h0, -1.0)
    sa = np.zeros_like(h0); ca = np.zeros_like(h0); ha = np.zeros_like(h0)
    if origin:
        occ = np.maximum(occ, slope_along(T, c, d) * kscale / m - 1)
    for i in range(1, N + 1):
        s = sx * ((i - off) / N) ** p
        t = T.sample(uv + d * s[..., None])
        cb = t[..., 3] - h0 - m * s
        occ = np.maximum(occ, (t[..., 3] - h0) / np.maximum(m * s, eps) - 1)
        dl = s - sa
        dh = t[..., 3] - h0
        if secant > 0 and i > 1:        # k-free: extend the chord through the last two samples forward
            sl = (dh - ha) / np.maximum(dl, 1e-5)
            w = secant * dl
            occ = np.maximum(occ, (dh + sl * w) / np.maximum(m * (s + w), eps) - 1)
        if lin > 0:                     # Hermite: tangent line from the sample's normal, both sides
            g = slope_along(T, t, d) * kscale
            w = lin * dl
            occ = np.maximum(occ, (dh - g * w) / np.maximum(m * (s - w), eps) - 1)
            occ = np.maximum(occ, (dh + g * w) / np.maximum(m * (s + w), eps) - 1)
        ha = dh
        sa, ca = s, cb
    if centre:
        return np.clip(0.5 - occ / W, 0, 1)
    return np.clip(1 - occ / soft, 0, 1)
