"""
Shadow lab: CPU harness for prototyping few-sample height-field shadows.

Emulates the GLSL sampling environment in numpy (bilinear + trilinear mips,
REPEAT wrap), computes a brute-force area-light reference, and scores
candidate shadow functions against it.

Geometry (physical, matches pomHardShadow / getParallaxOffset in shaders.js):
    surface point   P0 = (uv, ds * h(uv))            h in [0,1], ds = depth_scale
    light ray       P(tau) = P0 + L * tau            L unit, tangent space, L.z > 0
    occluded iff    ds * h(uv + L.xy*tau) > ds*h0 + L.z*tau   for some tau > 0
    ray leaves the height volume at tau_exit = ds * (1 - h0) / L.z

Candidate signature:
    fn(T, uv, L, ds, N) -> visibility array, shape uv.shape[:-1], 1 = lit
      T   Tex object. T.sample(uv, lod=0.0) -> RGBA (..., 4) exactly like
          texture()/textureLod() on tex_norm: rgb = encoded normal (0..1),
          a = height. Every call counts as one fetch per pixel (T.fetches).
      uv  (H, W, 2) surface uv of each pixel (after parallax)
      L   (3,) unit tangent-space light direction
      N   sample budget hint (the shadow_steps uniform)
    The fetch at `uv` itself is FREE in the real shader (the shading pass
    already reads normal+height there): get it via T.center(uv), not counted.

Run `python3 lab/lab.py` for the baseline table.
"""
import os, sys, math
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
TEXTURES = ["bump", "bricks", "rocks"]
ELEVS = [15.0, 30.0, 50.0, 70.0]      # degrees above the surface plane
AZIMS = [25.0, 140.0, 265.0]          # degrees
DS_DEFAULT = 0.08                     # demo default: scale 8 -> 0.08
LIGHT_RADIUS = 0.05                   # angular radius of the reference area light (rad)
RES = 256                             # evaluation grid, 1 pixel == 1 texel


def light(elev_deg, azim_deg):
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


class Tex:
    def __init__(self, name):
        self.name = name
        img = np.asarray(Image.open(os.path.join(ROOT, name + "_normal.png")).convert("RGBA"))
        self.mips = [img.astype(np.float32) / 255.0]
        while self.mips[-1].shape[0] > 1:           # box-filtered mip chain (generateMipmap)
            m = self.mips[-1]
            self.mips.append(0.25 * (m[0::2, 0::2] + m[1::2, 0::2] + m[0::2, 1::2] + m[1::2, 1::2]))
        self.size = img.shape[0]
        self.fetches = 0

    def _bilinear(self, level, uv):
        m = self.mips[level]
        n = m.shape[0]
        x = uv[..., 0] * n - 0.5
        y = uv[..., 1] * n - 0.5
        x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
        fx = (x - x0)[..., None].astype(np.float32); fy = (y - y0)[..., None].astype(np.float32)
        x1 = (x0 + 1) % n; y1 = (y0 + 1) % n; x0 %= n; y0 %= n
        return ((m[y0, x0] * (1 - fx) + m[y0, x1] * fx) * (1 - fy)
                + (m[y1, x0] * (1 - fx) + m[y1, x1] * fx) * fy)

    def _sample(self, uv, lod):
        top = len(self.mips) - 1
        if np.isscalar(lod):
            lod = float(min(max(lod, 0.0), top))
            l0 = int(math.floor(lod)); f = lod - l0
            out = self._bilinear(l0, uv)
            if f > 0:
                out = out * (1 - f) + self._bilinear(min(l0 + 1, top), uv) * f
            return out
        lod = np.clip(np.broadcast_to(lod, uv.shape[:-1]), 0.0, top)
        l0 = np.floor(lod).astype(int); f = (lod - l0)[..., None].astype(np.float32)
        out = np.zeros(uv.shape[:-1] + (4,), np.float32)
        for lv in range(int(l0.min()), int(min(l0.max() + 1, top)) + 1):
            s = self._bilinear(lv, uv)
            out += s * np.where(l0 == lv, 1 - f[..., 0], 0)[..., None]
            out += s * np.where(l0 + 1 == lv, f[..., 0], 0)[..., None]
        return out

    def sample(self, uv, lod=0.0):
        """textureLod(tex_norm, uv, lod). Counts one fetch."""
        self.fetches += 1
        return self._sample(uv, lod)

    def center(self, uv):
        """The fetch at the shaded point: free (shading already needs it)."""
        return self._sample(uv, 0.0)

    def height(self, uv):   # uncounted, for references only
        return self._sample(uv, 0.0)[..., 3]

    def uv_grid(self, res=RES):
        """Pixel-centre uv grid covering res x res texels (1 px == 1 texel)."""
        span = res / self.size
        c = (np.arange(res) + 0.5) / res * span
        u, v = np.meshgrid(c, c)
        return np.stack([u, v], -1)


def decode_normal(rgba):
    n = rgba[..., :3] * 2.0 - 1.0
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


# ---------------------------------------------------------------- reference
def hard_reference(T, uv, L, ds, step_texels=0.5, bias=1e-3):
    """Brute-force hard shadow: fine linear march until the ray leaves the volume."""
    h0 = T.height(uv)
    Lxy = L[:2]; hlen = float(np.linalg.norm(Lxy))
    if hlen < 1e-6:
        return np.ones_like(h0)
    dtau = step_texels / T.size / hlen            # tau per step (uv step = step_texels texels)
    nsteps = int(math.ceil(ds / L[2] / dtau)) + 1
    vis = np.ones_like(h0)
    for i in range(1, nsteps + 1):
        tau = i * dtau
        ray_h = h0 + L[2] * tau / ds
        if ray_h.min() > 1.0:
            break
        h = T.height(uv + Lxy * tau)
        vis = np.where(h > ray_h + bias, 0.0, vis)
    return vis


def soft_reference(T, uv, L, ds, radius=LIGHT_RADIUS, rings=((0.0, 1), (0.5, 5), (0.9, 9))):
    """Area light: average hard shadows over a disc of directions around L."""
    up = np.array([0.0, 0.0, 1.0])
    a = np.cross(L, up); a /= np.linalg.norm(a)
    b = np.cross(L, a)
    acc = 0.0; cnt = 0
    for r, k in rings:
        for j in range(k):
            phi = 2 * math.pi * (j + 0.5 * r) / k
            d = L + radius * r * (math.cos(phi) * a + math.sin(phi) * b)
            d /= np.linalg.norm(d)
            if d[2] < 0.02:
                d[2] = 0.02; d /= np.linalg.norm(d)
            acc = acc + hard_reference(T, uv, d, ds); cnt += 1
    return acc / cnt


def cached_reference(T, elev, azim, ds=DS_DEFAULT, radius=LIGHT_RADIUS):
    os.makedirs(CACHE, exist_ok=True)
    key = f"{T.name}_e{elev:g}_a{azim:g}_ds{ds:g}_r{radius:g}.npy"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        return np.load(path)
    L = light(elev, azim)
    ref = soft_reference(T, T.uv_grid(), L, ds, radius) if radius > 0 else hard_reference(T, T.uv_grid(), L, ds)
    np.save(path, ref.astype(np.float32))
    return ref


# ---------------------------------------------------------------- baselines
def fxps(T, uv, L, ds, N, alpha=0.5):
    """fastApproximateShadow from shaders.js, verbatim."""
    ray = L * ds
    h0 = T.center(uv)[..., 3]
    shadow = np.zeros_like(h0)
    for i in range(1, int(N) + 1):
        t = (i / N) ** alpha
        ray_h = h0 + ray[2] * t
        sh = T.sample(uv + ray[:2] * t)[..., 3]
        shadow = np.maximum(shadow, (sh - ray_h) / i)
    return np.clip(1.0 - shadow * N, 0.0, 1.0)


def pom_soft(T, uv, L, ds, N):
    """pomSoftShadow from shaders.js, verbatim (depth convention)."""
    d0 = 1.0 - T.center(uv)[..., 3]
    dps = d0 / N
    uv_step = L[:2] * ds / L[2]
    m = np.zeros_like(d0)
    for i in range(int(N)):
        k = i + 1
        sd = 1.0 - T.sample(uv + uv_step[None, None, :] * (dps * k)[..., None])[..., 3]
        m = np.maximum(m, ((d0 - dps * k) - sd) * (1.0 - k / N))
    return np.clip(1.0 - m * N, 0.0, 1.0)


def no_shadow(T, uv, L, ds, N):
    return np.ones(uv.shape[:-1], np.float32)


# ---------------------------------------------------------------- metrics
def blur(img, sigma=2.0):
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    out = sum(w * np.roll(img, s, 0) for w, s in zip(k, range(-r, r + 1)))
    return sum(w * np.roll(out, s, 1) for w, s in zip(k, range(-r, r + 1)))


def ndotl(T, uv, L):
    return np.clip(decode_normal(T.center(uv)) @ L.astype(np.float32), 0.0, 1.0)


def evaluate(fn, N, ds=DS_DEFAULT, textures=TEXTURES, elevs=ELEVS, azims=AZIMS, verbose=False, **kw):
    """Score fn against the soft reference. Returns dict of mean metrics.

    rmse      RMSE of raw visibility
    rmse_lit  RMSE of max(N.L,0)*visibility  (what actually reaches the screen)
    rmse_blur RMSE of lit after sigma=2px blur (large-scale grounding, ignores edge detail)
    bias      mean(candidate) - mean(reference) of lit   (+ = too bright)
    flicker   mean|d cand| / mean|d ref| of lit when the light azimuth moves 2 deg
              (1 = moves like the reference, >>1 = pops/crawls, <<1 = shadows stuck)
    fetches   texture fetches per pixel actually used (centre fetch excluded)
    """
    rows = []
    for name in textures:
        T = Tex(name); uv = T.uv_grid()
        for e in elevs:
            for a in azims:
                L = light(e, a); L2 = light(e, a + 2.0)
                ref = cached_reference(T, e, a, ds); ref2 = cached_reference(T, e, a + 2.0, ds)
                T.fetches = 0
                c = np.clip(fn(T, uv, L, ds, N, **kw), 0, 1); f = T.fetches
                c2 = np.clip(fn(T, uv, L2, ds, N, **kw), 0, 1)
                nl, nl2 = ndotl(T, uv, L), ndotl(T, uv, L2)
                lc, lr = nl * c, nl * ref
                row = dict(tex=name, elev=e, azim=a,
                           rmse=float(np.sqrt(np.mean((c - ref) ** 2))),
                           rmse_lit=float(np.sqrt(np.mean((lc - lr) ** 2))),
                           rmse_blur=float(np.sqrt(np.mean((blur(lc) - blur(lr)) ** 2))),
                           bias=float(lc.mean() - lr.mean()),
                           flicker=float(np.abs(nl2 * c2 - lc).mean() / max(np.abs(nl2 * ref2 - lr).mean(), 1e-6)),
                           fetches=f)
                rows.append(row)
                if verbose:
                    print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    keys = ["rmse", "rmse_lit", "rmse_blur", "bias", "flicker", "fetches"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["rows"] = rows
    return out


def fmt(name, m):
    return (f"{name:28s} rmse={m['rmse']:.4f} lit={m['rmse_lit']:.4f} blur={m['rmse_blur']:.4f} "
            f"bias={m['bias']:+.4f} flicker={m['flicker']:.2f} fetches={m['fetches']:.1f}")


# ---------------------------------------------------------------- images
def save_sheet(path, fns, tex="rocks", elev=30.0, azim=25.0, N=4, ds=DS_DEFAULT, lit=True, scale=1):
    """Contact sheet: reference | each candidate, side by side. fns = {label: fn}."""
    T = Tex(tex); uv = T.uv_grid(); L = light(elev, azim)
    nl = ndotl(T, uv, L) if lit else 1.0
    tiles = [("reference", cached_reference(T, elev, azim, ds))]
    tiles += [(k, np.clip(f(T, uv, L, ds, N), 0, 1)) for k, f in fns.items()]
    imgs = [np.clip((0.3 + 0.7 * nl * t) if lit else t, 0, 1) for _, t in tiles]
    sheet = np.concatenate(imgs, 1)
    im = Image.fromarray((sheet * 255).astype(np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)
    return [k for k, _ in tiles]


if __name__ == "__main__":
    for N in (3, 4, 6):
        print(f"--- N={N}")
        print(fmt("no shadow", evaluate(no_shadow, N)))
        print(fmt("fxps a=0.5", evaluate(fxps, N)))
        print(fmt("pom soft", evaluate(pom_soft, N)))
    print(save_sheet(os.path.join(CACHE, "baseline_rocks.png"), {"fxps": fxps, "pom_soft": pom_soft}))
