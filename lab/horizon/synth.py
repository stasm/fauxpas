"""Synthetic stress height fields (thin occluders) + uncached references stored in horizon/refs/."""
from hz import *
class SynTex(lab.Tex):
    def __init__(self, name, h):
        self.name = name
        n = h.shape[0]
        img = np.zeros((n, n, 4), np.float32)
        gy, gx = np.gradient(h); s = 8.0   # arbitrary bump strength for the normal
        nrm = np.stack([-gx * s, -gy * s, np.ones_like(h)], -1); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        img[..., :3] = nrm * 0.5 + 0.5; img[..., 3] = h
        self.mips = [img]
        while self.mips[-1].shape[0] > 1:
            m = self.mips[-1]
            self.mips.append(0.25 * (m[0::2, 0::2] + m[1::2, 0::2] + m[0::2, 1::2] + m[1::2, 1::2]))
        self.size = n; self.fetches = 0

def make(kind, n=256):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    if kind == 'posts':      # thin posts and thin walls on a floor, plus a broad dome
        h = np.full((n, n), 0.15, np.float32)
        for cx, cy, rad, top in [(40, 40, 2.5, 1.0), (120, 60, 4, 0.8), (200, 50, 1.5, 1.0), (60, 150, 6, 0.6), (150, 150, 3, 1.0), (220, 200, 10, 0.9), (90, 220, 2, 0.7)]:
            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            h = np.maximum(h, 0.15 + (top - 0.15) * np.clip((rad + 1 - d), 0, 1))
        # thin diagonal wall
        d = np.abs((x - y) * 0.7071 - 20); seg = (x + y > 200) & (x + y < 330)
        h = np.maximum(h, np.where(seg, 0.15 + 0.7 * np.clip(2.5 - d, 0, 1), 0))
        return h
    if kind == 'ridges':     # periodic thin ridges of varying height + fine ripples
        h = 0.2 + 0.03 * np.sin(x * 0.9) * np.sin(y * 0.7)
        for k, (pos, w, top) in enumerate([(30, 2, 1.0), (90, 4, 0.7), (150, 1.5, 0.9), (210, 8, 0.5)]):
            d = np.abs(x - pos - 6 * np.sin(y * 2 * np.pi / n * 2))
            h = np.maximum(h, 0.2 + (top - 0.2) * np.clip(w + 1 - d, 0, 1))
        return h.astype(np.float32)

REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refs')
def ref(T, elev, azim, ds=0.08):
    os.makedirs(REFS, exist_ok=True)
    p = os.path.join(REFS, f"{T.name}_e{elev:g}_a{azim:g}_ds{ds:g}.npy")
    if os.path.exists(p): return np.load(p)
    r = lab.soft_reference(T, T.uv_grid(), lab.light(elev, azim), ds).astype(np.float32)
    np.save(p, r); return r

def score(fn, N, T, elev, azim, ds=0.08):
    uv = T.uv_grid(); L = lab.light(elev, azim); nl = lab.ndotl(T, uv, L)
    c = np.clip(fn(T, uv, L, ds, N), 0, 1); r = ref(T, elev, azim, ds)
    lc, lr = nl * c, nl * r
    return dict(lit=float(np.sqrt(np.mean((lc - lr) ** 2))), blur=float(np.sqrt(np.mean((lab.blur(lc) - lab.blur(lr)) ** 2))), bias=float(lc.mean() - lr.mean()))

def sheet(path, fns, T, elev, azim, N=4, ds=0.08, scale=1, raw=False):
    from PIL import Image
    uv = T.uv_grid(); L = lab.light(elev, azim); nl = 1.0 if raw else lab.ndotl(T, uv, L)
    tiles = [ref(T, elev, azim, ds)] + [np.clip(f(T, uv, L, ds, N), 0, 1) for f in fns.values()]
    im = Image.fromarray((np.concatenate([np.clip(t if raw else 0.3 + 0.7 * nl * t, 0, 1) for t in tiles], 1) * 255).astype(np.uint8))
    if scale != 1: im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)

if __name__ == '__main__':
    import time
    for kind in ['posts', 'ridges']:
        T = SynTex(kind, make(kind))
        for e in (15, 30, 50):
            t = time.time(); ref(T, e, 25); ref(T, e, 140); print(kind, e, time.time() - t, flush=True)
