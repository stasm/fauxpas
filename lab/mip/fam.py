"""Parametric family of prefiltered (mip) light-ray shadows. Exploration only."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import lab

def ign(shape):
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    return np.modf(52.9829189 * np.modf(0.06711056 * x + 0.00583715 * y)[0])[0].astype(np.float32)

def fam(T, uv, L, ds, N, sched='uni', lodmode='match', lodbias=0.0, range_cap=None, range_frac=1.0,
        wmode='rise', wk=1.0, wmin=0.0, hbias=0.0, combine='min', jitter=0.0, ratio=2.0, power=2.0,
        off=0.5, shape='lin', fade=False, lodconst=0.0, conek=0.1, baselod=0.0, basemode='max'):
    N = int(N)
    c = T._sample(uv, baselod); h0 = c[..., 3]
    hl = float(np.hypot(L[0], L[1])); Lz = float(L[2])
    tau_exit = ds * (1.0 - h0) / Lz                 # per pixel
    tau_max = tau_exit * range_frac
    if range_cap is not None:                      # cap on horizontal reach, in uv units
        tau_max = np.minimum(tau_max, range_cap / max(hl, 1e-6))
    j = (ign(h0.shape) - 0.5) * jitter if jitter else 0.0
    # normalised positions t_i in (0,1] and spacing s_i
    ts, ss = [], []
    for i in range(1, N + 1):
        if sched == 'uni':   t = (i - 0.5) / N; s = 1.0 / N
        elif sched == 'uni1': t = i / N; s = 1.0 / N
        elif sched == 'geo':            # cells [r^(i-1-N), r^(i-N)], sample at cell centre; first cell from 0
            hi = ratio ** (i - N); lo = ratio ** (i - 1 - N) if i > 1 else 0.0
            t = 0.5 * (hi + lo); s = hi - lo
        elif sched == 'pow':
            hi = (i / N) ** power; lo = ((i - 1) / N) ** power
            t = 0.5 * (hi + lo); s = hi - lo
        ts.append(t); ss.append(s)
    vis = np.ones_like(h0); acc = np.zeros_like(h0)
    for t, s in zip(ts, ss):
        tj = t + j * s
        tau = tau_max * tj
        sp_tex = tau_max * s * hl * T.size          # footprint length in texels (lod-0)
        if lodmode == 'match':
            lod = np.log2(np.maximum(sp_tex, 1e-3)) + lodbias
            lod = np.maximum(lod, 0.0)
        elif lodmode == 'cone':                      # footprint = conek * distance travelled
            lod = np.maximum(np.log2(np.maximum(conek * tau * hl * T.size, 1e-3)), 0.0)
        elif lodmode == 'const':
            lod = lodconst
        else:
            lod = 0.0
        if baselod > 0:
            lod = np.maximum(lod, baselod) if basemode == 'max' else lod + baselod
        h = T.sample(uv + (L[:2] * 1.0)[None, None, :] * tau[..., None], lod)[..., 3]
        rise = Lz * tau / ds                        # ray height above h0 at sample (normalised)
        pen = h + hbias * (np.maximum(lod, 0) if not np.isscalar(lod) else max(lod, 0)) - (h0 + rise)
        if wmode == 'rise':       # half the ray rise across the footprint
            w = wk * Lz * tau_max * s / ds
        elif wmode == 'dist':     # proportional to distance (cone / area light)
            w = wk * rise
        elif wmode == 'const':
            w = wk
        x = np.clip((pen + off * w) / (w + wmin + 1e-6), 0.0, 1.0)
        if shape == 'smooth': x = x * x * (3 - 2 * x)
        if fade: x = x * (1.0 - tj)
        if combine == 'min': acc = np.maximum(acc, x)
        elif combine == 'prod': vis = vis * (1 - x)
        elif combine == 'sum': acc = acc + x
    if combine == 'prod': return vis
    return 1.0 - np.clip(acc, 0, 1)

def run(name, Ns=(4,), **kw):
    for N in Ns:
        m = lab.evaluate(lambda T, uv, L, ds, n: fam(T, uv, L, ds, n, **kw), N, azims=AZ)
        print(lab.fmt(f"{name} N={N}", m), flush=True)
    return m

AZ = [25.0, 265.0]
def _one(a):
    name, N, kw = a
    m = lab.evaluate(lambda T, uv, L, ds, n: fam(T, uv, L, ds, n, **kw), N, azims=AZ)
    r = m['rows']
    pt = ' '.join(f"{t[:3]}={np.mean([x['rmse_lit'] for x in r if x['tex']==t]):.3f}/{np.mean([x['flicker'] for x in r if x['tex']==t]):.2f}" for t in lab.TEXTURES)
    lo = np.mean([x['rmse_lit'] for x in r if x['elev'] <= 30]); 
    return lab.fmt(f"{name} N={N}", m)[:-12] + f" | lit/flk {pt} | lowlit={lo:.3f}"

def sweep(items, Ns=(4,)):
    from multiprocessing import Pool
    jobs = [(n, N, kw) for n, kw in items for N in Ns]
    with Pool(4) as p:
        for line in p.imap(_one, jobs): print(line, flush=True)

def crop_sheet(path, fns, tex, elev, azim=25.0, N=4, ds=0.08, crop=(0, 0, 128), scale=2, lit=False):
    from PIL import Image
    T = lab.Tex(tex); uv = T.uv_grid(); L = lab.light(elev, azim)
    nl = lab.ndotl(T, uv, L) if lit else 1.0
    y, x, s = crop
    tiles = [lab.cached_reference(T, elev, azim, ds)] + [np.clip(f(T, uv, L, ds, N), 0, 1) for f in fns.values()]
    tiles = [np.clip(0.3 + 0.7 * nl * t, 0, 1)[y:y + s, x:x + s] if lit else t[y:y + s, x:x + s] for t in tiles]
    sh = np.concatenate(tiles, 1)
    im = Image.fromarray((sh * 255).astype(np.uint8)).resize((sh.shape[1] * scale, sh.shape[0] * scale), Image.NEAREST)
    im.save(path)


def motion(fn, N, tex='rocks', elev=30.0, azim=25.0, ds=0.08, mode='azim', step=0.5, frames=9):
    """jerk = mean|2nd diff| / mean|1st diff| of lit over a slow light sweep (0 = glides, 2 = steps)."""
    T = lab.Tex(tex); uv = T.uv_grid(); fr = []
    for k in range(frames):
        L = lab.light(elev, azim + k * step) if mode == 'azim' else lab.light(elev + k * step, azim)
        fr.append(lab.ndotl(T, uv, L) * np.clip(fn(T, uv, L, ds, N), 0, 1))
    fr = np.stack(fr); d1 = np.abs(np.diff(fr, axis=0)); d2 = np.abs(np.diff(fr, 2, axis=0))
    return float(d2.mean() / max(d1.mean(), 1e-9)), float(d1.mean()), float(d1.max(axis=0).mean())

def minif(kw, N, m, tex, elev, azim=25.0, ds=0.08, basemode='max', base=None):
    """1 px = m texels. Reference = box average of the cached full-res reference visibility."""
    T = lab.Tex(tex); L = lab.light(elev, azim)
    ref = lab.cached_reference(T, elev, azim, ds)
    R = lab.RES // m
    refm = ref.reshape(R, m, R, m).mean((1, 3))
    span = lab.RES / T.size
    c = (np.arange(R) + 0.5) / R * span
    u, v = np.meshgrid(c, c); uv = np.stack([u, v], -1)
    bl = math.log2(m) if base is None else base   # hardware lod at the shaded pixel
    vis = np.clip(fam(T, uv, L, ds, N, baselod=bl, basemode=basemode, **kw), 0, 1)
    return float(np.sqrt(np.mean((vis - refm) ** 2))), vis, refm
