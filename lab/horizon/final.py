"""Final tables / sheets for horizon_shadow. Run: python3 final.py"""
from hz2 import *
from e4 import stab
import horizon_shadow as H
f = H.horizon_shadow
print('== N sweep (real textures, ds=0.08)')
for N in (2, 3, 4, 6):
    m = lab.evaluate(f, N); print(lab.fmt('horizon N=%d' % N, m)); print('    ' + per_elev(m)); print('    ' + per_tex(m))
    print('    synth: ' + synth_scores(f, N), flush=True)
for N in (2, 3, 4, 6):
    print(lab.fmt('horizon uncapped N=%d' % N, lab.evaluate(H.variant(smax=1e3), N)), flush=True)
print('== zenith / grazing sanity')
T = lab.Tex('rocks'); uv = T.uv_grid()
for L in [np.array([0, 0, 1.0]), lab.light(89.5, 25), lab.light(86, 25), lab.light(3, 25), lab.light(0.5, 25)]:
    v = f(T, uv, L, 0.08, 4); print('L.z=%.4f mean vis %.3f min %.3f nan %d' % (L[2], v.mean(), v.min(), np.isnan(v).sum()))
print('== depth_scale robustness (own refs, not cached in lab/cache)')
def ref_ds(T, e, a, ds):
    p = os.path.join(REFS, f"{T.name}_e{e:g}_a{a:g}_ds{ds:g}.npy")
    if os.path.exists(p): return np.load(p)
    r = lab.soft_reference(T, T.uv_grid(), lab.light(e, a), ds).astype(np.float32); os.makedirs(REFS, exist_ok=True); np.save(p, r); return r
for tex, e, a in [('rocks', 30, 25), ('bump', 15, 140)]:
    T = lab.Tex(tex); uv = T.uv_grid(); L = lab.light(e, a); nl = lab.ndotl(T, uv, L)
    for ds in (0.04, 0.08, 0.16):
        r = ref_ds(T, e, a, ds) if ds != 0.08 else lab.cached_reference(T, e, a)
        row = []
        for k, fn in [('horizon', f), ('uncapped', H.variant(smax=1e3)), ('fxps', lab.fxps), ('none', lab.no_shadow)]:
            c = np.clip(fn(T, uv, L, ds, 4), 0, 1); row.append('%s lit=%.3f blur=%.3f' % (k, np.sqrt(np.mean((nl * (c - r)) ** 2)), np.sqrt(np.mean((lab.blur(nl * c) - lab.blur(nl * r)) ** 2))))
        print(tex, 'e%d ds=%.2f  ' % (e, ds) + ' | '.join(row), flush=True)
        if ds != 0.08:
            from PIL import Image
            tiles = [r] + [np.clip(fn(T, uv, L, ds, 4), 0, 1) for fn in (f, H.variant(smax=1e3), lab.fxps)]
            Image.fromarray((np.concatenate([0.3 + 0.7 * nl * t for t in tiles], 1) * 255).astype(np.uint8)).save('final_ds%g_%s_e%d.png' % (ds, tex, e))
print('== sheets: reference | horizon N=4 | horizon uncapped | fxps | pom_soft')
fns = {'horizon': f, 'uncapped': H.variant(smax=1e3), 'fxps': lab.fxps, 'pom_soft': lab.pom_soft}
for tex, e, a in [('rocks', 30, 25), ('bricks', 50, 140), ('bump', 15, 140), ('rocks', 15, 265), ('bricks', 15, 25), ('bump', 50, 25)]:
    lab.save_sheet('final_%s_e%d.png' % (tex, e), fns, tex=tex, elev=e, azim=a, N=4)
for kind in ('posts', 'ridges'):
    T = SynTex(kind, make(kind)); sheet('final_syn_%s_e30.png' % kind, fns, T, 30, 25)
