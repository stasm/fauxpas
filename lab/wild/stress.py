from hlib import *
import os
from PIL import Image
mine = horizon(0.5,0.5,0.1,0,'sum','space')
soft = horizon(0.5,1.0,0.2,0,'sum','space')
# 1) popping: per-pixel change for a 2 deg azimuth step, tail percentiles, vs reference
def tails(fn, N=4):
    out=[]
    for name in lab.TEXTURES:
        T=lab.Tex(name); uv=T.uv_grid()
        for e in lab.ELEVS:
            for a in lab.AZIMS:
                L=lab.light(e,a); L2=lab.light(e,a+2)
                if fn is None: c,c2=lab.cached_reference(T,e,a),lab.cached_reference(T,e,a+2)
                else: c,c2=sat(fn(T,uv,L,0.08,N)),sat(fn(T,uv,L2,0.08,N))
                d=np.abs(lab.ndotl(T,uv,L2)*c2-lab.ndotl(T,uv,L)*c)
                out.append([np.percentile(d,99),np.percentile(d,99.9),d.max()])
    return np.mean(out,0).round(3)
print('2deg step |d lit|  p99 p99.9 max')
for k,f in (('reference',None),('fxps',lab.fxps),('hz b.5 w.1',mine),('hz b1 w.2',soft),('hz sharp',horizon(0.5,0.25,0.05,0,'sum','space'))):
    print(f"  {k:12s}", tails(f))
# 2) out-of-cache references, stored in my own dir
def sheet(tag, tex, e, a, ds):
    T=lab.Tex(tex); uv=T.uv_grid(); L=lab.light(e,a)
    p=f'/home/user/fauxpas/lab/wild/ref_{tag}.npy'
    ref=np.load(p) if os.path.exists(p) else lab.soft_reference(T,uv,L,ds); np.save(p,ref)
    nl=lab.ndotl(T,uv,L); tiles=[ref]+[sat(f(T,uv,L,ds,4)) for f in (lab.fxps,mine,soft)]
    for nm,t in zip(('fxps','hz','hzsoft'),tiles[1:]):
        print(f"  {tag} {nm:7s} lit_rmse={np.sqrt(np.mean((nl*t-nl*ref)**2)):.3f} bias={np.mean(nl*t-nl*ref):+.3f}")
    Image.fromarray((np.concatenate([np.clip(0.3+0.7*nl*t,0,1) for t in tiles],1)*255).astype(np.uint8)).save(f'/home/user/fauxpas/lab/wild/stress_{tag}.png')
sheet('bump_e8_ds08','bump',8.0,140.0,0.08)
sheet('rocks_e30_ds20','rocks',30.0,25.0,0.2)
sheet('bricks_e30_ds20','bricks',30.0,265.0,0.2)
sheet('bump_e30_ds02','bump',30.0,25.0,0.02)
