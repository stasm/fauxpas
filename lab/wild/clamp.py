from horizon_shadow import *
import math
from PIL import Image
def clamped(cmax):
    def f(T,uv,L,ds,N):
        hl=math.hypot(L[0],L[1]); cot=hl/max(L[2],1e-3)
        if cot>cmax:
            z=hl/cmax; L=np.array([L[0],L[1],z])   # raise the virtual light; horizon_shadow only uses xy direction and cot
        return horizon_shadow(T,uv,L,ds,N)
    return f
T=lab.Tex('bump'); uv=T.uv_grid(); L=lab.light(8.0,140.0); ref=np.load('ref_bump_e8_ds08.npy'); nl=lab.ndotl(T,uv,L)
tiles=[ref]
for nm,f in (('fxps',lab.fxps),('xhs',horizon_shadow),('xhs cot<=4',clamped(4.0)),('xhs cot<=3',clamped(3.0))):
    t=np.clip(f(T,uv,L,0.08,4),0,1); tiles.append(t)
    print(f"{nm:12s} e8 lit_rmse={np.sqrt(np.mean((nl*t-nl*ref)**2)):.3f} bias={np.mean(nl*t-nl*ref):+.3f}")
Image.fromarray((np.concatenate([np.clip(0.3+0.7*nl*t,0,1) for t in tiles],1)*255).astype(np.uint8)).save('final_stress_bump_e8.png')
print(lab.fmt('xhs nomip(beta=0) N=4', lab.evaluate(variant(beta=0.0),4)))
print(lab.fmt('xhs cot<=4 N=4', lab.evaluate(clamped(4.0),4)))
