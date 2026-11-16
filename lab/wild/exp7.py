from horizon_shadow import *
from common import sat
def tails(fn, N=4):
    out=[]
    for name in lab.TEXTURES:
        T=lab.Tex(name); uv=T.uv_grid()
        for e in lab.ELEVS:
            for a in lab.AZIMS:
                L=lab.light(e,a); L2=lab.light(e,a+2)
                c,c2=sat(fn(T,uv,L,0.08,N)),sat(fn(T,uv,L2,0.08,N))
                d=np.abs(lab.ndotl(T,uv,L2)*c2-lab.ndotl(T,uv,L)*c)
                out.append([np.percentile(d,99),np.percentile(d,99.9),d.max()])
    return np.mean(out,0).round(3)
for beta,wr,wa in ((0.5,0.1,0.0),(0.5,0.1,0.02),(0.5,0.1,0.05),(0.5,0.0,0.06),(0.75,0.1,0.03),(1.0,0.1,0.03),(0.75,0.15,0.03)):
    f=variant(beta=beta,w_rel=wr,w_abs=wa)
    print(lab.fmt(f"b{beta} wr{wr} wa{wa}", lab.evaluate(f,4)), 'tails', tails(f), flush=True)
