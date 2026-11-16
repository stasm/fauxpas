"""Stability under fine light rotation + proxy reference at arbitrary elevation."""
from hz import *
def stab(fn, N, tex, elev, a0=25.0, step=0.5, n=9, ds=0.08):
    T=lab.Tex(tex); uv=T.uv_grid()
    fr=[]
    for j in range(n):
        L=lab.light(elev, a0+j*step)
        fr.append(lab.ndotl(T,uv,L)*np.clip(fn(T,uv,L,ds,N),0,1))
    fr=np.array(fr); d1=np.abs(np.diff(fr,axis=0)); d2=np.abs(np.diff(fr,n=2,axis=0))
    return d1.mean(), np.percentile(d1,99.9), d2.mean(), np.percentile(d2,99.9)
if __name__=='__main__':
    proxy=mk(dist='uni')
    print(lab.fmt('proxy N=64', lab.evaluate(proxy,64,elevs=[15.0,50.0])))
    cands={'uni4':(mk(dist='uni'),4),'geo15_4':(mk(dist='geo15'),4),'quad4':(mk(dist='quad'),4),'uni6':(mk(dist='uni'),6),
           'fxps4':(lab.fxps,4),'proxy64':(proxy,64)}
    for tex in lab.TEXTURES:
        for elev in (8,15,30):
            for k,(f,N) in cands.items():
                print(tex,elev,'%-8s'%k,'d1 mean %.4f p99.9 %.3f | d2 mean %.4f p99.9 %.3f'%stab(f,N,tex,elev), flush=True)
