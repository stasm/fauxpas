from horizon_shadow import *
from common import geom, sat
def hb(beta, wr, wa, b, npow=1.0):
    def f(T, uv, L, ds, N):
        N=int(N); h0=T.center(uv)[...,3]; Lh,cot=geom(L); H=1-h0; dexit=ds*cot*H
        lod=np.log2(np.maximum(beta*dexit*T.size/N**npow,1.0)); iw=1/(wr*H+wa); occ=0
        for i in range(N):
            fr=(i+0.5)/N; h=T.sample(uv+Lh*(fr*dexit)[...,None],lod)[...,3]
            occ=occ+sat(b+(h-h0-fr*H)*iw)
        return 1-sat(occ)
    return f
for beta,wr,wa,b,npow in ((0.75,0.15,0.03,0,1),(0.75,0.15,0.03,0.25,1),(0.75,0.15,0.03,0.5,1),(0.75,0.25,0.03,0.5,1),(0.5,0.15,0.03,0.25,1),(0.6,0.15,0.03,0.0,1),(1.5,0.15,0.03,0.0,2),(1.5,0.15,0.03,0.25,2)):
    for N in (2,4):
        print(lab.fmt(f"b{beta} wr{wr} bias{b} np{npow} N={N}", lab.evaluate(hb(beta,wr,wa,b,npow),N,azims=[140.0])), flush=True)
