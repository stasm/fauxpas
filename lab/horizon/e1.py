import time
from hz import *
t=time.time()
run('fxps', lab.fxps, detail=True); print(time.time()-t)
run('fxps a=1', lambda T,uv,L,ds,N: lab.fxps(T,uv,L,ds,N,alpha=1.0), detail=True)
run('pom_soft', lab.pom_soft, detail=True)
# physically exact horizon, r = 0.05
for dist in ['uni','quad','geo2']:
    for reach in ['exit','full']:
        run('phys r.05 %s %s'%(dist,reach), mk(dist=dist, reach=reach, r=0.05), detail=True)
