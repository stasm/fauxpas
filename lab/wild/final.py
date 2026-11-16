from horizon_shadow import *
from common import per_tex
import sys
if sys.argv[1]=='sheets':
    fx3=lambda T,uv,L,ds,N: lab.fxps(T,uv,L,ds,3)
    fns={'fxps N=4':lab.fxps,'xhs N=2':lambda T,uv,L,ds,N: horizon_shadow(T,uv,L,ds,2),'xhs N=4':horizon_shadow,
         'xhs sharp N=4':variant(beta=0.5,w_rel=0.1,w_abs=0.02)}
    for t,e,a in (('bump',15,140),('rocks',30,25),('bricks',50,265),('bump',70,25),('rocks',15,265)):
        print(lab.save_sheet(f'/home/user/fauxpas/lab/wild/final_{t}_e{e}.png', fns, tex=t, elev=float(e), azim=float(a), N=4))
else:
    for N in (1,2,3,4,6):
        print(lab.fmt(f"xhs N={N}", lab.evaluate(horizon_shadow, N)), flush=True)
    for N in (2,3,4,6):
        print(lab.fmt(f"xhs-sharp N={N}", lab.evaluate(variant(beta=0.5,w_rel=0.1,w_abs=0.02), N)), flush=True)
    for N in (2,3,4,6): print(lab.fmt(f"pom_soft N={N}", lab.evaluate(lab.pom_soft, N)), flush=True)
    per_tex('xhs', horizon_shadow, 4); per_tex('xhs', horizon_shadow, 2)
