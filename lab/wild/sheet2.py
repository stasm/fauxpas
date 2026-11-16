from conelib import *
import sys
F=[0.2,0.45,0.8]
fns = {f'b{b} k{k}': cone(F,b,k,0.0,'sum') for b,k in ((0,0.3),(0.25,0.3),(0.5,0.3),(0.5,0.6),(1.0,0.6))}
for t,e,a in (('bump',15,140),('rocks',15,25),('bricks',30,265)):
    print(lab.save_sheet(f'/home/user/fauxpas/lab/wild/s2_{t}_{e}.png', fns, tex=t, elev=float(e), azim=float(a)))
