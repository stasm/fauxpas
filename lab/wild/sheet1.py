from common import *
from exp2lib import multi
fns = {'fxps4': lambda T,uv,L,ds,N: lab.fxps(T,uv,L,ds,4), '1s g8': multi([0.5],8), '2s g8 sum': multi([0.3,0.75],8,comb='sum'), '3s g8 sum': multi([0.2,0.45,0.75],8,comb='sum')}
for t,e,a in (('rocks',30,25),('bump',15,140),('bricks',50,265),('bricks',15,25)):
    print(lab.save_sheet(f'/home/user/fauxpas/lab/wild/s1_{t}_{e}.png', fns, tex=t, elev=float(e), azim=float(a)))
