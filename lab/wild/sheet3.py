from hlib import *
fns = {'fxps': lab.fxps, 'b.25 w.05': horizon(0.5,0.25,0.05,0,'sum','space'), 'b.5 w.05': horizon(0.5,0.5,0.05,0,'sum','space'),
       'b.5 w.12': horizon(0.5,0.5,0.12,0,'sum','space'), 'b1 w.12': horizon(0.5,1.0,0.12,0,'sum','space')}
for t,e,a in (('bump',15,140),('bump',30,25),('rocks',30,25),('bricks',15,25),('bricks',50,265)):
    print(lab.save_sheet(f'/home/user/fauxpas/lab/wild/s3_{t}_{e}.png', fns, tex=t, elev=float(e), azim=float(a), N=4))
