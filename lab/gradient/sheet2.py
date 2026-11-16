from cands import *
fns={'pt.25':lambda *a: point(*a,p=1.5,off=0.5,soft=0.25),'hmid.25':lambda *a: herm(*a,origin=False),'fxps':lab.fxps}
for tex,e,a in (('bump',15,265),('rocks',30,25),('bricks',15,25)):
    print(lab.save_sheet(f's2_{tex}_e{e}.png',fns,tex=tex,elev=e,azim=a,N=4,scale=2))
