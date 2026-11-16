from cands import *
fns={'pt.1':lambda *a: point(*a,p=1.5,off=0.5,soft=0.1),'pt.25':lambda *a: point(*a,p=1.5,off=0.5,soft=0.25),
     'hmid.25':lambda *a: herm(*a,origin=False),'hlin.25':lambda *a: herm(*a,mid=False,origin=False,lin=0.25),'fxps':lab.fxps}
for tex,e,a in (('rocks',15,25),('bricks',30,140),('bump',15,265),('bricks',15,25)):
    print(lab.save_sheet(f's1_{tex}_e{e}.png',fns,tex=tex,elev=e,azim=a,N=4))
