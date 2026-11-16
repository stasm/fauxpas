from g2 import *
G=lambda **o: (lambda *a: gs(*a,**o))
fns={'base':G(),'p1':G(p=1.0),'fxps':lab.fxps}
for tex,e,a in (('bump',15,265),('rocks',15,25),('bricks',30,140),('bump',50,25)):
    print(lab.save_sheet(f's4_{tex}_e{e}.png',fns,tex=tex,elev=e,azim=a,N=4,scale=2))
