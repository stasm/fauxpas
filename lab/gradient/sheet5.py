from g2 import *
G=lambda **o: (lambda *a: gs(*a,**o))
fns={'lin.25':G(lin=.25),'lin.5':G(lin=.5),'secant.5':G(secant=.5)}
print(lab.save_sheet(f's5_bump_e15.png',fns,tex='bump',elev=15,azim=265,N=4,scale=2))
print(lab.save_sheet(f's5_bump_e15_N6.png',{'base N6':G(),'p1 N6':G(p=1.0)},tex='bump',elev=15,azim=265,N=6,scale=2))
