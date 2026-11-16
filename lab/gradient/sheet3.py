from cands import *
P=lambda **o: (lambda *a: point(*a,off=0.5,soft=0.25,**o))
fns={'p1':P(p=1.0),'p1 fade1':P(p=1.0,fade=1.0),'p1.5 fade1':P(p=1.5,fade=1.0),'p1 grow4':P(p=1.0,grow=4.0)}
print(lab.save_sheet('s3_bump_e15.png',fns,tex='bump',elev=15,azim=265,N=4,scale=2))
for k,f in fns.items(): detail(k,f,azims=[25.0])
