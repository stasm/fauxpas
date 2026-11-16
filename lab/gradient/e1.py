from common import *
import itertools
print(lab.fmt('fxps', lab.evaluate(lab.fxps,4))); print(lab.fmt('pom_soft', lab.evaluate(lab.pom_soft,4)))
for p,soft,off,den in itertools.product((1.0,1.5,2.0),(0.25,0.5,1.0,2.0),(0.0,0.5),(4,5)):
    if off==0.5 and den==5: continue
    summarize(f"pt p={p} soft={soft} off={off} den={den}", lambda *a: point(*a,p=p,soft=soft,off=off,den=den))
