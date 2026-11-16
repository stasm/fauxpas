from g2 import *
G=lambda **o: (lambda *a: gs(*a,**o))
for N in (2,3,4,6):
    detail("base p1.5", G(), N)
    summarize("p1", G(p=1.0), N)
    summarize("hermite lin.2 (oracle k)", G(lin=.2), N)
    summarize("fxps", lab.fxps, N)
m=lab.evaluate(G(),4)
for r in sorted(m['rows'],key=lambda r:-r['rmse_lit'])[:6]: print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
for r in sorted(m['rows'],key=lambda r:-abs(r['flicker']-1))[:4]: print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
