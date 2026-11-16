from cands import *
kw=dict(azims=[25.0])
A=lambda **o: (lambda *a: adaptive(*a,**o))
detail("point .25", lambda *a: point(*a,p=1.5,off=0.5,soft=0.25), **kw)
for o in (dict(), dict(use_nz=False), dict(alpha=0.5), dict(alpha=2.0), dict(first=0.05), dict(smin=0.15), dict(newton=True), dict(newton=True,smin=0.1)):
    detail(f"adaptive {o}", A(**o), **kw)
