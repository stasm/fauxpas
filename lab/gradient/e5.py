from g2 import *
kw=dict(azims=[25.0])
G=lambda **o: (lambda *a: gs(*a,**o))
for name,o in [("base",dict()),("secant.25",dict(secant=.25)),("secant.5",dict(secant=.5)),("lin.15",dict(lin=.15)),("lin.25",dict(lin=.25)),("lin.5",dict(lin=.5)),
  ("lin.25 k*.5",dict(lin=.25,kscale=.5)),("lin.25 wrong sign",dict(lin=.25,kscale=-1)),("origin",dict(origin=True))]:
    detail(name,G(**o),**kw)
