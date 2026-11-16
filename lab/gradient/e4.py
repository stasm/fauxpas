from g2 import *
kw=dict(azims=[25.0])
G=lambda **o: (lambda *a: gs(*a,**o))
for name,o in [("centred physW",dict()),("eps.005",dict(eps=.005)),("eps .05",dict(eps=.05)),("centred W.2",dict(W=.2)),("centred W.4",dict(W=.4)),("W.6",dict(W=.6)),
  ("physW r=.1",dict(radius=.1)),
  ("physW secant.5",dict(secant=.5)),("physW lin.25",dict(lin=.25)),
  ("physW h0+.02",dict(h0bias=.02)),("physW h0-.02",dict(h0bias=-.02)),("physW p1",dict(p=1.0)),("physW p2",dict(p=2.0))]:
    detail(name,G(**o),**kw)
