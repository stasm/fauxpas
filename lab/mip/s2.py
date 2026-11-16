from fam import *
f=lambda **kw: (lambda T,uv,L,ds,n: fam(T,uv,L,ds,n,**kw))
fns={'pt w.5':f(lodmode='none',wmode='rise',wk=0.5),'mip w1':f(wmode='rise',wk=1.0),'fxps':lab.fxps}
for tex,e in [('rocks',30),('bricks',15),('bump',50)]:
    lab.save_sheet(f'a_{tex}_{e}.png',fns,tex=tex,elev=e)
m=lab.evaluate(fns['pt w.5'],4,textures=['rocks'],verbose=True)
