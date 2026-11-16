from fam import *
f=lambda **kw: (lambda T,uv,L,ds,n: fam(T,uv,L,ds,n,sched='uni',wmode='rise',wk=0.5,wmin=0.01,**kw))
fns={'pt':f(lodmode='none'),'m-2':f(lodbias=-2),'m0':f(lodbias=0)}
for tex,e in [('rocks',15),('bricks',15),('bump',30)]:
  for N in (2,4):
    lab.save_sheet(f'b_{tex}_{e}_N{N}.png',fns,tex=tex,elev=e,N=N)
