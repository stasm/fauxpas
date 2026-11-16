from fam import *
f=lambda **kw: (lambda T,uv,L,ds,n: fam(T,uv,L,ds,n,sched='uni',wmode='rise',wk=0.5,wmin=0.01,**kw))
fns={'pt':f(lodmode='none'),'m-2':f(lodbias=-2),'m0':f(lodbias=0)}
for tex,e,N in [('rocks',15,2),('rocks',15,4),('bricks',15,2),('bricks',30,3),('bump',15,2),('bump',30,4)]:
    crop_sheet(f'c_{tex}_{e}_N{N}.png',fns,tex,e,N=N)
