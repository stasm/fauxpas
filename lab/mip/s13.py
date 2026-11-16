from fam import *
from multiprocessing import Pool
b=dict(sched='uni',wmode='phys',wk=1.0,wmin=0.003,lodmode='none',combine='prod')
C={'pt':b,'light1':dict(b,lodmode='light',conek=1.0),'light2':dict(b,lodmode='light',conek=2.0),'m-2':dict(b,lodmode='match',lodbias=-2),'m-1':dict(b,lodmode='match',lodbias=-1),'m0':dict(b,lodmode='match',lodbias=0),
   'const1':dict(b,lodmode='const',lodconst=1.0),'jit':dict(b,jitter=1.0)}
CFG=[('bump',15),('bump',50),('rocks',30),('bricks',15),('bricks',50)]
def job(a):
    name,N=a
    if name=='fxps': fn=lab.fxps
    elif name=='ref': fn=lambda T,uv,L,ds,n: lab.soft_reference(T,uv,L,ds)
    else:
        kw=C[name]; fn=lambda T,uv,L,ds,n: fam(T,uv,L,ds,n,**kw)
    out=f"{name:8s} N={N}"
    for tex,e in (CFG if name!='ref' else CFG[2:4]):
        fr=5 if name=='ref' else 9
        j,d1,_=motion(fn,N,tex,e,frames=fr); j2,_,_=motion(fn,N,tex,e,mode='elev',frames=fr)
        out+=f" | {tex[:3]}{e} jA={j:.3f} jE={j2:.3f} mot={d1*1e3:.2f}"
    return out
if __name__=='__main__':
    with Pool(4) as p:
        for l in p.imap(job,[(n,N) for n in C for N in (2,4)]+[('fxps',4),('ref',0)]): print(l,flush=True)
