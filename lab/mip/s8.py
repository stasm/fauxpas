from fam import *
from multiprocessing import Pool
b=dict(sched='uni',wmode='rise',wk=0.5,wmin=0.01)
C={'pt':dict(b,lodmode='none'),'pt w1':dict(b,lodmode='none',wk=1.0),'m-2':dict(b,lodbias=-2),'m-1':dict(b,lodbias=-1),'m0':dict(b,lodbias=0),
   'markov pt':dict(b,lodmode='none',wmode='dist',wk=1.0,off=1.0),'markov m-1':dict(b,lodbias=-1,wmode='dist',wk=1.0,off=1.0),
   'markov m0':dict(b,lodbias=0,wmode='dist',wk=1.0,off=1.0),'half m0':dict(b,lodbias=0,wmode='dist',wk=0.5,off=1.0),'half pt':dict(b,lodmode='none',wmode='dist',wk=0.5,off=1.0)}
def job(a):
    name,N=a; kw=C[name]; fn=lambda T,uv,L,ds,n: fam(T,uv,L,ds,n,**kw)
    out=f"{name:12s} N={N}"
    for tex,e in [('bump',15),('rocks',30),('bricks',15)]:
        j,d1,_=motion(fn,N,tex,e); j2,_,_=motion(fn,N,tex,e,mode='elev')
        out+=f" | {tex[:3]}{e} jerkA={j:.2f} jerkE={j2:.2f} mot={d1*1e3:.2f}"
    return out
if __name__=='__main__':
    def lf(T,uv,L,ds,n): return lab.fxps(T,uv,L,ds,n)
    with Pool(4) as p:
        for l in p.imap(job,[(n,N) for n in C for N in (2,4)]): print(l,flush=True)
    for N in (4,):
        out='fxps N=4'
        for tex,e in [('bump',15),('rocks',30),('bricks',15)]:
            j,d1,_=motion(lab.fxps,N,tex,e); j2,_,_=motion(lab.fxps,N,tex,e,mode='elev'); out+=f" | {tex[:3]}{e} jerkA={j:.2f} jerkE={j2:.2f} mot={d1*1e3:.2f}"
        print(out)
    sweep([(k,v) for k,v in C.items() if 'markov' in k or 'half' in k or k=='pt w1'],Ns=(2,4))
