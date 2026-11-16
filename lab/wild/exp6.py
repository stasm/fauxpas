from hlib import *
for beta,w0 in ((0.5,0.05),(0.5,0.1),(1.0,0.1),(1.0,0.2)):
    for comb in ('sum','max'):
        run(f"hz b{beta} w{w0} {comb}", horizon(0.5,beta,w0,0,comb,'space'), Ns=(4,))
# add-ons on top of hz b.5 w.1 sum
base = horizon(0.5,0.5,0.1,0,'sum','space')
def with_cavity(lod, g):
    def f(T,uv,L,ds,N):
        v = base(T,uv,L,ds,N-1); h0=T.center(uv)[...,3]
        return v*(1-sat(g*(T.sample(uv,lod)[...,3]-h0)))
    return f
def with_zero(k):
    def f(T,uv,L,ds,N):
        v = base(T,uv,L,ds,N); h0=T.center(uv)[...,3]; Lh,cot=geom(L)
        return v*(1-k*(1-h0)*min(ds*cot*4,1.0))
    return f
run("hz N=3 alone", base, Ns=(3,))
for lod,g in ((3.0,0.5),(5.0,0.5),(5.0,1.0)): run(f"hz3+cavity lod{lod} g{g}", with_cavity(lod,g), Ns=(4,))
for k in (0.1,0.25): run(f"hz4*zero k{k}", with_zero(k), Ns=(4,))
