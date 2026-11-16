from synth import *
cands = {'uni':mk(dist='uni'), 'mid':mk(dist='mid'), 'quad':mk(dist='quad'), 'midquad':mk(dist='midquad'), 'geo15':mk(dist='geo15'), 'geo2':mk(dist='geo2'),
         'mid c.08':mk(dist='mid',reach='exitc',dmax=0.08), 'quad c.12':mk(dist='quad',reach='exitc',dmax=0.12),
         'mid r.1':mk(dist='mid', r=0.1), 'fxps':lab.fxps, 'fxps a1':lambda T,uv,L,ds,N: lab.fxps(T,uv,L,ds,N,alpha=1.0), 'pom_soft':lab.pom_soft}
for kind in ['posts','ridges']:
    T = SynTex(kind, make(kind))
    for N in (4, 6):
        print('---', kind, 'N', N)
        for k, f in cands.items():
            s = [score(f, N, T, e, a) for e in (15, 30, 50) for a in (25, 140)]
            print('%-10s' % k, ' '.join('e%d: lit=%.3f blur=%.3f' % (e, np.mean([x['lit'] for x in s[2*j:2*j+2]]), np.mean([x['blur'] for x in s[2*j:2*j+2]])) for j, e in enumerate((15,30,50))))
    sheet('syn_%s_e15.png' % kind, {k: cands[k] for k in ['mid','quad','geo2','fxps']}, T, 15, 25)
    sheet('syn_%s_e30.png' % kind, {k: cands[k] for k in ['mid','quad','geo2','fxps']}, T, 30, 25)
