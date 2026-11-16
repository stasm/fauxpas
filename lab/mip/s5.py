from fam import *
b=dict(sched='uni',wmode='rise',wk=0.5,wmin=0.01)
it=[('pt',dict(b,lodmode='none'))]
for lb in (-3,-2,-1): it.append((f'match{lb}',dict(b,lodbias=lb)))
for lc in (1,2): it.append((f'const{lc}',dict(b,lodmode='const',lodconst=lc)))
for k in (0.1,0.25,0.5): it.append((f'cone{k}',dict(b,lodmode='cone',conek=k)))
for hb in (0.02,0.05): it.append((f'match0 hb{hb}',dict(b,hbias=hb)))
it.append(('geo match-1',dict(b,sched='geo',lodbias=-1)))
it.append(('geo match-2',dict(b,sched='geo',lodbias=-2)))
sweep(it,Ns=(2,4))
