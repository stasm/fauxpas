from fam import *
b=dict(sched='uni',wmode='dist',wk=0.35,wmin=0.003,lodmode='none',combine='prod')
it=[]
for wk in (0.2,0.27,0.35,0.45): it.append((f'pt dist{wk}',dict(b,wk=wk)))
for k in (1.0,1.5,2.0): it.append((f'pt phys{k}',dict(b,wmode='phys',wk=k)))
it+=[('dist.35 min',dict(b,combine='min')),('dist.35 uni1',dict(b,sched='uni1')),('dist.35 smooth',dict(b,shape='smooth')),
     ('dist.35 wmin.001',dict(b,wmin=0.001)),('dist.35 wmin.006',dict(b,wmin=0.006)),
     ('dist.35 light1',dict(b,lodmode='light',conek=1.0)),('dist.35 light2',dict(b,lodmode='light',conek=2.0)),
     ('dist.35 m-2',dict(b,lodmode='match',lodbias=-2)),('dist.35 pow1.5',dict(b,sched='pow',power=1.5)),('dist.35 geo',dict(b,sched='geo')),
     ('dist.35 off.6',dict(b,off=0.6)),('dist.35 off.4',dict(b,off=0.4))]
sweep(it,Ns=(2,4))
