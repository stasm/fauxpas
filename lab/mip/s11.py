from fam import *
b=dict(sched='uni',wmode='rise',wk=0.5,wmin=0.003,lodmode='none',combine='prod')
it=[('pt rise.5',b),('pt wmin0',dict(b,wmin=0.0)),('pt dist.5',dict(b,wmode='dist')),('pt dist.35',dict(b,wmode='dist',wk=0.35)),
    ('pt mix.25/.25',dict(b,wmode='mix',wk=0.25,wd=0.25)),('pt mix.35/.15',dict(b,wmode='mix',wk=0.35,wd=0.15)),
    ('m-2 rise.5',dict(b,lodmode='match',lodbias=-2)),('m-1 rise.5',dict(b,lodmode='match',lodbias=-1)),
    ('cone.1 dist.5',dict(b,lodmode='cone',conek=0.1,wmode='dist')),('cone.2 mix',dict(b,lodmode='cone',conek=0.2,wmode='mix',wk=0.25,wd=0.25)),
    ('uni1 pt rise.5',dict(b,sched='uni1'))]
sweep(it,Ns=(2,3,4,6))
