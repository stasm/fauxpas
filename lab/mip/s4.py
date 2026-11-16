from fam import *
it=[]
for sched in ['uni','uni1','geo','pow']:
    for lodmode in ['none','match']:
        for wmode,wk in [('rise',0.5),('rise',1.0),('dist',0.5),('dist',1.0)]:
            it.append((f"{sched} {lodmode} {wmode}{wk}", dict(sched=sched, lodmode=lodmode, wmode=wmode, wk=wk, wmin=0.01)))
sweep(it)
