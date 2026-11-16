from fam import *
print(lab.fmt('fxps', lab.evaluate(lab.fxps, 4)))
for sched in ['uni','uni1','geo','pow']:
    for lodmode in ['none','match']:
        for wmode,wk in [('rise',0.5),('rise',1.0),('rise',2.0),('dist',0.5),('dist',1.0)]:
            run(f"{sched} {lodmode} {wmode}{wk}", sched=sched, lodmode=lodmode, wmode=wmode, wk=wk)
