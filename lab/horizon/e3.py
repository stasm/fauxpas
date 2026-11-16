from hz import *
import itertools
print('--- dist x N, reach=exit, r=0.05')
for dist in ['uni','mid','quad','sqrt','geo2','geo15']:
    for N in (2,3,4,6):
        run('%s'%dist, mk(dist=dist, reach='exit'), Ns=(N,))
print('--- kexit')
for ke in [0.7,0.85,0.95,1.0,1.1]:
    run('uni kexit %.2f'%ke, mk(dist='uni', kexit=ke), detail=True)
print('--- r')
for r in [0.03,0.05,0.08,0.12,0.2]:
    run('uni r=%.2f'%r, mk(dist='uni', r=r))
print('--- ramp/combine')
run('smooth', mk(dist='uni', ramp='smooth'))
run('prod', mk(dist='uni', combine='prod'))
run('sum', mk(dist='uni', combine='sum'))
run('prod r.1', mk(dist='uni', combine='prod', r=0.1))
