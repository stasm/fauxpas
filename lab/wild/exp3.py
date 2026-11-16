from conelib import *
F={2:[0.3,0.75],3:[0.2,0.45,0.8],4:[0.15,0.3,0.55,0.85]}
for n in (2,3,4):
  for beta in (0.25,0.5):
    for kappa in (0.2,0.4):
      for b in (0.0,0.25):
        for comb in ('max','sum'):
            run(f"cone{n} b{beta} k{kappa} bias{b} {comb}", cone(F[n],beta,kappa,b,comb), azims=[140.0])
