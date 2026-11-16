from hlib import *
import itertools
for a,beta,w0,kappa,lm in itertools.product((0.35,0.5),(0.25,0.4),(0.05,0.12),(0.0,0.15),('dist','space')):
    for N in (3,4):
        run(f"hz a{a} b{beta} w{w0} k{kappa} {lm}", horizon(a,beta,w0,kappa,'sum',lm), Ns=(N,), azims=[140.0])
