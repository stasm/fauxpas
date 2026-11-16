from hz2 import *
from e4 import stab
import horizon_shadow as H
for r in (0.05, 0.07, 0.1):
    for sm in (0.03, 0.04, 1e3):
        f = H.variant(r=r, smax=sm)
        m = lab.evaluate(f, 4); print(lab.fmt('r=%.2f smax=%g' % (r, sm), m)); print('    ' + per_elev(m)); print('    ' + per_tex(m))
        print('    synth: ' + synth_scores(f, 4))
        print('    pops e15: ' + ' '.join('%s %.2f/%.3f' % ((t,) + tuple(np.array(stab(f, 4, t, 15)[2:]) * [1e3, 1])) for t in lab.TEXTURES), flush=True)
