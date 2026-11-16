from hz3 import *
from e4 import stab
C = {'exit': mk3(smax=0), 'cap.03 nofade': mk3(smax=0.03, fade=0), 'cap.03 fade': mk3(smax=0.03), 'cap.03 fade uni': mk3(smax=0.03, dist='uni'),
     'cap.03 fade quad': mk3(smax=0.03, dist='midquad'), 'cap.03 fade p1': mk3(smax=0.03, fpow=1), 'cap.02 fade': mk3(smax=0.02), 'fxps': lab.fxps}
for k, f in C.items():
    print('%-18s' % k, ' '.join('%s d2 %.2f/%.3f' % ((t,) + tuple(np.array(stab(f, 4, t, 15)[2:]) * [1e3, 1])) for t in lab.TEXTURES), flush=True)
ks = ['exit', 'cap.03 nofade', 'cap.03 fade', 'cap.03 fade quad', 'cap.03 fade p1', 'cap.02 fade']
lab.save_sheet('r4_bump_e15.png', {k: C[k] for k in ks}, tex='bump', elev=15, azim=140)
lab.save_sheet('r4_bricks_e15.png', {k: C[k] for k in ks}, tex='bricks', elev=15, azim=25)
T = SynTex('posts', make('posts')); sheet('syn4_posts_e15.png', {k: C[k] for k in ks}, T, 15, 25)
T = SynTex('ridges', make('ridges')); sheet('syn4_ridges_e30.png', {k: C[k] for k in ks}, T, 30, 25)
