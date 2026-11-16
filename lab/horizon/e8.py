from hz3 import *
C = {'A exit mid': mk3(smax=0), 'b.5': mk3(smax=0, beta=0.5), 'b1': mk3(smax=0, beta=1.0), 'b1 uni': mk3(smax=0, beta=1.0, dist='uni'),
     'b1 one': mk3(smax=0, beta=1.0, onesided=True), 'cap.03 b.5': mk3(beta=0.5), 'cap.03 b1': mk3(beta=1.0), 'cap.04 b1': mk3(beta=1.0, smax=0.04)}
for k, f in C.items(): full(k, f)
ks = ['A exit mid', 'b.5', 'b1', 'b1 one', 'cap.03 b1']
for kind in ['posts', 'ridges']:
    T = SynTex(kind, make(kind)); sheet('syn3_%s_e15.png' % kind, {k: C[k] for k in ks}, T, 15, 25)
for tex in lab.TEXTURES:
    lab.save_sheet('r3_%s_e15.png' % tex, {k: C[k] for k in ks}, tex=tex, elev=15, azim=140)
    lab.save_sheet('r3_%s_e30.png' % tex, {k: C[k] for k in ks}, tex=tex, elev=30, azim=25)
