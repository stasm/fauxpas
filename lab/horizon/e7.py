from hz2 import *
C = {'A exit mid': mk2(smax=0), 'B cap.02': mk2(smax=0.02, fade=0), 'C cap.02 fade': mk2(smax=0.02), 'C cap.03 fade': mk2(smax=0.03),
     'D cap.03 quad fade': mk2(smax=0.03, dist='midquad'), 'E cap.03 fade r.1': mk2(smax=0.03, r=0.1), 'fxps': lab.fxps}
for k, f in C.items(): full(k, f)
for kind in ['posts', 'ridges']:
    T = SynTex(kind, make(kind))
    for e in (15, 30):
        sheet('syn2_%s_e%d.png' % (kind, e), {k: C[k] for k in ['A exit mid', 'B cap.02', 'C cap.02 fade', 'D cap.03 quad fade', 'fxps']}, T, e, 25)
for tex in lab.TEXTURES:
    lab.save_sheet('r2_%s_e15.png' % tex, {k: C[k] for k in ['A exit mid', 'C cap.02 fade', 'D cap.03 quad fade', 'fxps']}, tex=tex, elev=15, azim=140)
