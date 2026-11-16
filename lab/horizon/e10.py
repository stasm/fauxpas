from hz3 import *
from PIL import Image
T = lab.Tex('bump'); h = T.mips[0][..., 3]
print('pyramid row profile', np.round(h[70, 140:215:5], 2))
C = {'r.05': mk3(smax=0.03, fade=0), 'r.1': mk3(smax=0.03, fade=0, r=0.1), 'r.15': mk3(smax=0.03, fade=0, r=0.15), 'b2': mk3(smax=0.03, fade=0, beta=2.0),
     'r.1 exit': mk3(smax=0, fade=0, r=0.1)}
lab.save_sheet('r5_bump_e15.png', C, tex='bump', elev=15, azim=140)
lab.save_sheet('r5_bump_e30.png', C, tex='bump', elev=30, azim=140)
for f in ['r5_bump_e15.png', 'r5_bump_e30.png']:
    im = Image.open(f); n = im.width // 256
    out = Image.new('L', (200 * n, 110))
    for i in range(n): out.paste(im.crop((i * 256 + 40, 20, i * 256 + 240, 130)), (i * 200, 0))
    out.resize((out.width * 3 // 2, 165), Image.BILINEAR).save('zoom_' + f)
