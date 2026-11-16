from fam import *
for n in lab.TEXTURES:
    T=lab.Tex(n); h=T.mips[0][...,3]
    print(n,T.size,'h pct',np.percentile(h,[0,5,25,50,75,95,100]).round(3))
    g=np.hypot(*np.gradient(h))*T.size*0.08
    print('  slope(ds=.08) pct',np.percentile(g,[50,75,90,99]).round(3))
    nz=T.mips[0][...,2]*2-1; print('  nz pct',np.percentile(nz,[1,10,50]).round(3))
    for l in range(1,5):
        nn=T.mips[l][...,:3]*2-1; print('  mip',l,'|n| mean',np.linalg.norm(nn,axis=-1).mean().round(3))
