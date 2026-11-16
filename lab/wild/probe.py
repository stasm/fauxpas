import sys; sys.path.insert(0,'/home/user/fauxpas/lab'); import lab, numpy as np
# 1) are normals consistent with height gradient? implied depth scale
for n in lab.TEXTURES:
    T=lab.Tex(n)
    for lv in (0,2,4):
        m=T.mips[lv]; h=m[...,3]; N=m.shape[0]
        gx=(np.roll(h,-1,1)-np.roll(h,1,1))*0.5*N; gy=(np.roll(h,-1,0)-np.roll(h,1,0))*0.5*N
        nn=m[...,:3]*2-1; ln=np.linalg.norm(nn,axis=-1)
        sx=-nn[...,0]/np.maximum(nn[...,2],1e-3); sy=-nn[...,1]/np.maximum(nn[...,2],1e-3)
        # fit s = k*g  -> k = implied ds
        kx=(sx*gx).sum()/(gx*gx).sum(); ky=(sy*gy).sum()/(gy*gy).sum()
        cx=np.corrcoef(sx.ravel(),gx.ravel())[0,1]; cy=np.corrcoef(sy.ravel(),gy.ravel())[0,1]
        print(n,lv,'implied ds x,y',round(kx,4),round(ky,4),'corr',round(cx,3),round(cy,3),'mean|n|',ln.mean().round(3))
# 2) reference stats: mean visibility per elev
for n in lab.TEXTURES:
    T=lab.Tex(n)
    print(n,[round(float(lab.cached_reference(T,e,25.).mean()),3) for e in lab.ELEVS])
