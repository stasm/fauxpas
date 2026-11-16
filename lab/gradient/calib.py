import sys; sys.path.insert(0,'/home/user/fauxpas/lab')
import lab, numpy as np
for name in lab.TEXTURES:
    T=lab.Tex(name); m=T.mips[0]; n=T.size
    h=m[...,3]
    nn=m[...,:3]*2-1; raw_len=np.linalg.norm(nn,axis=-1)
    nn=nn/np.maximum(raw_len[...,None],1e-6)
    # central differences, per-uv units (x = column = u, y = row = v)
    dhdu=(np.roll(h,-1,1)-np.roll(h,1,1))*0.5*n
    dhdv=(np.roll(h,-1,0)-np.roll(h,1,0))*0.5*n
    gx=-nn[...,0]/np.maximum(nn[...,2],1e-3); gy=-nn[...,1]/np.maximum(nn[...,2],1e-3)
    print(f"== {name} size={n} h[min,mean,max]={h.min():.3f},{h.mean():.3f},{h.max():.3f} hstd={h.std():.3f} nz mean={nn[...,2].mean():.3f} min={nn[...,2].min():.3f} rawlen mean={raw_len.mean():.3f} std={raw_len.std():.3f}")
    print(f"   rms dh/du={np.sqrt((dhdu**2).mean()):.2f}  rms gx={np.sqrt((gx**2).mean()):.3f}; quantised height levels={len(np.unique(h))}")
    for lab_,d,g in (("u/x",dhdu,gx),("v/y",dhdv,gy)):
        c=np.corrcoef(d.ravel(),g.ravel())[0,1]
        k=(d*g).sum()/(g*g).sum()
        # robust k: median ratio where |g| big
        msk=np.abs(g)>0.2
        kmed=np.median(d[msk]/g[msk]) if msk.sum() else float('nan')
        sign_agree=np.mean(np.sign(d[msk])==np.sign(g[msk]*np.sign(k))) if msk.sum() else float('nan')
        print(f"   {lab_}: corr={c:+.3f}  k_lsq={k:+.2f}  k_median(|g|>.2)={kmed:+.2f}  sign-agree={sign_agree:.3f}  frac|g|>.2={msk.mean():.3f}")
    # cross terms (axis swap check)
    print(f"   cross corr(dhdu,gy)={np.corrcoef(dhdu.ravel(),gy.ravel())[0,1]:+.3f} corr(dhdv,gx)={np.corrcoef(dhdv.ravel(),gx.ravel())[0,1]:+.3f}")
    # smoothed heights: does normal correspond to blurred/sharper height?
    for s in (1,2,4):
        hb=lab.blur(h,s)
        du=(np.roll(hb,-1,1)-np.roll(hb,1,1))*0.5*n; dv=(np.roll(hb,-1,0)-np.roll(hb,1,0))*0.5*n
        print(f"   blur s={s}: corr u={np.corrcoef(du.ravel(),gx.ravel())[0,1]:+.3f} v={np.corrcoef(dv.ravel(),gy.ravel())[0,1]:+.3f}  k_u={(du*gx).sum()/(gx*gx).sum():+.2f}")
    # directional: slope along L for a few azimuths
    for a in lab.AZIMS:
        d=np.array([np.cos(np.radians(a)),np.sin(np.radians(a))])
        dd=dhdu*d[0]+dhdv*d[1]; gg=gx*d[0]+gy*d[1]
        print(f"   az={a}: corr={np.corrcoef(dd.ravel(),gg.ravel())[0,1]:+.3f}", end="")
    print()
