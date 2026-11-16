from hz import *
for n in lab.TEXTURES:
    T=lab.Tex(n); h=T.mips[0][...,3]
    print(n, 'h mean %.3f p5 %.3f p50 %.3f p95 %.3f'%(h.mean(), *np.percentile(h,[5,50,95])), 'grad/texel %.4f'%np.abs(np.diff(h,axis=1)).mean())
    for e in lab.ELEVS:
        ref=lab.cached_reference(T,e,25.0); print('   elev',e,'ref mean %.3f frac<0.5 %.3f'%(ref.mean(),(ref<0.5).mean()))
fns={'phys uni exit':mk(dist='uni',reach='exit'),'fxps':lab.fxps}
for tex in lab.TEXTURES:
    lab.save_sheet('s_phys_%s_e15.png'%tex, fns, tex=tex, elev=15, azim=25, N=4)
    lab.save_sheet('s_phys_%s_e50.png'%tex, fns, tex=tex, elev=50, azim=25, N=4)
