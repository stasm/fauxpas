from fam import *
b=dict(sched='uni',wmode='rise',wk=0.5,wmin=0.003,lodmode='none',combine='prod')
mm=dict(b,lodmode='match',lodbias=-1)
cases=[('lod0 everywhere (no mips)',b,dict(base=0.0)),('base=hw lod',b,{}),('base=hw+1',b,dict(base='+1')),('hw-1',b,dict(base='-1')),
       ('match-1 max(hw)',mm,{}),('match-1 + hw',mm,dict(basemode='add')),('match-2 + hw',dict(mm,lodbias=-2),dict(basemode='add'))]
for m in (2,4,8):
    print('--- 1 px =',m,'texels')
    for name,kw,o in cases:
        o=dict(o)
        if o.get('base')=='+1': o['base']=math.log2(m)+1
        if o.get('base')=='-1': o['base']=math.log2(m)-1
        r=[];hf=[]
        for tex in lab.TEXTURES:
            for e in (15,30,50,70):
                for a in (25.0,265.0):
                    e_,vis,refm=minif(kw,4,m,tex,e,a,**o); r.append(e_)
                    # pixel-scale noise: energy of the candidate's laplacian relative to the reference's
                    lap=lambda x: x-0.25*(np.roll(x,1,0)+np.roll(x,-1,0)+np.roll(x,1,1)+np.roll(x,-1,1))
                    hf.append(np.abs(lap(vis)).mean()/max(np.abs(lap(refm)).mean(),1e-6))
        print(f"{name:28s} rmse={np.mean(r):.4f}  hf_ratio={np.mean(hf):.2f}")
