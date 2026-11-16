from hz import *
# 1. FXPS alpha=1 == horizon test with light slope ds*m and ramp width 1/|Lxy| in slope
def fxps_as_horizon(T, uv, L, ds, N):
    h0 = T.center(uv)[..., 3]; lxy = math.hypot(L[0], L[1]); S = None
    for i in range(1, N + 1):
        d = ds * lxy * i / N                       # uv distance: reach = ds*|Lxy|, uniform
        s = ds * (T.sample(uv + L[:2] / lxy * d)[..., 3] - h0) / d   # true horizon slope
        S = s if S is None else np.maximum(S, s)
    return np.clip(1.0 - lxy * (S - ds * L[2] / lxy), 0, 1)
T = lab.Tex('rocks'); uv = T.uv_grid(); L = lab.light(30, 25)
a = lab.fxps(T, uv, L, 0.08, 4, alpha=1.0); b = fxps_as_horizon(T, uv, L, 0.08, 4)
print('max |fxps(a=1) - horizon form| =', np.abs(a - b).max())
# 2. what does each FXPS ingredient buy?  (physical test, FXPS reach) and (FXPS test, physical reach)
run('phys test, reach=ds (fxps reach), uni', mk(dist='uni', reach='fxps'), detail=True)
run('phys test, fixed d=0.08 uni', mk(dist='uni', reach='fixed', dmax=0.08), detail=True)
run('fxps a=1', lambda T,uv,L,ds,N: lab.fxps(T,uv,L,ds,N,alpha=1.0), Ns=(2,3,4,6))
run('fxps a=.5', lab.fxps, Ns=(2,3,4,6))
# 3. range clamp on exit ray
for dm in [0.04, 0.08, 0.12, 0.2]:
    run('exitc mid dmax=%.2f' % dm, mk(dist='mid', reach='exitc', dmax=dm), detail=True)
run('exit mid', mk(dist='mid', reach='exit'), detail=True)
