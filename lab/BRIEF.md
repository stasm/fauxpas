# Shadow lab brief (shared by all exploration agents)

## Goal
Design an ultrafast self-shadowing term for height-mapped surfaces that gives a
**believable grounding effect** (objects/bumps look anchored: dark contact
shadows on the far-from-light side of raised features, crevices darkened) at a
budget of **about 4 texture samples (range 2–6)**. It will be paired with
*iterative parallax mapping* (`getParallaxOffset` in `shaders.js`), which is
itself a cheap, approximate, few-sample technique. "Plausible and stable" beats
"accurate but noisy". It must not band, pop or crawl as the light or surface rotates.

The incumbent is FXPS (`fastApproximateShadow` in `shaders.js`). Its author
considers the math shoehorned. Known problems:
- Its ray is `lightDir * depth_scale` in *both* uv and normalized-height units,
  so the ray rises `depth_scale`× too slowly vs. the physical geometry.
- `t = pow(i/N, alpha)` with `1/i` weights and `*N` normalisation has no
  geometric meaning; quality does **not** improve with N (see baseline table).
- No use of mips, of the normal that comes free with every fetch, or of where
  the ray leaves the height volume.
Do not assume any of FXPS's structure is right. Challenge everything, including
whether ray marching toward the light is the right primitive at all.

## Facts about the runtime (WebGL2, GLSL ES 3.00)
- One texture `tex_norm`: RGB = tangent-space normal (encoded 0..1), A = height
  (1 = top). Every fetch returns normal AND height.
- The fetch at the shaded uv is free (shading reads it anyway): `T.center(uv)`.
- Mipmaps are currently NOT generated, but can be enabled (`textureLod` is fair
  game; textures are power-of-two, REPEAT wrap). The lab's `T.sample(uv, lod)`
  emulates trilinear `textureLod` with a box-filtered chain.
- Tangent-space light dir `L` (unit, L.z > 0 when lit). uv offset along the
  light is `uv + L.xy * tau`; physical geometry is in the docstring of `lab.py`.
- depth_scale default 0.08 (UI range ~0.01–0.2).
- No temporal accumulation / TAA. Per-pixel dither is allowed only if the
  result looks clean without filtering (be honest about visible noise).
- Loops must be bounded by a compile-time constant with early `break`.
- Cost model: texture fetches dominate. Avoid `pow`, `exp`, `normalize`, `sqrt`
  inside the loop when a cheaper form works; a handful of MADs per sample is fine.

## Harness
`lab/lab.py` (read its docstring first). Reference shadows (area light, radius
0.05 rad, brute force) are precomputed in `lab/cache/` for 3 textures × 4
elevations × 3 azimuths at depth_scale 0.08 — **do not delete or recompute the
cache** (it takes 15 min); references for other depth_scale values are computed
on demand and are slow (~10 s each), so use them sparingly.

```python
import sys; sys.path.insert(0, 'lab')
import lab
def my_shadow(T, uv, L, ds, N):
    c = T.center(uv); h0 = c[..., 3]          # free
    ...  T.sample(uv + ..., lod)               # counted
    return vis
print(lab.fmt('mine', lab.evaluate(my_shadow, 4)))      # also N=2,3,6
lab.save_sheet('/path/out.png', {'mine': my_shadow, 'fxps': lab.fxps}, tex='rocks', elev=30, azim=25, N=4)
```
Baseline (N=4): no shadow lit=0.168 blur=0.141 · FXPS lit=0.104 blur=0.076 flicker=1.11 · POM soft lit=0.110 blur=0.082.
`evaluate(..., verbose=True)` prints per-config rows; check low elevations (15°, 30°) and each texture separately — averages hide failures.

The metrics are a guide, not the goal. The reference is a physically soft
shadow; a 4-sample method cannot match it everywhere, and a slightly
"wrong" but smooth, well-grounded result can be preferable. **Look at the
images** (Read the PNG contact sheets) for all three textures at low and high
light elevations and judge: banding, stair-steps, light leaking, detached or
floating shadows, over-darkening of flat areas, popping.

## Rules
- Work ONLY inside your own directory `lab/<your-agent-dir>/` (create it). Do
  not edit `lab/lab.py`, `shaders.js`, or any other repo file. Do not commit,
  do not push, do not spawn further agents.
- Keep your numpy candidate faithful to what GLSL can do per-pixel.
- Report negative results honestly — "idea X does not help, because Y" is a
  useful outcome.

## Deliverable (your final message, be concise and concrete)
1. Best candidate: numpy function file path + equivalent **GLSL ES 3.00
   function** `float xyzShadow(vec2 uv, vec3 lightDir, vec4 center)` (center =
   the free texel at uv), using uniforms `tex_norm`, `depth_scale`,
   `shadow_steps`.
2. Metrics table at N = 2, 3, 4, 6 (all columns from `lab.fmt`) + per-texture
   notes; path of 2–3 contact sheets that best show strengths and failures.
3. One-paragraph derivation: what each term *means* geometrically. No
   shoehorning — if a constant is a tuned fudge, say so.
4. What you tried that failed and why. What you'd combine it with.
