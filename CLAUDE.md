# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FauxPas is a self-contained WebGL 2.0 bump mapping demo implementing multiple parallax mapping and self-shadowing techniques. It includes a reference implementation of the Fast Approximate Parallax Shadows (FXPS) paper (`paper.html`) and supporting figures (`appendix.html`).

## Running

No build step. Serve the root directory with any HTTP server (needed for texture loading):

```
python3 -m http.server 8000
```

Or use `serve.py` for cross-origin isolation headers (enables high-resolution `performance.now()` timers):

```
python3 serve.py
```

Open `index.html` in a browser. Click any canvas to pause/resume rotation.

## File layout

| File | Purpose |
|---|---|
| `bump_mapping.js` | Shaders, WebGL setup, render loop, matrix library, multi-instance UI logic |
| `index.html` | Canvas strip + controls UI |
| `paper.html` | FXPS paper (text only) |
| `appendix.html` | FXPS paper figures (Canvas 2D rendered) |
| `serve.py` | Dev HTTP server that adds COOP/COEP headers |
| `bump_normal.png` / `bump_diffuse.png` | "Classic" texture set |
| `bricks_normal.png` / `bricks_diffuse.png` | "Bricks" texture set |
| `rocks_normal.png` / `rocks_diffuse.png` | "Rocks" texture set |

## Architecture

### Multi-instance system

The page supports up to **4 side-by-side canvases** (`MAX_INSTANCES = 4`), each with its own WebGL context and independent settings. This lets users compare techniques directly.

- `shared` — global object holding `time`, `paused`, `instances[]`, `selectedIndex`
- `createInstance(canvasEl)` — allocates a WebGL context, compiles shaders, uploads mesh buffers and textures, returns an `inst` object
- `renderInstance(inst, time)` — renders one frame into `inst.canvas`
- `addInstance()` / `removeInstance(index)` — DOM and GL lifecycle management
- Controls are bound once (`bindControlEvents()`) and write to `shared.instances[shared.selectedIndex].settings`
- `syncControlsFromInstance(inst)` — syncs all control DOM elements when the selected instance changes
- The render loop uses `setInterval` at 15 ms, not `requestAnimationFrame`

Each `inst` object carries:
- `inst.gl` — the WebGL2 context
- `inst.settings` — per-instance state (shading_type, shadow_type, scale, steps, …)
- `inst.tex_sets[0..2]` — three `{norm, diffuse}` texture pairs, one per texture set
- `inst.vbo_pos/tang/bitang/uv`, `inst.index_buffer` — mesh buffers
- `inst.pgm` — compiled shader program
- `inst.frame_time_avg` — exponential moving average of frame time (used for stats overlay)

### Shader structure (inside `bump_mapping.js`)

The shader uses **GLSL ES 3.00** (`#version 300 es`), i.e., WebGL 2. Do not use WebGL 1 constructs (`texture2D`, `gl_FragColor`, `varying`, `attribute`) when editing shader source.

**Vertex shader** (`vert_src`): transforms positions and constructs a TBN matrix to convert light and view positions into tangent space. The light position is hardcoded at `vec3(1, 2, 0)` in world space. The camera is always at the origin.

**Fragment shader** (`frag_src`) has three layers:

1. **Parallax UV functions** — perturb texture coordinates based on view direction. Selected by the `type` uniform:

| `type` | Name | Function |
|---|---|---|
| 0 | None | (flat shading) |
| 1 | Normal mapping | (no UV offset) |
| 2 | Parallax mapping | `parallax_uv()` — single-sample height offset |
| 3 | Steep parallax | `parallax_uv()` — linear ray march |
| 4 | POM | `parallax_uv()` — linear march + interpolation |
| 5 | Iterative | `getParallaxOffset()` — Premecz convergence method |

2. **Shadow functions** — ray-march along the light direction through the height field. Selected by the `shadow_type` uniform:

| `shadow_type` | Name in UI | GLSL function |
|---|---|---|
| 0 | None | — |
| 1 | Hard POM | `pomHardShadow` |
| 2 | Soft POM | `pomSoftShadow` |
| 3 | HAPS | `heightAdaptiveShadow` (height-adaptive α) |
| 4 | Contact | `contactHardeningShadow` |
| 5 | Binary Search | `binarySearchShadow` |
| 6 | Cone Traced | `coneTracedShadow` |
| 7 | Relief | `reliefMappingShadow` |
| 8 | FXPS | `fastApproximateShadow` (fixed α = `fxps_alpha`) |

3. **`main()`** — dispatches to the selected parallax and shadow functions, then applies Lambertian shading with a constant 0.3 ambient term.

### Texture packing

All three texture sets share the same layout:

- `tex_norm` (TEXTURE0): RGB = tangent-space normal, **A = height (0 = bottom, 1 = top)**
- `tex_diffuse` (TEXTURE1): RGB = diffuse color

The fragment shader reads height as `texture(tex_norm, uv).a`. The `texDepth()` helper inverts this to depth convention (`1.0 - height`).

There is **no separate depth/height texture**. The third file referenced in some old documentation (`bump_depth.png`) no longer exists.

### Key conventions

- Height is in the **alpha channel** of the normal map. Surface height = `texture(tex_norm, uv).a`. Depth = `1.0 - height`.
- All parallax/shadow loops use `for (int i = 0; i < 32; i++)` with an early `break` against `num_layers` or `shadow_steps`. This keeps loops bounded by a compile-time constant (a requirement inherited from WebGL 1 that is maintained for clarity).
- Matrices are flat 16-element arrays in column-major order (OpenGL convention). The JS matrix functions are written as their mathematical transposes to compensate.
- The `norm_mtx` uniform is `transpose(inverse(model_mtx))`, computed on the JS side via `mtx_transpose(mtx_inverse(model))`.
- The FXPS technique (`fastApproximateShadow`, `shadow_type == 8`) uses a power-law step distribution `t = pow(i/N, alpha)` with `1/i` weighting and `strength = N` normalization. The exponent `alpha` is exposed as the `fxps_alpha` slider (default 0.5). See `paper.html` for the derivation.
- HAPS (`heightAdaptiveShadow`, `shadow_type == 3`) is the height-adaptive variant that mixes alpha per-fragment based on surface height.

### Adding a new parallax technique

1. Write the GLSL function in `frag_src`.
2. Add an `else if (type == N)` branch in `main()` (or in `parallax_uv()`).
3. Add a radio button in `index.html` with the new string value.
4. Add the corresponding `case` in `shadingTypeToInt()` and update `SHADING_LABELS` in `bump_mapping.js`.
5. Add a cost entry in `update_cost_labels_from()` if relevant.

### Adding a new shadow technique

1. Write the GLSL function in `frag_src`.
2. Add an `else if (shadow_type == N)` branch in `main()`.
3. Add a radio button in `index.html` with the new string value.
4. Add the corresponding `case` in `shadowTypeToInt()` and update `SHADOW_LABELS`.
5. Add a cost entry in `update_cost_labels_from()`.

### Matrix library (`bump_mapping.js`)

Hand-rolled, no external dependencies:

| Function | Description |
|---|---|
| `mtx_identity()` | 4×4 identity |
| `mtx_mul(a, b)` | Matrix multiplication |
| `mtx_transpose(a)` | Transpose |
| `mtx_inverse(m)` | Cofactor-expansion inverse |
| `mtx_translation(x,y,z)` | Translation matrix |
| `mtx_rotation_x(r)` / `mtx_rotation_y(r)` | Rotation matrices |
| `mtx_perspective(fov_y, aspect, near, far)` | Perspective projection |

### Texture sets

| UI label | `tex_set` index | Normal map | Diffuse map |
|---|---|---|---|
| Classic | 0 | `bump_normal.png` | `bump_diffuse.png` |
| Bricks | 1 | `bricks_normal.png` | `bricks_diffuse.png` |
| Rocks | 2 | `rocks_normal.png` | `rocks_diffuse.png` |

Each instance loads all three sets into its own GL context on startup. Texture loading is asynchronous; a solid red 1×1 pixel is used as the placeholder until the image loads.
