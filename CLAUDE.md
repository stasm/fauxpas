# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FauxPas is a self-contained WebGL 1.0 bump mapping demo implementing multiple parallax mapping and self-shadowing techniques. It includes a reference implementation of the Fast Approximate Parallax Shadows (FXPS) paper (`paper.html`).

## Running

No build step. Serve the root directory with any HTTP server (needed for texture loading):

```
python3 -m http.server 8000
```

Open `index.html` in a browser. Click the canvas to pause/resume rotation.

## Architecture

Everything lives in two files:

- **`bump_mapping.js`** — Contains the vertex shader, fragment shader (as JS template strings), WebGL setup, render loop, and a hand-rolled matrix math library. The fragment shader implements all parallax and shadow techniques.
- **`index.html`** — UI controls (radio buttons, sliders) and canvas. Controls are read each frame in `update_and_render()` via DOM queries.

### Shader structure (inside `bump_mapping.js`)

The vertex shader transforms positions into tangent space via a TBN matrix. The fragment shader has three layers:

1. **Parallax UV functions** (`parallax_uv`, `getParallaxOffset`) — perturb texture coordinates based on view direction. Selected by the `type` uniform (0–5).
2. **Shadow functions** (`pomHardShadow`, `pomSoftShadow`, `fastApproximateShadow`, `contactHardeningShadow`, `binarySearchShadow`, `coneTracedShadow`, `reliefMappingShadow`) — ray-march along the light direction through the height field. Selected by the `shadow_type` uniform (0–7).
3. **`main()`** — dispatches to the selected parallax and shadow functions, then applies Lambertian shading.

### Adding a new technique

1. Write the GLSL function in the fragment shader string (`frag_src`).
2. Add an `else if (shadow_type == N)` or `if (type == N)` branch in `main()`.
3. Add a radio button in `index.html` with a new `value`.
4. Add the corresponding `case` in the JS switch inside `update_and_render()`.

### Key conventions

- Height maps use the `.r` channel. The depth textures store depth (0 = top, 1 = bottom), so surface height is `1.0 - texture2D(tex_depth, uv).r`.
- All parallax/shadow loops are bounded by `for (int i = 0; i < 32; i++)` with an early `break` against `num_layers`, since WebGL 1.0 requires constant loop bounds.
- Matrices are flat 16-element arrays in column-major order (OpenGL convention). The JS code reads as row-major transposed.
- The light position is hardcoded at `(1, 2, 0)` in world space (vertex shader line 58).

### Textures

Three PNG files (`bump_normal.png`, `bump_diffuse.png`, `bump_depth.png`) are loaded asynchronously and bound to `TEXTURE0`–`TEXTURE2`.
