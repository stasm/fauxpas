import { vert_src, frag_src } from './shaders.js';
import { get_shader, load_texture } from './webgl-utils.js';
import {
    mtx_perspective, mtx_translation, mtx_rotation_x,
    mtx_rotation_y, mtx_mul, mtx_transpose, mtx_inverse,
} from './matrix.js';

// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------
export function shadingTypeToInt(val) {
    switch (val) {
        case "normal":    return 1;
        case "parallax":  return 2;
        case "steep":     return 3;
        case "pom":       return 4;
        case "iterative":      return 5;
        case "contact_refinement": return 6;
        default:               return 0;
    }
}

export function shadowTypeToInt(val) {
    switch (val) {
        case "hard":       return 1;
        case "soft":       return 2;
        case "haps":       return 3;
        case "contact":    return 4;
        case "binary":     return 5;
        case "cone":       return 6;
        case "relief":     return 7;
        case "fxps":       return 8;
        case "slope":      return 9;
        case "nms":        return 10;
        case "horizon":    return 11;
        case "fxps_phys":  return 12;
        default:           return 0;
    }
}

// Texture filtering modes, mirroring OpenMW's options
// (SceneManager::setFilterSettings): [min filter, mag filter].
function texFilterParams(gl, mode) {
    switch (mode) {
        case "none":     return [gl.LINEAR, gl.LINEAR];                  // texture mipmap = none
        case "nearest":  return [gl.NEAREST_MIPMAP_NEAREST, gl.NEAREST]; // mag/min = nearest
        case "bilinear": return [gl.LINEAR_MIPMAP_NEAREST, gl.LINEAR];   // texture mipmap = nearest (default)
        default:         return [gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR];    // texture mipmap = linear
    }
}

// ---------------------------------------------------------------------------
// Instance factory
// ---------------------------------------------------------------------------
export function createInstance(canvasEl) {
    var inst = {};
    inst.canvas = canvasEl;
    inst.frame_time_avg = 0;

    // Default settings
    inst.settings = {
        tex_set:      0,
        show_tex:     true,
        tex_filter:   'trilinear',
        shading_type: 'iterative',
        shadow_type:  'fxps',
        scale:        8,
        bias_ratio:   100,
        steps:        12,
        shadow_steps: 3,
        fxps_alpha:   0.5,
        haps_k:       0.5,
    };

    var gl;
    try { gl = canvasEl.getContext("webgl2"); } catch (_) {}
    if (!gl) {
        alert("Unable to initialize WebGL 2. Your browser may not support it.");
        return null;
    }
    inst.gl = gl;

    gl.clearColor(1.0, 1.0, 1.0, 1.0);
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.CULL_FACE);

    // Shaders
    var frag = get_shader(gl, frag_src, true);
    var vert = get_shader(gl, vert_src, false);
    inst.pgm = gl.createProgram();
    gl.attachShader(inst.pgm, vert);
    gl.attachShader(inst.pgm, frag);
    gl.linkProgram(inst.pgm);
    if (!gl.getProgramParameter(inst.pgm, gl.LINK_STATUS)) {
        alert("Shader link error: " + gl.getProgramInfoLog(inst.pgm));
    }
    gl.useProgram(inst.pgm);

    inst.attr_pos = gl.getAttribLocation(inst.pgm, "vert_pos");
    gl.enableVertexAttribArray(inst.attr_pos);
    inst.attr_tang = gl.getAttribLocation(inst.pgm, "vert_tang");
    gl.enableVertexAttribArray(inst.attr_tang);
    inst.attr_bitang = gl.getAttribLocation(inst.pgm, "vert_bitang");
    gl.enableVertexAttribArray(inst.attr_bitang);
    inst.attr_uv = gl.getAttribLocation(inst.pgm, "vert_uv");
    gl.enableVertexAttribArray(inst.attr_uv);

    // Mesh buffers
    inst.vbo_pos = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, inst.vbo_pos);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1,
        -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1,
        1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1,
        -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1,
        -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1,
        -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1,
    ]), gl.STATIC_DRAW);

    inst.vbo_tang = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, inst.vbo_tang);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
        1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
    ]), gl.STATIC_DRAW);

    inst.vbo_bitang = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, inst.vbo_bitang);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
    ]), gl.STATIC_DRAW);

    inst.vbo_uv = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, inst.vbo_uv);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        0, 1, 1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 1, 0, 0, 1,
        1, 1, 0, 0, 0, 1, 1, 0,
        0, 1, 1, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 0, 1, 1,
    ]), gl.STATIC_DRAW);

    inst.index_buffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, inst.index_buffer);
    var indices = [
        0, 1, 2, 0, 3, 1,
        4, 6, 5, 4, 5, 7,
        8, 9, 10, 8, 11, 9,
        12, 14, 13, 12, 13, 15,
        16, 18, 17, 16, 17, 19,
        20, 21, 22, 20, 23, 21,
    ];
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    inst.num_indices = indices.length;

    // Textures (each GL context needs its own copies)
    inst.tex_sets = [
        { norm: load_texture(gl, "bump_normal.png"),   diffuse: load_texture(gl, "bump_diffuse.png") },
        { norm: load_texture(gl, "bricks_normal.png"), diffuse: load_texture(gl, "bricks_diffuse.png") },
        { norm: load_texture(gl, "rocks_normal.png"),  diffuse: load_texture(gl, "rocks_diffuse.png") },
    ];

    return inst;
}

// ---------------------------------------------------------------------------
// Per-instance render
// ---------------------------------------------------------------------------
export function renderInstance(inst, time) {
    var gl = inst.gl;
    var s  = inst.settings;

    // Sync backing resolution to display size for sharpness
    var dw = inst.canvas.clientWidth;
    var dh = inst.canvas.clientHeight;
    if (inst.canvas.width !== dw || inst.canvas.height !== dh) {
        inst.canvas.width = dw;
        inst.canvas.height = dh;
        gl.viewport(0, 0, dw, dh);
    }

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Use actual display aspect so the cube is never distorted.
    // The fixed height + FOV means the cube is always the same size vertically;
    // wider canvases just reveal more of the scene on the sides.
    var a = mtx_perspective(45, dw / dh, 0.1, 100.0);
    var b = mtx_translation(0, 0, -4.5);
    var c = mtx_rotation_x(0.4);
    var d = mtx_rotation_y(time * 0.0075);
    var model = mtx_mul(mtx_mul(b, c), d);

    gl.uniformMatrix4fv(gl.getUniformLocation(inst.pgm, "model_mtx"), false, model);
    gl.uniformMatrix4fv(gl.getUniformLocation(inst.pgm, "norm_mtx"), false, mtx_transpose(mtx_inverse(model)));
    gl.uniformMatrix4fv(gl.getUniformLocation(inst.pgm, "proj_mtx"), false, mtx_mul(a, model));

    // Texture set
    var set = inst.tex_sets[s.tex_set];
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, set.norm);
    gl.uniform1i(gl.getUniformLocation(inst.pgm, "tex_norm"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, set.diffuse);
    gl.uniform1i(gl.getUniformLocation(inst.pgm, "tex_diffuse"), 1);

    // Texture filtering (applied every frame: cheap, and covers late async loads)
    var filt = texFilterParams(gl, s.tex_filter);
    for (var unit = 0; unit < 2; unit++) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt[0]);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt[1]);
    }

    // Technique uniforms
    var type = shadingTypeToInt(s.shading_type);
    gl.uniform1i(gl.getUniformLocation(inst.pgm, "type"), type);

    var scale = 0.01 * s.scale;
    gl.uniform1f(gl.getUniformLocation(inst.pgm, "depth_scale"), scale);

    var ratio = 0.01 * s.bias_ratio;
    gl.uniform1f(gl.getUniformLocation(inst.pgm, "parallax_bias"), ratio * scale);

    gl.uniform1f(gl.getUniformLocation(inst.pgm, "num_layers"), s.steps);
    gl.uniform1i(gl.getUniformLocation(inst.pgm, "show_tex"), s.show_tex ? 1 : 0);

    var shadow = shadowTypeToInt(s.shadow_type);
    gl.uniform1i(gl.getUniformLocation(inst.pgm, "shadow_type"), shadow);
    gl.uniform1f(gl.getUniformLocation(inst.pgm, "shadow_steps"), s.shadow_steps);
    gl.uniform1f(gl.getUniformLocation(inst.pgm, "fxps_alpha"), s.fxps_alpha);
    gl.uniform1f(gl.getUniformLocation(inst.pgm, "haps_k"), s.haps_k);

    // Draw
    gl.bindBuffer(gl.ARRAY_BUFFER, inst.vbo_pos);
    gl.vertexAttribPointer(inst.attr_pos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, inst.vbo_tang);
    gl.vertexAttribPointer(inst.attr_tang, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, inst.vbo_bitang);
    gl.vertexAttribPointer(inst.attr_bitang, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, inst.vbo_uv);
    gl.vertexAttribPointer(inst.attr_uv, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, inst.index_buffer);

    var t0 = performance.now();
    gl.drawElements(gl.TRIANGLES, inst.num_indices, gl.UNSIGNED_SHORT, 0);
    gl.finish();
    var frame_ms = performance.now() - t0;
    inst.frame_time_avg = inst.frame_time_avg === 0
        ? frame_ms
        : inst.frame_time_avg * 0.95 + frame_ms * 0.05;

    // Stats overlay
    var N = s.steps;
    var S = s.shadow_steps;
    var parallax_samples = 0;
    switch (s.shading_type) {
        case "diffuse":   parallax_samples = 0; break;
        case "normal":    parallax_samples = 0; break;
        case "parallax":  parallax_samples = 1; break;
        case "steep":     parallax_samples = N; break;
        case "pom":       parallax_samples = N + 1; break;
        case "iterative":      parallax_samples = N; break;
    }
    var shadow_samples = 0;
    switch (s.shadow_type) {
        case "none":       shadow_samples = 0; break;
        case "hard":       shadow_samples = 1 + S; break;
        case "soft":       shadow_samples = 1 + S; break;
        case "fxps":       shadow_samples = 1 + S; break;
        case "haps":       shadow_samples = 1 + S; break;
        case "contact":    shadow_samples = 1 + S; break;
        case "binary":     shadow_samples = S + 10; break;
        case "cone":       shadow_samples = 1 + S; break;
        case "relief":     shadow_samples = S + 7; break;
        case "slope":      shadow_samples = 1 + S; break;
        case "horizon":    shadow_samples = 1 + S; break;
        case "fxps_phys":  shadow_samples = 1 + S; break;
    }
    var total_samples = parallax_samples + shadow_samples;
    var time_str = inst.frame_time_avg < 1
        ? (inst.frame_time_avg * 1000).toFixed(0) + " \u00B5s"
        : inst.frame_time_avg.toFixed(2) + " ms";
    inst.statsOverlayEl.textContent = time_str + " | ~" + total_samples + " tex/frag";
}
