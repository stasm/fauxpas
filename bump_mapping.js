// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------
var shared = {
    time: 0,
    paused: false,
    instances: [],
    selectedIndex: 0,
};

var MAX_INSTANCES = 4;

// ---------------------------------------------------------------------------
// Label maps
// ---------------------------------------------------------------------------
var SHADING_LABELS = {
    diffuse: 'None', normal: 'Normal Map', parallax: 'Parallax',
    steep: 'Steep', pom: 'POM', iterative: 'Iterative',
};
var SHADOW_LABELS = {
    none: 'No Shadow', hard: 'Hard POM', soft: 'Soft POM',
    fxps: 'FXPS', fxps_fixed: 'FXPS Fixed \u03b1', contact: 'Contact',
    binary: 'Binary Search', cone: 'Cone Traced', relief: 'Relief',
};

// ---------------------------------------------------------------------------
// Shaders (module-level constants, shared across all instances)
// ---------------------------------------------------------------------------
var vert_src = `#version 300 es
precision highp float;

in vec3 vert_pos;
in vec3 vert_tang;
in vec3 vert_bitang;
in vec2 vert_uv;

uniform mat4 model_mtx;
uniform mat4 norm_mtx;
uniform mat4 proj_mtx;

out vec2 frag_uv;
out vec3 ts_light_pos; // Tangent space values
out vec3 ts_view_pos;  //
out vec3 ts_frag_pos;  //

void main(void)
{
    gl_Position = proj_mtx * vec4(vert_pos, 1.0);
    ts_frag_pos = vec3(model_mtx * vec4(vert_pos, 1.0));
    vec3 vert_norm = cross(vert_bitang, vert_tang);

    vec3 t = normalize(mat3(norm_mtx) * vert_tang);
    vec3 b = normalize(mat3(norm_mtx) * vert_bitang);
    vec3 n = normalize(mat3(norm_mtx) * vert_norm);
    mat3 tbn = transpose(mat3(t, b, n));

    vec3 light_pos = vec3(1, 2, 0);
    ts_light_pos = tbn * light_pos;
    // Our camera is always at the origin
    ts_view_pos = tbn * vec3(0, 0, 0);
    ts_frag_pos = tbn * ts_frag_pos;

    frag_uv = vert_uv;
}
`;

var frag_src = `#version 300 es
precision highp float;

uniform sampler2D tex_norm;
uniform sampler2D tex_diffuse;
/*
    The type is controlled by the radio buttons below the canvas.
    0 = No bump mapping
    1 = Normal mapping
    2 = Parallax mapping
    3 = Steep parallax mapping
    4 = Parallax occlusion mapping
    5 = Iterative parallax mapping
*/
uniform int type;
uniform int shadow_type;
uniform int show_tex;
uniform float depth_scale;
uniform float parallax_bias;
uniform float num_layers;
uniform float shadow_steps;
uniform float fxps_alpha;

in vec2 frag_uv;
in vec3 ts_light_pos;
in vec3 ts_view_pos;
in vec3 ts_frag_pos;

out vec4 fragColor;

float texDepth(vec2 uv) { return 1.0 - texture(tex_norm, uv).a; }

vec2 parallax_uv(vec2 uv, vec3 view_dir)
{
    if (type == 2) {
        // Parallax mapping
        float depth = texDepth(uv);
        vec2 p = view_dir.xy * (depth * depth_scale) / view_dir.z;
        return uv - p;
    } else {
        float layer_depth = 1.0 / num_layers;
        float cur_layer_depth = 0.0;
        vec2 delta_uv = view_dir.xy * depth_scale / (view_dir.z * num_layers);
        vec2 cur_uv = uv;

        float depth_from_tex = texDepth(cur_uv);

        for (int i = 0; i < 32; i++) {
            cur_layer_depth += layer_depth;
            cur_uv -= delta_uv;
            depth_from_tex = texDepth(cur_uv);
            if (depth_from_tex < cur_layer_depth) {
                break;
            }
        }

        if (type == 3) {
            // Steep parallax mapping
            return cur_uv;
        } else {
            // Parallax occlusion mapping
            vec2 prev_uv = cur_uv + delta_uv;
            float next = depth_from_tex - cur_layer_depth;
            float prev = texDepth(prev_uv) - cur_layer_depth
                         + layer_depth;
            float weight = next / (next - prev);
            return mix(cur_uv, prev_uv, weight);
        }
    }
}

vec2 getParallaxOffset(vec2 uv, vec3 eyeDir)
{
    vec3 ray = vec3(0.0);

    for (int i = 0; i < 32; i++)
    {
        if (float(i) >= num_layers) break;
        vec4 texSample = texture(tex_norm, uv + ray.xy);
        float sampledHeight = texSample.a * depth_scale - parallax_bias;
        float heightDiff = sampledHeight - ray.z;
        ray += eyeDir * heightDiff * texSample.z;
    }

    return ray.xy;
}


float pomHardShadow(vec2 uv, vec3 lightDir)
{
    float surfaceDepth = texDepth(uv);
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / shadow_steps;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;

    for (int i = 0; i < 32; i++) {
        if (float(i) >= shadow_steps) break;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texDepth(cur_uv);
        if (sampleDepth < rayDepth) {
            return 0.0;
        }
    }
    return 1.0;
}

float pomSoftShadow(vec2 uv, vec3 lightDir)
{
    float surfaceDepth = texDepth(uv);
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / shadow_steps;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;
    float maxOcclusion = 0.0;

    for (int i = 0; i < 32; i++) {
        if (float(i) >= shadow_steps) break;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texDepth(cur_uv);
        float occlusion = (rayDepth - sampleDepth) / (float(i + 1) * depthPerStep);
        maxOcclusion = max(maxOcclusion, occlusion);
    }
    return clamp(1.0 - maxOcclusion, 0.0, 1.0);
}

float fastApproximateShadow(vec2 uv, vec3 lightDir)
{
    vec3 step = lightDir * depth_scale;
    float surfaceHeight = texture(tex_norm, uv).a;

    // Height-adaptive exponent: valleys (h~0) -> low alpha (far-field),
    // peaks (h~1) -> high alpha (contact shadows)
    float alpha = mix(0.5, 2.0, surfaceHeight);

    float shadow = 0.0;
    for (int i = 1; i <= 32; i++) {
        if (float(i) > shadow_steps) break;
        float t = pow(float(i) / shadow_steps, alpha);
        float rayHeight = surfaceHeight + step.z * t;
        float sampleHeight = texture(tex_norm, uv + step.xy * t).a;
        shadow = max(shadow, (sampleHeight - rayHeight) / float(i));
    }

    // Scale-invariant normalization: strength = N makes sample count a pure quality knob
    return clamp(1.0 - shadow * shadow_steps, 0.0, 1.0);
}

float fastApproximateShadowFixed(vec2 uv, vec3 lightDir)
{
    vec3 step = lightDir * depth_scale;
    float surfaceHeight = texture(tex_norm, uv).a;

    float shadow = 0.0;
    for (int i = 1; i <= 32; i++) {
        if (float(i) > shadow_steps) break;
        float t = pow(float(i) / shadow_steps, fxps_alpha);
        float rayHeight = surfaceHeight + step.z * t;
        float sampleHeight = texture(tex_norm, uv + step.xy * t).a;
        shadow = max(shadow, (sampleHeight - rayHeight) / float(i));
    }

    return clamp(1.0 - shadow * shadow_steps, 0.0, 1.0);
}

float contactHardeningShadow(vec2 uv, vec3 lightDir)
{
    float surfaceDepth = texDepth(uv);
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / shadow_steps;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;
    float penumbraScale = 0.5;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;
    float shadow = 1.0;

    for (int i = 0; i < 32; i++) {
        if (float(i) >= shadow_steps) break;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texDepth(cur_uv);
        if (sampleDepth < rayDepth) {
            float occluderDist = float(i + 1) * depthPerStep;
            float totalDist = surfaceDepth;
            shadow = clamp(occluderDist / (totalDist * penumbraScale), 0.0, 1.0);
            return shadow;
        }
    }
    return 1.0;
}

float binarySearchShadow(vec2 uv, vec3 lightDir)
{
    float surfaceDepth = texDepth(uv);
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / shadow_steps;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;
    float prevRayDepth = rayDepth;
    vec2 prev_uv = cur_uv;

    bool found = false;
    for (int i = 0; i < 32; i++) {
        if (float(i) >= shadow_steps) break;
        prev_uv = cur_uv;
        prevRayDepth = rayDepth;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texDepth(cur_uv);
        if (sampleDepth < rayDepth) {
            found = true;
            break;
        }
    }

    if (!found) return 1.0;

    // Binary search refinement within the intersection interval
    vec2 lo_uv = prev_uv;
    float lo_depth = prevRayDepth;
    vec2 hi_uv = cur_uv;
    float hi_depth = rayDepth;

    for (int i = 0; i < 8; i++) {
        vec2 mid_uv = (lo_uv + hi_uv) * 0.5;
        float mid_depth = (lo_depth + hi_depth) * 0.5;
        float sampleDepth = texDepth(mid_uv);
        if (sampleDepth < mid_depth) {
            hi_uv = mid_uv;
            hi_depth = mid_depth;
        } else {
            lo_uv = mid_uv;
            lo_depth = mid_depth;
        }
    }

    float finalSample = texDepth((lo_uv + hi_uv) * 0.5);
    float finalRay = (lo_depth + hi_depth) * 0.5;
    return finalSample < finalRay ? 0.0 : 1.0;
}

float coneTracedShadow(vec2 uv, vec3 lightDir)
{
    float surfaceDepth = texDepth(uv);
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / shadow_steps;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;
    float coneSlope = 0.5;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;
    float minRatio = 1.0;

    for (int i = 1; i <= 32; i++) {
        if (float(i) > shadow_steps) break;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texDepth(cur_uv);
        float coneRadius = coneSlope * float(i) * depthPerStep;
        float penetration = rayDepth - sampleDepth;
        if (penetration > 0.0) {
            float ratio = coneRadius / penetration;
            minRatio = min(minRatio, ratio);
        }
    }

    return clamp(minRatio, 0.0, 1.0);
}

float reliefMappingShadow(vec2 uv, vec3 lightDir)
{
    float surfaceDepth = texDepth(uv);
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / shadow_steps;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;
    float prevRayDepth = rayDepth;
    vec2 prev_uv = cur_uv;

    bool found = false;
    for (int i = 0; i < 32; i++) {
        if (float(i) >= shadow_steps) break;
        prev_uv = cur_uv;
        prevRayDepth = rayDepth;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texDepth(cur_uv);
        if (sampleDepth < rayDepth) {
            found = true;
            break;
        }
    }

    if (!found) return 1.0;

    // Binary search refinement (5 iterations)
    vec2 lo_uv = prev_uv;
    float lo_depth = prevRayDepth;
    vec2 hi_uv = cur_uv;
    float hi_depth = rayDepth;

    for (int i = 0; i < 5; i++) {
        vec2 mid_uv = (lo_uv + hi_uv) * 0.5;
        float mid_depth = (lo_depth + hi_depth) * 0.5;
        float sampleDepth = texDepth(mid_uv);
        if (sampleDepth < mid_depth) {
            hi_uv = mid_uv;
            hi_depth = mid_depth;
        } else {
            lo_uv = mid_uv;
            lo_depth = mid_depth;
        }
    }

    float finalRay = (lo_depth + hi_depth) * 0.5;
    float finalSample = texDepth((lo_uv + hi_uv) * 0.5);
    float depthDiff = finalRay - finalSample;
    return clamp(1.0 - depthDiff * shadow_steps, 0.0, 1.0);
}

void main(void)
{
    vec3 light_dir = normalize(ts_light_pos - ts_frag_pos);
    vec3 view_dir = normalize(ts_view_pos - ts_frag_pos);

    // Only perturb the texture coordinates if a parallax technique is selected
    vec2 uv;
    if (type == 5) {
        uv = frag_uv + getParallaxOffset(frag_uv, view_dir);
    } else {
        uv = (type < 2) ? frag_uv : parallax_uv(frag_uv, view_dir);
    }

    vec3 albedo = texture(tex_diffuse, uv).rgb;
    if (show_tex == 0) { albedo = vec3(1,1,1); }
    vec3 ambient = 0.3 * albedo;

    float shadow = 1.0;
    if (shadow_type == 1) shadow = pomHardShadow(uv, light_dir);
    else if (shadow_type == 2) shadow = pomSoftShadow(uv, light_dir);
    else if (shadow_type == 3) shadow = fastApproximateShadow(uv, light_dir);
    else if (shadow_type == 4) shadow = contactHardeningShadow(uv, light_dir);
    else if (shadow_type == 5) shadow = binarySearchShadow(uv, light_dir);
    else if (shadow_type == 6) shadow = coneTracedShadow(uv, light_dir);
    else if (shadow_type == 7) shadow = reliefMappingShadow(uv, light_dir);
    else if (shadow_type == 8) shadow = fastApproximateShadowFixed(uv, light_dir);

    if (type == 0) {
        // No bump mapping
        vec3 norm = vec3(0,0,1);
        float diffuse = max(dot(light_dir, norm), 0.0);
        fragColor = vec4(diffuse * shadow * albedo + ambient, 1.0);

    } else {
        // Normal mapping
        vec3 norm = normalize(texture(tex_norm, uv).rgb * 2.0 - 1.0);
        float diffuse = max(dot(light_dir, norm), 0.0);
        fragColor = vec4(diffuse * shadow * albedo + ambient, 1.0);
    }
}
`;

// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------
function shadingTypeToInt(val) {
    switch (val) {
        case "normal":    return 1;
        case "parallax":  return 2;
        case "steep":     return 3;
        case "pom":       return 4;
        case "iterative": return 5;
        default:          return 0;
    }
}

function shadowTypeToInt(val) {
    switch (val) {
        case "hard":       return 1;
        case "soft":       return 2;
        case "fxps":       return 3;
        case "contact":    return 4;
        case "binary":     return 5;
        case "cone":       return 6;
        case "relief":     return 7;
        case "fxps_fixed": return 8;
        default:           return 0;
    }
}

// ---------------------------------------------------------------------------
// Cost labels
// ---------------------------------------------------------------------------
function update_cost_labels_from(N, S) {
    var shading_costs = {
        'diffuse':   '0 tex',
        'normal':    '0 tex',
        'parallax':  '1 tex',
        'steep':     '\u2264' + (1 + N) + ' tex',
        'pom':       '\u2264' + (N + 2) + ' tex',
        'iterative': N + ' tex',
    };
    var shadow_costs = {
        'none':       '0 tex',
        'hard':       '\u2264' + (1 + S) + ' tex',
        'soft':       (1 + S) + ' tex',
        'fxps':       (1 + S) + ' tex',
        'fxps_fixed': (1 + S) + ' tex',
        'contact':    '\u2264' + (1 + S) + ' tex',
        'binary':     '\u2264' + (S + 10) + ' tex',
        'cone':       (1 + S) + ' tex',
        'relief':     '\u2264' + (S + 7) + ' tex',
    };
    for (var val in shading_costs) {
        var el = document.getElementById('cost_shading_' + val);
        if (el) el.textContent = '(' + shading_costs[val] + ')';
    }
    for (var val in shadow_costs) {
        var el = document.getElementById('cost_shadow_' + val);
        if (el) el.textContent = '(' + shadow_costs[val] + ')';
    }
}

function update_cost_labels() {
    var N = parseFloat(document.getElementById("steps").value);
    var S = parseFloat(document.getElementById("shadow_steps").value);
    update_cost_labels_from(N, S);
}

// ---------------------------------------------------------------------------
// Instance factory
// ---------------------------------------------------------------------------
function createInstance(canvasEl) {
    var inst = {};
    inst.canvas = canvasEl;
    inst.frame_time_avg = 0;

    // Default settings
    inst.settings = {
        tex_set:      0,
        show_tex:     true,
        shading_type: 'iterative',
        shadow_type:  'fxps',
        scale:        8,
        bias_ratio:   100,
        steps:        12,
        shadow_steps: 3,
        fxps_alpha:   1.0,
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
        { norm: load_texture(gl, "rocks_normal.png"),  diffuse: load_texture(gl, "rocks_difuse.png") },
    ];

    return inst;
}

// ---------------------------------------------------------------------------
// Per-instance render
// ---------------------------------------------------------------------------
function renderInstance(inst, time) {
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
        case "iterative": parallax_samples = N; break;
    }
    var shadow_samples = 0;
    switch (s.shadow_type) {
        case "none":       shadow_samples = 0; break;
        case "hard":       shadow_samples = 1 + S; break;
        case "soft":       shadow_samples = 1 + S; break;
        case "fxps":       shadow_samples = 1 + S; break;
        case "fxps_fixed": shadow_samples = 1 + S; break;
        case "contact":    shadow_samples = 1 + S; break;
        case "binary":     shadow_samples = S + 10; break;
        case "cone":       shadow_samples = 1 + S; break;
        case "relief":     shadow_samples = S + 7; break;
    }
    var total_samples = parallax_samples + shadow_samples;
    var time_str = inst.frame_time_avg < 1
        ? (inst.frame_time_avg * 1000).toFixed(0) + " \u00B5s"
        : inst.frame_time_avg.toFixed(2) + " ms";
    inst.statsOverlayEl.textContent = time_str + " | ~" + total_samples + " tex/frag";
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
function startRenderLoop() {
    setInterval(function () {
        if (!shared.paused) {
            shared.time++;
        }
        for (var i = 0; i < shared.instances.length; i++) {
            renderInstance(shared.instances[i], shared.time);
        }
    }, 15);
}

// ---------------------------------------------------------------------------
// Title bar
// ---------------------------------------------------------------------------
function updateTitleBar(inst) {
    var s = inst.settings;
    inst.titleTextEl.textContent =
        (SHADING_LABELS[s.shading_type] || s.shading_type)
        + '  \u2502  '
        + (SHADOW_LABELS[s.shadow_type] || s.shadow_type);
}

// ---------------------------------------------------------------------------
// Selection & control sync
// ---------------------------------------------------------------------------
function selectInstance(index) {
    shared.selectedIndex = index;
    for (var i = 0; i < shared.instances.length; i++) {
        var cls = shared.instances[i].titleBarEl.classList;
        if (i === index) cls.add('selected'); else cls.remove('selected');
    }
    syncControlsFromInstance(shared.instances[index]);
}

function syncControlsFromInstance(inst) {
    var s = inst.settings;

    // Texture set radios
    var texNames = ['bump', 'bricks', 'rocks'];
    var texRadio = document.querySelector('input[name="tex_set"][value="' + texNames[s.tex_set] + '"]');
    if (texRadio) texRadio.checked = true;

    // Show tex checkbox
    document.getElementById('show_tex').checked = s.show_tex;

    // Shading type radio
    var shadingRadio = document.querySelector('input[name="shading_type"][value="' + s.shading_type + '"]');
    if (shadingRadio) shadingRadio.checked = true;

    // Shadow type radio
    var shadowRadio = document.querySelector('input[name="shadow_type"][value="' + s.shadow_type + '"]');
    if (shadowRadio) shadowRadio.checked = true;

    // Sliders + display spans
    document.getElementById('scale').value = s.scale;
    document.getElementById('scale_val').textContent = (0.01 * s.scale).toFixed(2);

    document.getElementById('bias_ratio').value = s.bias_ratio;
    document.getElementById('bias_ratio_val').textContent = Math.round(s.bias_ratio) + '%';

    document.getElementById('steps').value = s.steps;
    document.getElementById('steps_val').textContent = s.steps;

    document.getElementById('shadow_steps').value = s.shadow_steps;
    document.getElementById('shadow_steps_val').textContent = s.shadow_steps;

    document.getElementById('fxps_alpha').value = s.fxps_alpha;
    document.getElementById('fxps_alpha_val').textContent = parseFloat(s.fxps_alpha).toFixed(1);

    // Conditional visibility
    var type = shadingTypeToInt(s.shading_type);
    document.getElementById('scale_control').style.visibility = type >= 2 ? 'visible' : 'hidden';
    document.getElementById('bias_ratio_control').style.visibility = type == 5 ? 'visible' : 'hidden';
    document.getElementById('step_control').style.visibility = type >= 3 ? 'visible' : 'hidden';

    var shadow = shadowTypeToInt(s.shadow_type);
    document.getElementById('shadow_step_control').style.visibility = shadow > 0 ? 'visible' : 'hidden';
    document.getElementById('fxps_alpha_control').style.visibility = shadow == 8 ? 'visible' : 'hidden';

    // Cost labels
    update_cost_labels_from(s.steps, s.shadow_steps);
}

// ---------------------------------------------------------------------------
// Control event binding (once, writes to selected instance)
// ---------------------------------------------------------------------------
function bindControlEvents() {
    function sel() { return shared.instances[shared.selectedIndex]; }

    // Texture set radios
    var texSetMap = { bump: 0, bricks: 1, rocks: 2 };
    var texRadios = document.querySelectorAll('input[name="tex_set"]');
    for (var i = 0; i < texRadios.length; i++) {
        texRadios[i].addEventListener('change', function () {
            sel().settings.tex_set = texSetMap[this.value];
        });
    }

    // Show tex checkbox
    document.getElementById('show_tex').addEventListener('change', function () {
        sel().settings.show_tex = this.checked;
    });

    // Shading type radios
    var shadingRadios = document.querySelectorAll('input[name="shading_type"]');
    for (var i = 0; i < shadingRadios.length; i++) {
        shadingRadios[i].addEventListener('change', function () {
            sel().settings.shading_type = this.value;
            var type = shadingTypeToInt(this.value);
            document.getElementById('scale_control').style.visibility = type >= 2 ? 'visible' : 'hidden';
            document.getElementById('bias_ratio_control').style.visibility = type == 5 ? 'visible' : 'hidden';
            document.getElementById('step_control').style.visibility = type >= 3 ? 'visible' : 'hidden';
            updateTitleBar(sel());
        });
    }

    // Shadow type radios
    var shadowRadios = document.querySelectorAll('input[name="shadow_type"]');
    for (var i = 0; i < shadowRadios.length; i++) {
        shadowRadios[i].addEventListener('change', function () {
            sel().settings.shadow_type = this.value;
            var shadow = shadowTypeToInt(this.value);
            document.getElementById('shadow_step_control').style.visibility = shadow > 0 ? 'visible' : 'hidden';
            document.getElementById('fxps_alpha_control').style.visibility = shadow == 8 ? 'visible' : 'hidden';
            updateTitleBar(sel());
        });
    }

    // Sliders
    document.getElementById('scale').addEventListener('input', function () {
        sel().settings.scale = parseFloat(this.value);
        document.getElementById('scale_val').textContent = (0.01 * sel().settings.scale).toFixed(2);
    });
    document.getElementById('bias_ratio').addEventListener('input', function () {
        sel().settings.bias_ratio = parseFloat(this.value);
        document.getElementById('bias_ratio_val').textContent = Math.round(sel().settings.bias_ratio) + '%';
    });
    document.getElementById('steps').addEventListener('input', function () {
        sel().settings.steps = parseFloat(this.value);
        document.getElementById('steps_val').textContent = sel().settings.steps;
        update_cost_labels();
    });
    document.getElementById('shadow_steps').addEventListener('input', function () {
        sel().settings.shadow_steps = parseFloat(this.value);
        document.getElementById('shadow_steps_val').textContent = sel().settings.shadow_steps;
        update_cost_labels();
    });
    document.getElementById('fxps_alpha').addEventListener('input', function () {
        sel().settings.fxps_alpha = parseFloat(this.value);
        document.getElementById('fxps_alpha_val').textContent = sel().settings.fxps_alpha.toFixed(1);
    });
}

// ---------------------------------------------------------------------------
// Add / remove instances
// ---------------------------------------------------------------------------
function addInstance() {
    if (shared.instances.length >= MAX_INSTANCES) return;

    var strip = document.getElementById('canvas_strip');
    var addBtn = document.getElementById('add_canvas_btn');

    // Container
    var container = document.createElement('div');
    container.className = 'canvas-container';

    // Title bar
    var titleBar = document.createElement('div');
    titleBar.className = 'canvas-title-bar';
    var titleText = document.createElement('span');
    titleText.className = 'title-text';
    titleBar.appendChild(titleText);

    // Close button (hidden if only one instance — managed in removeInstance)
    var closeBtn = document.createElement('button');
    closeBtn.className = 'canvas-close-btn';
    closeBtn.textContent = '\u00d7';
    closeBtn.title = 'Remove';
    titleBar.appendChild(closeBtn);
    container.appendChild(titleBar);

    // Canvas
    var canvasEl = document.createElement('canvas');
    canvasEl.width = 2;   // will be synced to display size on first render
    canvasEl.height = 2;
    container.appendChild(canvasEl);

    // Stats overlay
    var statsEl = document.createElement('div');
    statsEl.className = 'stats-overlay';
    container.appendChild(statsEl);

    strip.insertBefore(container, addBtn);

    var inst = createInstance(canvasEl);
    if (!inst) {
        container.remove();
        return;
    }

    inst.containerEl = container;
    inst.titleBarEl = titleBar;
    inst.titleTextEl = titleText;
    inst.closeBtnEl = closeBtn;
    inst.statsOverlayEl = statsEl;

    shared.instances.push(inst);

    // Wire events — close over inst object, not index
    titleBar.addEventListener('click', function (e) {
        if (e.target === closeBtn) return;
        selectInstance(shared.instances.indexOf(inst));
    });
    canvasEl.addEventListener('click', function () {
        shared.paused = !shared.paused;
    });
    closeBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        removeInstance(shared.instances.indexOf(inst));
    });

    updateTitleBar(inst);
    selectInstance(shared.instances.indexOf(inst));
    updateCloseButtons();
    updateAddButton();
}

function removeInstance(index) {
    if (index < 0 || shared.instances.length <= 1) return;
    var inst = shared.instances[index];
    inst.containerEl.remove();
    shared.instances.splice(index, 1);

    var newSel = Math.min(shared.selectedIndex, shared.instances.length - 1);
    selectInstance(newSel);
    updateCloseButtons();
    updateAddButton();
}

function updateCloseButtons() {
    var hide = shared.instances.length <= 1;
    for (var i = 0; i < shared.instances.length; i++) {
        shared.instances[i].closeBtnEl.style.display = hide ? 'none' : '';
    }
}

function updateAddButton() {
    var btn = document.getElementById('add_canvas_btn');
    btn.style.display = shared.instances.length >= MAX_INSTANCES ? 'none' : '';
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function initBumpMapping() {
    document.getElementById('add_canvas_btn').addEventListener('click', function () {
        addInstance();
    });
    bindControlEvents();
    addInstance();
    startRenderLoop();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBumpMapping);
} else {
    initBumpMapping();
}

// ---------------------------------------------------------------------------
// Shader compilation
// ---------------------------------------------------------------------------
function get_shader(gl, src, is_frag) {
    var shader = gl.createShader(is_frag ? gl.FRAGMENT_SHADER :
        gl.VERTEX_SHADER);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert("An error occurred compiling the shaders: " +
            gl.getShaderInfoLog(shader));
        return null;
    }

    return shader;
}

function load_texture(gl, img_path) {
    var tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
        new Uint8Array([255, 0, 0, 255])); // red

    var img = new Image();
    img.onload = function () {
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    }
    img.src = img_path;
    return tex;
}

/*
    Matrix utility functions

    Note that OpenGL expects column-major arrays, but JavaScript, is row-major.
    So each matrix in code is written as the transpose of its mathematical form.
*/
function mtx_zero() {
    return [
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    ];
}

function mtx_identity() {
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ];
}

function mtx_mul(a, b) {
    var c = mtx_zero();

    for (var i = 0; i < 4; i++) {
        for (var j = 0; j < 4; j++) {
            for (var k = 0; k < 4; k++) {
                c[i + j * 4] += a[i + k * 4] * b[k + j * 4];
            }
        }
    }

    return c;
}

function mtx_transpose(a) {
    var b = mtx_zero();

    for (var i = 0; i < 4; i++) {
        for (var j = 0; j < 4; j++) {
            b[i + j * 4] = a[j + i * 4];
        }
    }

    return b;
}

function mtx_inverse(m) {
    var inv = mtx_zero();
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0) {
        console.log("Error: Non-invertible matrix");
        return mtx_zero();
    }

    det = 1.0 / det;
    for (var i = 0; i < 16; i++) {
        inv[i] *= det;
    }
    return inv;
}

function mtx_translation(x, y, z) {
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        x, y, z, 1
    ];
}

function mtx_rotation_x(r) {
    return [
        1, 0, 0, 0,
        0, Math.cos(r), Math.sin(r), 0,
        0, -Math.sin(r), Math.cos(r), 0,
        0, 0, 0, 1
    ];
}

function mtx_rotation_y(r) {
    return [
        Math.cos(r), 0, -Math.sin(r), 0,
        0, 1, 0, 0,
        Math.sin(r), 0, Math.cos(r), 0,
        0, 0, 0, 1
    ];
}

function mtx_perspective(fov_y, aspect, z_near, z_far) {
    var top = z_near * Math.tan(fov_y * Math.PI / 360.0);
    var bot = -top;
    var left = bot * aspect;
    var right = top * aspect;

    var X = 2 * z_near / (right - left);
    var Y = 2 * z_near / (top - bot);
    var A = (right + left) / (right - left);
    var B = (top + bot) / (top - bot);
    var C = -(z_far + z_near) / (z_far - z_near);
    var D = -2 * z_far * z_near / (z_far - z_near);

    return [
        X, 0.0, 0.0, 0.0,
        0.0, Y, 0.0, 0.0,
        A, B, C, -1.0,
        0.0, 0.0, D, 0.0
    ];
}
