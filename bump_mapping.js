var canvas;
var gl;
var time_paused = false;
var time;

var vbo_pos, attr_pos;
var vbo_tang, attr_tang;
var vbo_bitang, attr_bitang;
var vbo_uv, attr_uv;
var index_buffer;
var num_indices;
var tex_norm, tex_diffuse;
var pgm;

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
*/
uniform int type;
uniform int shadow_type;
uniform int show_tex;
uniform float depth_scale;
uniform float parallax_bias;
uniform float num_layers;
uniform float shadow_steps;

in vec2 frag_uv;
in vec3 ts_light_pos;
in vec3 ts_view_pos;
in vec3 ts_frag_pos;

out vec4 fragColor;


vec2 parallax_uv(vec2 uv, vec3 view_dir)
{
    if (type == 2) {
        // Parallax mapping
        float depth = texture(tex_norm, uv).a;
        vec2 p = view_dir.xy * (depth * depth_scale) / view_dir.z;
        return uv - p;
    } else {
        float layer_depth = 1.0 / num_layers;
        float cur_layer_depth = 0.0;
        vec2 delta_uv = view_dir.xy * depth_scale / (view_dir.z * num_layers);
        vec2 cur_uv = uv;

        float depth_from_tex = texture(tex_norm, cur_uv).a;

        for (int i = 0; i < 32; i++) {
            cur_layer_depth += layer_depth;
            cur_uv -= delta_uv;
            depth_from_tex = texture(tex_norm, cur_uv).a;
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
            float prev = texture(tex_norm, prev_uv).a - cur_layer_depth
                         + layer_depth;
            float weight = next / (next - prev);
            return mix(cur_uv, prev_uv, weight);
        }
    }
}

vec2 getParallaxOffset(vec2 uv, vec3 eyeDir)
{
    vec3 view = vec3(-eyeDir.xy, eyeDir.z);
    vec3 ray = vec3(0.0);

    for (int i = 0; i < 32; i++)
    {
        if (float(i) >= num_layers) break;
        vec4 texSample = texture(tex_norm, uv + ray.xy);
        float sampledHeight = texSample.a * depth_scale - parallax_bias;
        float normalZ = texSample.b * 2.0 - 1.0;
        float heightDiff = sampledHeight - ray.z;
        ray += view * heightDiff * normalZ;
    }

    return ray.xy;
}

float pomHardShadow(vec2 uv, vec3 lightDir)
{
    float surfaceDepth = texture(tex_norm, uv).a;
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / num_layers;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;

    for (int i = 0; i < 32; i++) {
        if (float(i) >= num_layers) break;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texture(tex_norm, cur_uv).a;
        if (sampleDepth < rayDepth) {
            return 0.0;
        }
    }
    return 1.0;
}

float pomSoftShadow(vec2 uv, vec3 lightDir)
{
    float surfaceDepth = texture(tex_norm, uv).a;
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / num_layers;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;
    float maxOcclusion = 0.0;

    for (int i = 0; i < 32; i++) {
        if (float(i) >= num_layers) break;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texture(tex_norm, cur_uv).a;
        float occlusion = (rayDepth - sampleDepth) / (float(i + 1) * depthPerStep);
        maxOcclusion = max(maxOcclusion, occlusion);
    }
    return clamp(1.0 - maxOcclusion, 0.0, 1.0);
}

float fastApproximateShadow(vec2 uv, vec3 lightDir)
{
    vec2 rayStep = lightDir.xy * depth_scale;
    float heightStep = lightDir.z * depth_scale;
    float surfaceHeight = 1.0 - texture(tex_norm, uv).a;

    // Height-adaptive exponent: valleys (h~0) -> low alpha (far-field),
    // peaks (h~1) -> high alpha (contact shadows)
    float alpha = mix(0.5, 2.0, surfaceHeight);

    float shadow = 0.0;
    for (int i = 1; i <= 32; i++) {
        if (float(i) > shadow_steps) break;
        float t = pow(float(i) / shadow_steps, alpha);
        float rayHeight = surfaceHeight + heightStep * t;
        float sampleHeight = 1.0 - texture(tex_norm, uv + rayStep * t).a;
        shadow = max(shadow, (sampleHeight - rayHeight) / float(i));
    }

    // Scale-invariant normalization: strength = N makes sample count a pure quality knob
    return clamp(1.0 - shadow * shadow_steps, 0.0, 1.0);
}

float contactHardeningShadow(vec2 uv, vec3 lightDir)
{
    float surfaceDepth = texture(tex_norm, uv).a;
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / num_layers;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;
    float penumbraScale = 0.5;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;
    float shadow = 1.0;

    for (int i = 0; i < 32; i++) {
        if (float(i) >= num_layers) break;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texture(tex_norm, cur_uv).a;
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
    float surfaceDepth = texture(tex_norm, uv).a;
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / num_layers;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;
    float prevRayDepth = rayDepth;
    vec2 prev_uv = cur_uv;

    bool found = false;
    for (int i = 0; i < 32; i++) {
        if (float(i) >= num_layers) break;
        prev_uv = cur_uv;
        prevRayDepth = rayDepth;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texture(tex_norm, cur_uv).a;
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
        float sampleDepth = texture(tex_norm, mid_uv).a;
        if (sampleDepth < mid_depth) {
            hi_uv = mid_uv;
            hi_depth = mid_depth;
        } else {
            lo_uv = mid_uv;
            lo_depth = mid_depth;
        }
    }

    float finalSample = texture(tex_norm, (lo_uv + hi_uv) * 0.5).a;
    float finalRay = (lo_depth + hi_depth) * 0.5;
    return finalSample < finalRay ? 0.0 : 1.0;
}

float coneTracedShadow(vec2 uv, vec3 lightDir)
{
    float surfaceDepth = texture(tex_norm, uv).a;
    if (surfaceDepth < 0.01) return 1.0;

    float depthPerStep = surfaceDepth / num_layers;
    vec2 uvPerStep = lightDir.xy * depth_scale * depthPerStep / lightDir.z;
    float coneSlope = 0.5;

    float rayDepth = surfaceDepth;
    vec2 cur_uv = uv;
    float minRatio = 1.0;

    for (int i = 1; i <= 32; i++) {
        if (float(i) > num_layers) break;
        cur_uv += uvPerStep;
        rayDepth -= depthPerStep;
        float sampleDepth = texture(tex_norm, cur_uv).a;
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
    float surfaceDepth = texture(tex_norm, uv).a;
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
        float sampleDepth = texture(tex_norm, cur_uv).a;
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
        float sampleDepth = texture(tex_norm, mid_uv).a;
        if (sampleDepth < mid_depth) {
            hi_uv = mid_uv;
            hi_depth = mid_depth;
        } else {
            lo_uv = mid_uv;
            lo_depth = mid_depth;
        }
    }

    float finalRay = (lo_depth + hi_depth) * 0.5;
    float finalSample = texture(tex_norm, (lo_uv + hi_uv) * 0.5).a;
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

// Estimates worst-case depth texture samples for each technique and updates
// the cost labels next to each radio button. Bump mapping uses num_layers (steps
// slider); fast approximate and relief mapping shadows use shadow_steps instead.
function update_cost_labels() {
    const N = parseFloat(document.getElementById("steps").value);
    const S = parseFloat(document.getElementById("shadow_steps").value);

    // tex_norm samples per parallax technique (depth in .a, normal in .rgb; excludes final shading lookup)
    const shading_costs = {
        'diffuse':   '0 tex',
        'normal':    '0 tex',
        'parallax':  '1 tex',
        'steep':     `\u2264${1 + N} tex`,   // early exit possible
        'pom':       `\u2264${N + 2} tex`,   // early exit + 1 extra for interpolation
        'iterative': `${N} tex`,               // N iterations × 1 fetch (depth in .a, normalZ in .b)
    };

    // Additional tex_norm samples per shadow technique
    const shadow_costs = {
        'none':             '0 tex',
        'hard':             `\u2264${1 + N} tex`,    // early exit on first occluder
        'soft':             `${1 + N} tex`,           // no early exit
        'iterative_shadow': `${1 + S} tex`,           // no early exit, uses shadow_steps
        'contact':          `\u2264${1 + N} tex`,    // early exit on first occluder
        'binary':           `\u2264${N + 10} tex`,   // linear ≤N + 8 bisect + 1 final
        'cone':             `${1 + N} tex`,           // no early exit
        'relief':           `\u2264${S + 7} tex`,    // linear ≤S + 5 bisect + 1 final, uses shadow_steps
    };

    for (const [val, cost] of Object.entries(shading_costs)) {
        const el = document.getElementById(`cost_shading_${val}`);
        if (el) el.textContent = `(${cost})`;
    }
    for (const [val, cost] of Object.entries(shadow_costs)) {
        const el = document.getElementById(`cost_shadow_${val}`);
        if (el) el.textContent = `(${cost})`;
    }
}

function initBumpMapping() {
    canvas = document.getElementById("gl_canvas");
    canvas.onclick = function (e) {
        time_paused = !time_paused;
    }

    // Init WebGL 2 context
    {
        gl = null;
        try { gl = canvas.getContext("webgl2"); }
        catch (_) { }
    }

    if (!gl) {
        alert("Unable to initialize WebGL 2. Your browser may not support it.");
        return;
    }

    // Init GL flags
    {
        gl.clearColor(1.0, 1.0, 1.0, 1.0);
        gl.clearDepth(1.0);
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.enable(gl.CULL_FACE);
    }

    // Init shaders
    {
        var frag = get_shader(gl, frag_src, true);
        var vert = get_shader(gl, vert_src, false);
        pgm = gl.createProgram();
        gl.attachShader(pgm, vert);
        gl.attachShader(pgm, frag);
        gl.linkProgram(pgm);

        if (!gl.getProgramParameter(pgm, gl.LINK_STATUS)) {
            alert("Unable to initialize the shader program: " +
                gl.getProgramInfoLog(pgm));
        }

        gl.useProgram(pgm);
        attr_pos = gl.getAttribLocation(pgm, "vert_pos");
        gl.enableVertexAttribArray(attr_pos);
        attr_tang = gl.getAttribLocation(pgm, "vert_tang");
        gl.enableVertexAttribArray(attr_tang);
        attr_bitang = gl.getAttribLocation(pgm, "vert_bitang");
        gl.enableVertexAttribArray(attr_bitang);
        attr_uv = gl.getAttribLocation(pgm, "vert_uv");
        gl.enableVertexAttribArray(attr_uv);
    }

    // Init meshes
    {
        // Positions
        vbo_pos = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo_pos);
        var verts = [
            -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, // Front
            -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, // Back
            1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, // Right
            -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, // Left
            -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, // Top
            -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, // Bottom
        ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);

        // Tangents
        vbo_tang = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo_tang);
        var tangs = [
            1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, // Front
            -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, // Back
            0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, // Right
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, // Left
            1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, // Top
            1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, // Bottom
        ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tangs), gl.STATIC_DRAW);

        // Bitangents
        vbo_bitang = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo_bitang);
        var bitangs = [
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, // Front
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, // Back
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, // Right
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, // Left
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, // Top
            0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, // Bot
        ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bitangs), gl.STATIC_DRAW);

        // UVs
        vbo_uv = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo_uv);
        var uvs = [
            0, 1, 1, 0, 0, 0, 1, 1, // Front
            1, 1, 0, 0, 1, 0, 0, 1, // Back
            1, 1, 0, 0, 0, 1, 1, 0, // Right
            0, 1, 1, 0, 1, 1, 0, 0, // Left
            0, 0, 1, 1, 0, 1, 1, 0, // Top
            0, 1, 1, 0, 0, 0, 1, 1, // Bottom
        ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uvs), gl.STATIC_DRAW);

        index_buffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index_buffer);
        var indices = [
            0, 1, 2, 0, 3, 1, // Front
            4, 6, 5, 4, 5, 7, // Back
            8, 9, 10, 8, 11, 9, // Right
            12, 14, 13, 12, 13, 15, // Left
            16, 18, 17, 16, 17, 19, // Top
            20, 21, 22, 20, 23, 21, // Bottom
        ];
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices),
            gl.STATIC_DRAW);
        num_indices = indices.length;
    }

    // Init textures
    {
        tex_norm = load_texture("bump_normal.png");
        tex_diffuse = load_texture("bump_diffuse.png");
    }

    time = 0;
    setInterval(update_and_render, 15);

    document.getElementById("steps").addEventListener("input", update_cost_labels);
    document.getElementById("shadow_steps").addEventListener("input", update_cost_labels);
    update_cost_labels();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBumpMapping);
} else {
    initBumpMapping();
}

function update_and_render() {
    if (!time_paused) {
        time++;
    }

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    var a = mtx_perspective(45, 680.0 / 382.0, 0.1, 100.0);
    var b = mtx_translation(0, 0, -4.5);
    var c = mtx_rotation_x(0.4);
    var d = mtx_rotation_y(time * 0.0075);

    var model = mtx_mul(mtx_mul(b, c), d);

    {
        var uni = gl.getUniformLocation(pgm, "model_mtx");
        gl.uniformMatrix4fv(uni, false, model);
    }

    {
        var uni = gl.getUniformLocation(pgm, "norm_mtx");
        gl.uniformMatrix4fv(uni, false, mtx_transpose(mtx_inverse(model)));
    }

    {
        var uni = gl.getUniformLocation(pgm, "proj_mtx");
        gl.uniformMatrix4fv(uni, false, mtx_mul(a, model));
    }

    {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, tex_norm);
        var uni = gl.getUniformLocation(pgm, "tex_norm");
        gl.uniform1i(uni, 0);
    }

    {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, tex_diffuse);
        var uni = gl.getUniformLocation(pgm, "tex_diffuse");
        gl.uniform1i(uni, 1);
    }


    {
        var type = 0;
        switch (document.querySelector('input[name="shading_type"]:checked').value) {
            case "normal": type = 1; break;
            case "parallax": type = 2; break;
            case "steep": type = 3; break;
            case "pom": type = 4; break;
            case "iterative": type = 5; break;
        }

        var step = document.getElementById("scale_control");
        if (type < 2) {
            step.style.visibility = "hidden";
        } else {
            step.style.visibility = "visible";
        }

        var biasCtrl = document.getElementById("bias_control");
        biasCtrl.style.visibility = (type == 5) ? "visible" : "hidden";

        var step = document.getElementById("step_control");
        if (type < 3) {
            step.style.visibility = "hidden";
        } else {
            step.style.visibility = "visible";
        }

        var uni = gl.getUniformLocation(pgm, "type");
        gl.uniform1i(uni, type);
    }

    {
        var scale = 0.01 * parseFloat(document.getElementById("scale").value);
        document.getElementById("scale_val").textContent = scale.toFixed(2);
        var uni = gl.getUniformLocation(pgm, "depth_scale");
        gl.uniform1f(uni, scale);
    }

    {
        var bias = 0.01 * parseFloat(document.getElementById("bias").value);
        document.getElementById("bias_val").textContent = bias.toFixed(2);
        var uni = gl.getUniformLocation(pgm, "parallax_bias");
        gl.uniform1f(uni, bias);
    }

    {
        var steps = parseFloat(document.getElementById("steps").value);
        document.getElementById("steps_val").textContent = steps;
        var uni = gl.getUniformLocation(pgm, "num_layers");
        gl.uniform1f(uni, steps);
    }

    {
        var show_tex = document.getElementById('show_tex').checked;
        var uni = gl.getUniformLocation(pgm, "show_tex");
        gl.uniform1i(uni, show_tex);
    }

    {
        var shadow = 0;
        switch (document.querySelector('input[name="shadow_type"]:checked').value) {
            case "hard": shadow = 1; break;
            case "soft": shadow = 2; break;
            case "iterative_shadow": shadow = 3; break;
            case "contact": shadow = 4; break;
            case "binary": shadow = 5; break;
            case "cone": shadow = 6; break;
            case "relief": shadow = 7; break;
        }
        var uni = gl.getUniformLocation(pgm, "shadow_type");
        gl.uniform1i(uni, shadow);

        var shadowStepCtrl = document.getElementById("shadow_step_control");
        shadowStepCtrl.style.visibility = (shadow == 3 || shadow == 7) ? "visible" : "hidden";
    }

    {
        var shadowSteps = parseFloat(document.getElementById("shadow_steps").value);
        document.getElementById("shadow_steps_val").textContent = shadowSteps;
        var uni = gl.getUniformLocation(pgm, "shadow_steps");
        gl.uniform1f(uni, shadowSteps);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo_pos);
    gl.vertexAttribPointer(attr_pos, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo_tang);
    gl.vertexAttribPointer(attr_tang, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo_bitang);
    gl.vertexAttribPointer(attr_bitang, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, vbo_uv);
    gl.vertexAttribPointer(attr_uv, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index_buffer);
    gl.drawElements(gl.TRIANGLES, num_indices, gl.UNSIGNED_SHORT, 0);
}

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

function load_texture(img_path) {
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
