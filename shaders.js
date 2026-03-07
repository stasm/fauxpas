// ---------------------------------------------------------------------------
// Shaders (shared across all instances)
// ---------------------------------------------------------------------------
export var vert_src = `#version 300 es
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

export var frag_src = `#version 300 es
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
    vec3 ray = lightDir * depth_scale;
    float surfaceHeight = texture(tex_norm, uv).a;

    float shadow = 0.0;
    for (int i = 1; i <= 32; i++) {
        if (float(i) > shadow_steps) break;
        float t = pow(float(i) / shadow_steps, fxps_alpha);
        float rayHeight = surfaceHeight + ray.z * t;
        float sampleHeight = texture(tex_norm, uv + ray.xy * t).a;
        shadow = max(shadow, (sampleHeight - rayHeight) / float(i));
    }

    return clamp(1.0 - shadow * shadow_steps, 0.0, 1.0);
}

float heightAdaptiveShadow(vec2 uv, vec3 lightDir)
{
    float surfaceHeight = texture(tex_norm, uv).a;
    vec3 ray = lightDir * depth_scale * exp(0.5 - surfaceHeight);

    float shadow = 0.0;
    for (int i = 1; i <= 32; i++) {
        if (float(i) > shadow_steps) break;
        float t = pow(float(i) / shadow_steps, fxps_alpha);
        float rayHeight = surfaceHeight + ray.z * t;
        float sampleHeight = texture(tex_norm, uv + ray.xy * t).a;
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
    else if (shadow_type == 3) shadow = heightAdaptiveShadow(uv, light_dir);
    else if (shadow_type == 4) shadow = contactHardeningShadow(uv, light_dir);
    else if (shadow_type == 5) shadow = binarySearchShadow(uv, light_dir);
    else if (shadow_type == 6) shadow = coneTracedShadow(uv, light_dir);
    else if (shadow_type == 7) shadow = reliefMappingShadow(uv, light_dir);
    else if (shadow_type == 8) shadow = fastApproximateShadow(uv, light_dir);

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
