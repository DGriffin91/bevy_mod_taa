// References:
// https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail
// http://behindthepixels.io/assets/files/TemporalAA.pdf
// http://leiy.cc/publications/TAA/TAA_EG2020_Talk.pdf
// https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING

#import bevy_pbr::mesh_view_bindings view, globals
#import bevy_pbr::utils PI
#import bevy_pbr::mesh_view_bindings as vb

const PI_SQ: f32 = 9.8696044010893586188344910;

// Controls how much to blend between the current and past samples
// Lower numbers = less of the current sample and more of the past sample = more smoothing
struct TAAUniform {
    inverse_view_proj: mat4x4<f32>, // not jittered
    prev_inverse_view_proj: mat4x4<f32>, // not jittered
    default_history_blend_rate: f32, // Default blend rate to use when no confidence in history
    min_history_blend_rate: f32, // Minimum blend rate allowed, to ensure at least some of the current sample is used
    velocity_rejection: f32,
    depth_rejection_px_radius: f32,
};

#import bevy_core_pipeline::fullscreen_vertex_shader  FullscreenVertexOutput

@group(0) @binding(20) var view_target: texture_2d<f32>;
@group(0) @binding(21) var view_linear_sampler: sampler;
@group(0) @binding(22) var history: texture_2d<f32>;
@group(0) @binding(23) var history_linear_sampler: sampler;
@group(0) @binding(24) var motion_history: texture_2d<f32>;
@group(0) @binding(25) var motion_vectors: texture_2d<f32>;
@group(0) @binding(26) var depth: texture_depth_2d;
@group(0) @binding(27) var<uniform> taa: TAAUniform;

struct Output {
    @location(0) view_target: vec4<f32>,
    @location(1) history: vec4<f32>,
    @location(2) motion_history: vec4<f32>,
};

// https://github.com/NVIDIAGameWorks/RayTracingDenoiser/blob/3c881ae3075f7ca754e22177877335b82e16da5a/Shaders/Include/Common.hlsli#L124
fn world_space_pixel_radius(linear_depth: f32) -> f32 {
    // https://github.com/NVIDIAGameWorks/RayTracingDenoiser/blob/3c881ae3075f7ca754e22177877335b82e16da5a/Source/Sigma.cpp#L107
    let is_orthographic = view.projection[3].w == 1.0;
    let unproject = 1.0 / (0.5 * view.viewport.w * view.projection[1][1]);
    return unproject * select(linear_depth, 1.0, is_orthographic);
}

/// Convert uv [0.0 .. 1.0] coordinate to ndc space xy [-1.0 .. 1.0]
fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return (uv - vec2(0.5)) * vec2(2.0, -2.0);
}

/// Convert a ndc space position to world space
fn position_ndc_to_world(ndc_pos: vec3<f32>) -> vec3<f32> {
    let world_pos = taa.inverse_view_proj * vec4(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Convert a ndc space position to world space
fn position_history_ndc_to_world(ndc_pos: vec3<f32>) -> vec3<f32> {
    let world_pos = taa.prev_inverse_view_proj * vec4(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

fn cubic_b(v: f32) -> vec4<f32> {
    let n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    let s = n * n * n;
    let x = s.x;
    let y = s.y - 4.0 * s.x;
    let z = s.z - 4.0 * s.y + 6.0 * s.x;
    let w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}

fn texture_sample_bicubic_b(tex: texture_2d<f32>, tex_sampler: sampler, uv: vec2<f32>, texture_size: vec2<f32>) -> vec4<f32> {
    var coords = uv * texture_size - 0.5;

    let fxy = fract(coords);
    coords = coords - fxy;

    let xcubic = cubic_b(fxy.x);
    let ycubic = cubic_b(fxy.y);

    let c = coords.xxyy + vec2(-0.5, 1.5).xyxy;
    
    let s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    var offset = c + vec4(xcubic.yw, ycubic.yw) / s;
    
    offset = offset * (1.0 / texture_size).xxyy;
    
    let sample0 = textureSampleLevel(tex, tex_sampler, offset.xz, 0.0);
    let sample1 = textureSampleLevel(tex, tex_sampler, offset.yz, 0.0);
    let sample2 = textureSampleLevel(tex, tex_sampler, offset.xw, 0.0);
    let sample3 = textureSampleLevel(tex, tex_sampler, offset.yw, 0.0);

    let sx = s.x / (s.x + s.y);
    let sy = s.z / (s.z + s.w);

    return mix(mix(sample3, sample2, vec4(sx)), mix(sample1, sample0, vec4(sx)), vec4(sy));
}

fn sinc(x: f32) -> f32 {
    return select(sin(PI * x) / (PI * x), 1.0, abs(x) < 0.001);
}

// 5-sample Catmull-Rom filtering
// Catmull-Rom filtering: https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
// Ignoring corners: https://www.activision.com/cdn/research/Dynamic_Temporal_Antialiasing_and_Upsampling_in_Call_of_Duty_v4.pdf#page=68
// Technically we should renormalize the weights since we're skipping the corners, but it's basically the same result
fn texture_sample_bicubic_catmull_rom(tex: texture_2d<f32>, tex_sampler: sampler, uv: vec2<f32>, texture_size: vec2<f32>) -> vec4<f32> {
    let texel_size = 1.0 / texture_size;
    let sample_position = uv * texture_size;
    let tex_pos1 = floor(sample_position - 0.5) + 0.5;
    let f = sample_position - tex_pos1;

    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    var w12 = w1 + w2;
    var offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    var tex_pos0 = tex_pos1 - 1.0;
    var tex_pos3 = tex_pos1 + 2.0;
    var tex_pos12 = tex_pos1 + offset12;

    tex_pos0 /= texture_size;
    tex_pos3 /= texture_size;
    tex_pos12 /= texture_size;

    var result = vec4(0.0);
    //result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos0.x, tex_pos0.y), 0.0) * w0.x * w0.y;
    result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos12.x, tex_pos0.y), 0.0) * w12.x * w0.y;
    //result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos3.x, tex_pos0.y), 0.0) * w3.x * w0.y;

    result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos0.x, tex_pos12.y), 0.0) * w0.x * w12.y;
    result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos12.x, tex_pos12.y), 0.0) * w12.x * w12.y;
    result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos3.x, tex_pos12.y), 0.0) * w3.x * w12.y;

    //result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos0.x, tex_pos3.y), 0.0) * w0.x * w3.y;
    result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos12.x, tex_pos3.y), 0.0) * w12.x * w3.y;
    //result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos3.x, tex_pos3.y), 0.0) * w3.x * w3.y;

    return result;
}

// TAA is ideally applied after tonemapping, but before post processing
// Post processing wants to go before tonemapping, which conflicts
// Solution: Put TAA before tonemapping, tonemap TAA input, apply TAA, invert-tonemap TAA output
// https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING, slide 20
// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve
fn rcp(x: f32) -> f32 { return 1.0 / x; }
fn max3(x: vec3<f32>) -> f32 { return max(x.r, max(x.g, x.b)); }
fn tonemap(color: vec3<f32>) -> vec3<f32> { return color * rcp(max3(color) + 1.0); }
fn reverse_tonemap(color: vec3<f32>) -> vec3<f32> { return color * rcp(1.0 - max3(color)); }

// The following 3 functions are from Playdead (MIT-licensed)
// https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
fn RGB_to_YCoCg(rgb: vec3<f32>) -> vec3<f32> {
    let y = (rgb.r / 4.0) + (rgb.g / 2.0) + (rgb.b / 4.0);
    let co = (rgb.r / 2.0) - (rgb.b / 2.0);
    let cg = (-rgb.r / 4.0) + (rgb.g / 2.0) - (rgb.b / 4.0);
    return vec3(y, co, cg);
}

fn YCoCg_to_RGB(ycocg: vec3<f32>) -> vec3<f32> {
    let r = ycocg.x + ycocg.y - ycocg.z;
    let g = ycocg.x + ycocg.z;
    let b = ycocg.x - ycocg.y - ycocg.z;
    return saturate(vec3(r, g, b));
}

fn clip_towards_aabb_center(history_color: vec3<f32>, current_color: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;
    let v_clip = history_color - p_clip;
    let v_unit = v_clip / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max3(a_unit);
    if ma_unit > 1.0 {
        return p_clip + (v_clip / ma_unit);
    } else {
        return history_color;
    }
}

fn sample_history(u: f32, v: f32) -> vec3<f32> {
    return textureSampleLevel(history, history_linear_sampler, vec2(u, v), 0.0).rgb;
}

fn sample_view_target(uv: vec2<f32>, texture_size: vec2<f32>) -> vec3<f32> {
    var sample = textureLoad(view_target, vec2<i32>(uv * texture_size), 0).rgb;
#ifdef TONEMAP
    sample = tonemap(sample);
#endif
    return RGB_to_YCoCg(sample);
}

// Dilate edges by picking the closest motion vector from 3x3 neighborhood
// https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING, slide 27
// https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/, Depth Dilation
fn get_closest_motion_vector_depth(uv: vec2<f32>, texture_size: vec2<f32>) -> vec4<f32> {
    var closest_depth = 0.0;
    var farthest_depth = 1.0;
    var closest_offset = vec2(0.0);
#ifndef WEBGL2
    for(var y = -1; y <= 1; y += 1) {
        for(var x = -1; x <= 1; x += 1) {
            let uv_offset = vec2<f32>(vec2<i32>(x, y)) / texture_size;
            let depth = textureLoad(depth, vec2<i32>((uv + uv_offset) * texture_size), 0);
            if (depth > closest_depth) {
                closest_depth = depth;
                closest_offset = uv_offset;
            }
            farthest_depth = min(farthest_depth, depth);
        }
    }
#endif //WEBGL2
    return vec4(textureLoad(motion_vectors, vec2<i32>((uv + closest_offset) * texture_size), 0).rg, closest_depth, farthest_depth);
}

@fragment
fn fragment(@location(0) uv: vec2<f32>) -> Output {
    let texture_size = vec2<f32>(textureDimensions(view_target));
    let texel_size = 1.0 / texture_size;
    let frag_coord = uv * texture_size;
    let ifrag_coord = vec2<i32>(frag_coord);
    var center_depth = 0.0;

    // Fetch the current sample
    var original_color = textureLoad(view_target, ifrag_coord, 0);
    
    var current_color = original_color.rgb;
#ifdef TONEMAP
    original_color = vec4(tonemap(original_color.rgb), original_color.a);
    current_color = original_color.rgb;
#endif

    let closest_motion_vector_depth = get_closest_motion_vector_depth(uv, texture_size);
    let closest_motion_vector = closest_motion_vector_depth.xy;

#ifndef RESET
    // How confident we are that the history is representative of the current frame
    var history_confidence = textureLoad(history, ifrag_coord, 0).a;
    var history_color = vec3(0.0);

    // Reproject to find the equivalent sample from the past
    let history_uv = uv - closest_motion_vector;
    let history_motion_vector_depth = textureLoad(motion_history, vec2<i32>(history_uv * texture_size), 0);
    let history_motion_vector = history_motion_vector_depth.xy;
    let history_depth = history_motion_vector_depth.z;
#ifdef SAMPLE2
    // Softens just slightly, but much less than bicubic_b
    var filtered_color = textureSampleLevel(view_target, view_linear_sampler, uv + vec2(-0.15, -0.15) * texel_size, 0.0).rgb * 0.25;
    filtered_color    += textureSampleLevel(view_target, view_linear_sampler, uv + vec2( 0.15, -0.15) * texel_size, 0.0).rgb * 0.25;
    filtered_color    += textureSampleLevel(view_target, view_linear_sampler, uv + vec2( 0.15,  0.15) * texel_size, 0.0).rgb * 0.25;
    filtered_color    += textureSampleLevel(view_target, view_linear_sampler, uv + vec2(-0.15,  0.15) * texel_size, 0.0).rgb * 0.25;
#else
    var filtered_color = texture_sample_bicubic_b(view_target, view_linear_sampler, uv, texture_size).rgb;
#endif

#ifdef TONEMAP
        filtered_color = tonemap(filtered_color);
#endif

    var reprojection_fail = false;
    // Fall back to bicubic if the reprojected uv is off screen 
    if reprojection_fail ||
        any(history_uv <= 0.0) ||
        any(history_uv >= 1.0) {
        current_color = filtered_color;
        history_confidence = 1.0;
        reprojection_fail = true;
    } else {
        history_color = texture_sample_bicubic_catmull_rom(history, history_linear_sampler, history_uv, texture_size).rgb;
        // Constrain past sample with 3x3 YCoCg variance clipping (reduces ghosting)
        // YCoCg: https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING, slide 33
        // Variance clipping: https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf

#ifdef VARIANCE_CLIPPING
        var moment_1 = vec3(0.0);
        var moment_2 = vec3(0.0);
        for (var x = -1.0; x <= 1.0; x += 1.0) {
            for (var y = -1.0; y <= 1.0; y += 1.0) {
                let sample_uv = uv + (vec2(x, y) * texel_size);
                let sample = sample_view_target(sample_uv, texture_size);
                moment_1 += sample;
                moment_2 += sample * sample;
            }
        }
        let mean = moment_1 / 9.0;
        let variance = (moment_2 / 9.0) - (mean * mean);
        let std_deviation = sqrt(max(variance, vec3(0.0)));
        history_color = RGB_to_YCoCg(history_color);
        history_color = clip_towards_aabb_center(history_color, RGB_to_YCoCg(current_color), mean - std_deviation, mean + std_deviation);
        history_color = YCoCg_to_RGB(history_color);
#else ifdef CLAMPING
        var min_color = vec3(99999.0);
        var max_color = vec3(-99999.0);
        for (var x = -1.0; x <= 1.0; x += 1.0) {
            for (var y = -1.0; y <= 1.0; y += 1.0) {
                let sample_uv = uv + (vec2(x, y) * texel_size);
                let sample = sample_view_target(sample_uv, texture_size);
                min_color = min(sample, min_color);
                max_color = max(sample, max_color);
            }
        }
        history_color = YCoCg_to_RGB(clamp(RGB_to_YCoCg(history_color), min_color, max_color));
#endif

        let pixel_motion_vector = abs(closest_motion_vector) * texture_size;
        if pixel_motion_vector.x < 0.01 && pixel_motion_vector.y < 0.01 {
            // Increment when pixels are not moving
            history_confidence += 10.0;
        } else {
            // Else reset
            history_confidence = 1.0;
        }
    }

    // Blend current and past sample
    // Use more of the history if we're confident in it (reduces noise when there is no motion)
    // https://hhoppe.com/supersample.pdf, section 4.1
    var current_color_factor = clamp(1.0 / history_confidence, taa.min_history_blend_rate, taa.default_history_blend_rate);
    current_color_factor = select(current_color_factor, 1.0, reprojection_fail);
    current_color = mix(history_color, current_color, current_color_factor);

#ifdef VELOCITY_REJECTION
    // See Velocity Rejection: https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/
    let motion_distance = distance(history_motion_vector, closest_motion_vector);
    let motion_disocclusion = saturate((motion_distance - 0.001) * taa.velocity_rejection);
    current_color = mix(current_color, filtered_color, motion_disocclusion);
#endif //VELOCITY_REJECTION

#ifdef DEPTH_REJECTION
#ifndef WEBGL2
    let closest_depth = closest_motion_vector_depth.z;
    let farthest_depth = closest_motion_vector_depth.w;
    center_depth = textureLoad(depth, vec2<i32>(uv * texture_size), 0);

    let closest_pixel_radius = world_space_pixel_radius(closest_depth);
    let farthest_pixel_radius = world_space_pixel_radius(farthest_depth);
    let avg_pixel_radius = ((closest_pixel_radius + farthest_pixel_radius) * 0.5);

    let history_closest_world_position = position_history_ndc_to_world(vec3(uv_to_ndc(history_uv), history_depth));
    let farthest_world_position = position_ndc_to_world(vec3(uv_to_ndc(uv), farthest_depth));
    let closest_world_position = position_ndc_to_world(vec3(uv_to_ndc(uv), closest_depth));

    let closest_dist = distance(view.world_position.xyz, closest_world_position);
    let farthest_dist = distance(view.world_position.xyz, farthest_world_position);
    let history_dist = distance(view.world_position.xyz, history_closest_world_position);

    let clamped_dist = clamp(history_dist, 
                             closest_dist - closest_pixel_radius * taa.depth_rejection_px_radius, 
                             farthest_dist + farthest_pixel_radius * taa.depth_rejection_px_radius);
    let px_dist = distance(clamped_dist, history_dist) / avg_pixel_radius;
    let factor = saturate((px_dist) * 0.1);
    current_color = mix(current_color, filtered_color, factor);
#endif //#ifndef WEBGL2
#endif //DEPTH_REJECTION

#endif // #ifndef RESET


    // Write output to history and view target
    var out: Output;

#ifdef RESET
    let history_confidence = 1.0 / taa.min_history_blend_rate;
#endif // RESET

#ifdef SAMPLE2
    out.history = vec4(original_color.rgb, history_confidence);
#else
    out.history = vec4(current_color, history_confidence);
#endif // SAMPLE2

#ifdef TONEMAP
    current_color = reverse_tonemap(current_color);
#endif // TONEMAP

    out.view_target = vec4(current_color, original_color.a);
    out.motion_history = vec4(textureLoad(motion_vectors, vec2<i32>(uv * texture_size), 0).xy, center_depth, 0.0);
    return out;
}
