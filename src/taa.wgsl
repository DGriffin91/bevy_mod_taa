// References:
// https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail
// http://behindthepixels.io/assets/files/TemporalAA.pdf
// http://leiy.cc/publications/TAA/TAA_EG2020_Talk.pdf
// https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING

#import bevy_pbr::utils PI
const PI_SQ: f32 = 9.8696044010893586188344910;

// Controls how much to blend between the current and past samples
// Lower numbers = less of the current sample and more of the past sample = more smoothing
struct TAAParameters {
    default_history_blend_rate: f32, // Default blend rate to use when no confidence in history
    min_history_blend_rate: f32, // Minimum blend rate allowed, to ensure at least some of the current sample is used
    future1: f32,
    future2: f32,
};

#import bevy_core_pipeline::fullscreen_vertex_shader  FullscreenVertexOutput

@group(0) @binding(0) var view_target: texture_2d<f32>;
@group(0) @binding(1) var view_linear_sampler: sampler;
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var history_linear_sampler: sampler;
@group(0) @binding(4) var history2: texture_2d<f32>;
@group(0) @binding(5) var history2_linear_sampler: sampler;
@group(0) @binding(6) var motion_history: texture_2d<f32>;
@group(0) @binding(7) var motion_vectors: texture_2d<f32>;
@group(0) @binding(8) var depth: texture_depth_2d;
@group(0) @binding(9) var<uniform> prams: TAAParameters;

struct Output {
    @location(0) view_target: vec4<f32>,
    @location(1) history: vec4<f32>,
    @location(2) motion_history: vec4<f32>,
};

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
    
    let sample0 = textureSample(tex, tex_sampler, offset.xz);
    let sample1 = textureSample(tex, tex_sampler, offset.yz);
    let sample2 = textureSample(tex, tex_sampler, offset.xw);
    let sample3 = textureSample(tex, tex_sampler, offset.yw);

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
fn get_closest_motion_vector(uv: vec2<f32>, texture_size: vec2<f32>) -> vec2<f32> {
    var closest_depth = 0.0;
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
        }
    }
#endif //WEBGL2
    return textureLoad(motion_vectors, vec2<i32>((uv + closest_offset) * texture_size), 0).rg;
}

@fragment
fn taa(@location(0) uv: vec2<f32>) -> Output {
    let texture_size = vec2<f32>(textureDimensions(view_target));
    let texel_size = 1.0 / texture_size;

    // Fetch the current sample
    var original_color = textureLoad(view_target, vec2<i32>(uv * texture_size), 0);
    var current_color = original_color.rgb;
#ifdef TONEMAP
    current_color = tonemap(current_color);
    original_color = vec4(tonemap(original_color.rgb), original_color.a);
#endif

#ifndef RESET
    let closest_motion_vector = get_closest_motion_vector(uv, texture_size);

    // How confident we are that the history is representative of the current frame
    var history_confidence = textureLoad(history, vec2<i32>(uv * texture_size), 0).a;
    var history_color = vec3(0.0);
#ifdef SAMPLE3
    var history2_color = vec3(0.0);
#endif

    // Reproject to find the equivalent sample from the past
    let history_uv = uv - closest_motion_vector;
    let previous_motion_vector = textureLoad(motion_history, vec2<i32>(history_uv * texture_size), 0).rg;
    let history2_uv = uv - closest_motion_vector - previous_motion_vector;
    var center_color_bicubic = texture_sample_bicubic_b(view_target, view_linear_sampler, uv, texture_size).rgb;

#ifdef TONEMAP
        center_color_bicubic = tonemap(center_color_bicubic);
#endif

    var reprojection_fail = false;
    // Fall back to bicubic if the reprojected uv is off screen 
    if reprojection_fail ||
        all(history_uv <= 0.0) ||
        all(history_uv >= 1.0) {
        current_color = center_color_bicubic;
        history_confidence = 1.0;
        reprojection_fail = true;
    } else {
        history_color = texture_sample_bicubic_catmull_rom(history, history_linear_sampler, history_uv, texture_size).rgb;
#ifdef SAMPLE3
        history2_color = texture_sample_bicubic_catmull_rom(history2, history2_linear_sampler, history2_uv, texture_size).rgb;
#endif

        // Constrain past sample with 3x3 YCoCg variance clipping (reduces ghosting)
        // YCoCg: https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING, slide 33
        // Variance clipping: https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf
        let s_tl = sample_view_target(uv + vec2(-texel_size.x,  texel_size.y), texture_size);
        let s_tm = sample_view_target(uv + vec2( 0.0,           texel_size.y), texture_size);
        let s_tr = sample_view_target(uv + vec2( texel_size.x,  texel_size.y), texture_size);
        let s_ml = sample_view_target(uv + vec2(-texel_size.x,  0.0), texture_size);
        let s_mm = RGB_to_YCoCg(current_color);
        let s_mr = sample_view_target(uv + vec2( texel_size.x,  0.0), texture_size);
        let s_bl = sample_view_target(uv + vec2(-texel_size.x, -texel_size.y), texture_size);
        let s_bm = sample_view_target(uv + vec2( 0.0,          -texel_size.y), texture_size);
        let s_br = sample_view_target(uv + vec2( texel_size.x, -texel_size.y), texture_size);
        let moment_1 = s_tl + s_tm + s_tr + s_ml + s_mm + s_mr + s_bl + s_bm + s_br;
        let moment_2 = (s_tl * s_tl) + (s_tm * s_tm) + (s_tr * s_tr) + (s_ml * s_ml) + (s_mm * s_mm) + (s_mr * s_mr) + (s_bl * s_bl) + (s_bm * s_bm) + (s_br * s_br);
        let mean = moment_1 / 9.0;
        let variance = (moment_2 / 9.0) - (mean * mean);
        let std_deviation = sqrt(max(variance, vec3(0.0)));
        history_color = RGB_to_YCoCg(history_color);
        history_color = clip_towards_aabb_center(history_color, s_mm, mean - std_deviation, mean + std_deviation);
        history_color = YCoCg_to_RGB(history_color);
#ifdef SAMPLE3
        history2_color = RGB_to_YCoCg(history2_color);
        history2_color = clip_towards_aabb_center(history2_color, s_mm, mean - std_deviation, mean + std_deviation);
        history2_color = YCoCg_to_RGB(history2_color);
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
    var current_color_factor = clamp(1.0 / history_confidence, prams.min_history_blend_rate, prams.default_history_blend_rate);
    current_color_factor = select(current_color_factor, 1.0, reprojection_fail);
#ifdef SAMPLE3
    current_color = select((current_color + history_color + history2_color) / 3.0, current_color, reprojection_fail);
#else
    current_color = mix(history_color, current_color, current_color_factor);
#endif

    // See Velocity Rejection: https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/
    let motion_distance = distance(previous_motion_vector, closest_motion_vector);
    let motion_disocclusion = saturate((motion_distance - 0.001) * 120.0); //The 120.0 was just hand tuned, needs further testing.
    current_color = mix(current_color, center_color_bicubic, motion_disocclusion);
#endif // #ifndef RESET


    // Write output to history and view target
    var out: Output;

#ifdef RESET
    let history_confidence = 1.0 / prams.min_history_blend_rate;
#endif // RESET

#ifdef SAMPLE2_OR_SAMPLE3
    out.history = vec4(original_color.rgb, history_confidence);
#else
    out.history = vec4(current_color, history_confidence);
#endif // SAMPLE2_OR_SAMPLE3

#ifdef TONEMAP
    current_color = reverse_tonemap(current_color);
#endif // TONEMAP

    out.view_target = vec4(current_color, original_color.a);
    out.motion_history = textureLoad(motion_vectors, vec2<i32>(uv * texture_size), 0);
    return out;
}
