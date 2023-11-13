// References:
// https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail

#import bevy_pbr::mesh_view_bindings::{view, globals}
#import bevy_pbr::utils::PI
#import bevy_pbr::mesh_view_bindings as vb

const PI_SQ: f32 = 9.8696044010893586188344910;

// Controls how much to blend between the current and past samples
// Lower numbers = less of the current sample and more of the past sample = more smoothing
struct Uniform {
    inverse_view_proj: mat4x4<f32>, // not jittered
    prev_inverse_view_proj: mat4x4<f32>, // not jittered
    velocity_rejection: f32,
    depth_rejection_px_radius: f32,
    normals_rejection_scale: f32,
};

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(21) var history_linear_sampler: sampler;
@group(0) @binding(22) var history_normals_linear_sampler: sampler;
@group(0) @binding(23) var data_history: texture_2d<f32>;
@group(0) @binding(24) var data_history_normals: texture_2d<f32>;
@group(0) @binding(25) var motion_vectors: texture_2d<f32>;
@group(0) @binding(26) var<uniform> uni: Uniform;
@group(0) @binding(27) var depth: texture_depth_2d;
@group(0) @binding(28) var normals: texture_2d<f32>;

struct Output {
    @location(0) disocclusion_output: vec4<f32>,
    @location(1) data_history: vec4<f32>,
    @location(2) data_history_normals: vec4<f32>,
};

// https://github.com/NVIDIAGameWorks/RayTracingDenoiser/blob/3c881ae3075f7ca754e22177877335b82e16da5a/Shaders/Include/Common.hlsli#L124
fn world_space_pixel_radius(ndc_view_z: f32) -> f32 {
    let perspective_near = vb::view.projection[3][2];
    let linear_depth = perspective_near / ndc_view_z;
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
    let world_pos = uni.inverse_view_proj * vec4(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Convert a ndc space position to world space
fn position_history_ndc_to_world(ndc_pos: vec3<f32>) -> vec3<f32> {
    let world_pos = uni.prev_inverse_view_proj * vec4(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Convert a world space position to ndc space
fn position_world_to_ndc(world_pos: vec3<f32>) -> vec3<f32> {
    let ndc_pos = vb::view.unjittered_view_proj * vec4(world_pos, 1.0);
    return ndc_pos.xyz / ndc_pos.w;
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
    let texture_size = view.viewport.zw;
    let texel_size = 1.0 / texture_size;
    let frag_coord = uv * texture_size;
    let ifrag_coord = vec2<i32>(frag_coord);
    var center_depth = 0.0;

    let closest_motion_vector_depth = get_closest_motion_vector_depth(uv, texture_size);
    let closest_motion_vector = closest_motion_vector_depth.xy;
    let closest_depth = closest_motion_vector_depth.z;
    let farthest_depth = closest_motion_vector_depth.w;

    // Reproject to find the equivalent sample from the past
    let history_uv = uv - closest_motion_vector;
    let history_motion_vector_depth = textureLoad(data_history, vec2<i32>(history_uv * texture_size), 0);
    let history_motion_vector = history_motion_vector_depth.xy;
    let history_depth = history_motion_vector_depth.z;
    let history_closest_depth = history_motion_vector_depth.w;

    let normals = textureLoad(normals, vec2<i32>(uv * texture_size), 0) * 2.0 - 1.0;

    var velocity_disocclusion = 1.0;
    var depth_disocclusion = 1.0;
    var normals_disocclusion = 1.0;

// VELOCITY_REJECTION
    var cam_movment = vec2(0.0);
#ifdef VELOCITY_REJECTION
#ifdef WEBGL2
    // Need to tune to match
    let motion_distance = distance(history_motion_vector, closest_motion_vector);
#else
    // See Velocity Rejection: https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/
    center_depth = textureLoad(depth, vec2<i32>(uv * texture_size), 0);

    // Camera movment motion vector
    let ndc1 = vec3(uv_to_ndc(uv), closest_depth);
    let ndc2 = vec3(uv_to_ndc(uv), history_closest_depth);
    let a = position_world_to_ndc(position_ndc_to_world(ndc1)).xy;
    let b = position_world_to_ndc(position_history_ndc_to_world(ndc2)).xy;
    cam_movment = (b - a) * vec2(0.5, -0.5);

    // Camera rotate
    let closest_b = position_world_to_ndc(position_history_ndc_to_world(ndc1)).xy;
    let cam_rotate = (closest_b - a) * vec2(0.5, -0.5);

    // Cancel out camera movment when checking motion vector distance
    let motion_distance_vs_hist = distance(history_motion_vector, closest_motion_vector - cam_movment);
    var motion_distance_vs_cam = distance(closest_motion_vector, cam_movment);
    let cam_rotate_vel = saturate(1.0 - length(cam_rotate) * 100.0);
    let motion_distance = motion_distance_vs_hist * motion_distance_vs_cam * 140.0;
#endif //WEBGL2
    velocity_disocclusion *= saturate((motion_distance - 0.001) * uni.velocity_rejection);
#endif //VELOCITY_REJECTION

#ifdef DEPTH_REJECTION
#ifndef WEBGL2
    center_depth = textureLoad(depth, vec2<i32>(uv * texture_size), 0);

    let farthest_pixel_radius = world_space_pixel_radius(farthest_depth);

    let history_world_position = position_history_ndc_to_world(vec3(uv_to_ndc(history_uv), history_depth));
    let farthest_world_position = position_ndc_to_world(vec3(uv_to_ndc(uv), farthest_depth));
    let closest_world_position = position_ndc_to_world(vec3(uv_to_ndc(uv), closest_depth));


    let pixel_radius_scaled = farthest_pixel_radius * uni.depth_rejection_px_radius;

    let aabb_min = min(farthest_world_position, closest_world_position) - pixel_radius_scaled;
    let aabb_max = max(farthest_world_position, closest_world_position) + pixel_radius_scaled;

    let clamped = clamp(history_world_position, aabb_min, aabb_max);
    let factor = saturate(distance(history_world_position, clamped) * pixel_radius_scaled) * f32((farthest_depth * history_depth) != 0.0);

    depth_disocclusion *= factor;
#endif //#ifndef WEBGL2
#endif //DEPTH_REJECTION

#ifdef NORMALS_REJECTION
    var min_nor_diff = 1.0;
    for(var y = -1; y <= 1; y += 1) {
        for(var x = -1; x <= 1; x += 1) {
            let history_normals = textureLoad(data_history_normals, vec2<i32>(history_uv * texture_size) + vec2(x, y), 0) * 2.0 - 1.0;
            min_nor_diff = min(min_nor_diff, 1.0 - saturate(dot(history_normals.xyz, normals.xyz) * 0.5 + 0.5));
        }
    }
    normals_disocclusion = saturate(saturate(min_nor_diff * min_nor_diff - 0.25) * uni.normals_rejection_scale);
#endif //NORMALS_REJECTION

    // Write output to history and view target
    var out: Output;

    out.disocclusion_output = vec4(velocity_disocclusion, depth_disocclusion, normals_disocclusion, 0.0);
    out.data_history = vec4(closest_motion_vector - cam_movment, center_depth, closest_depth);
    out.data_history_normals = normals * 0.5 + 0.5;

    return out;
}
