use bevy::app::{App, Plugin};
use bevy::asset::{load_internal_asset, Handle};
use bevy::core::FrameCount;
use bevy::core_pipeline::core_3d::{self, CORE_3D};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, ViewPrepassTextures};
use bevy::ecs::{
    prelude::{Bundle, Component, Entity},
    query::{QueryItem, With},
    schedule::IntoSystemConfigs,
    system::{Commands, Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy::math::vec2;
use bevy::prelude::{default, Camera3d, Vec2};
use bevy::reflect::Reflect;
use bevy::render::extract_component::{
    ComponentUniforms, DynamicUniformIndex, UniformComponentPlugin,
};
use bevy::render::render_resource::{BufferBindingType, ShaderType};
use bevy::render::{
    camera::{ExtractedCamera, MipBias, TemporalJitter},
    prelude::{Camera, Projection},
    render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner},
    render_resource::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingResource, BindingType, CachedRenderPipelineId,
        ColorTargetState, ColorWrites, Extent3d, FilterMode, FragmentState, MultisampleState,
        Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
        RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, Shader,
        ShaderStages, SpecializedRenderPipeline, SpecializedRenderPipelines, TextureDescriptor,
        TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDimension,
    },
    renderer::{RenderContext, RenderDevice},
    texture::{BevyDefault, CachedTexture, TextureCache},
    view::{ExtractedView, Msaa, ViewTarget},
    ExtractSchedule, MainWorld, Render, RenderApp, RenderSet,
};

use crate::fxaa::FxaaNode;

pub mod fxaa;

mod draw_3d_graph {
    pub mod node {
        /// Label for the TAA render node.
        pub const TAA: &str = "taa";
    }
}

const TAA_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(9832793854650921);

/// Plugin for temporal anti-aliasing. Disables multisample anti-aliasing (MSAA).
///
/// See [`TAASettings`] for more details.
pub struct TAAPlugin;

impl Plugin for TAAPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, TAA_SHADER_HANDLE, "taa.wgsl", Shader::from_wgsl);

        app.insert_resource(Msaa::Off)
            .register_type::<TAASettings>()
            .add_plugins(UniformComponentPlugin::<TAAParameters>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<TAAPipeline>>()
            .add_systems(ExtractSchedule, extract_taa_settings)
            .add_systems(
                Render,
                (
                    prepare_taa_jitter_and_mip_bias.in_set(RenderSet::ManageViews),
                    prepare_taa_pipelines.in_set(RenderSet::Prepare),
                    prepare_taa_history_textures.in_set(RenderSet::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<TAANode>>(CORE_3D, draw_3d_graph::node::TAA)
            .add_render_graph_node::<ViewNodeRunner<FxaaNode>>(CORE_3D, fxaa::FXAA_PREPASS)
            .add_render_graph_edges(
                CORE_3D,
                &[
                    core_3d::graph::node::END_MAIN_PASS,
                    fxaa::FXAA_PREPASS,
                    draw_3d_graph::node::TAA,
                    core_3d::graph::node::BLOOM,
                    core_3d::graph::node::TONEMAPPING,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<TAAPipeline>();
    }
}

/// Bundle to apply temporal anti-aliasing.
#[derive(Bundle, Default)]
pub struct TAABundle {
    pub settings: TAASettings,
    pub jitter: TemporalJitter,
    pub depth_prepass: DepthPrepass,
    pub motion_vector_prepass: MotionVectorPrepass,
}

impl TAABundle {
    pub fn sample2() -> TAABundle {
        TAABundle {
            settings: TAASettings {
                sequence: TAASequence::Sample2,
                parameters: TAAParameters {
                    default_history_blend_rate: 0.5,
                    min_history_blend_rate: 0.5,
                    ..default()
                },
                ..default()
            },
            ..default()
        }
    }
    pub fn sample4() -> TAABundle {
        TAABundle {
            settings: TAASettings {
                sequence: TAASequence::Sample4,
                parameters: TAAParameters {
                    default_history_blend_rate: 0.33,
                    min_history_blend_rate: 0.25,
                    ..default()
                },
                ..default()
            },
            ..default()
        }
    }
    pub fn sample8() -> TAABundle {
        TAABundle {
            settings: TAASettings {
                sequence: TAASequence::Sample8,
                parameters: TAAParameters {
                    default_history_blend_rate: 0.25,
                    min_history_blend_rate: 0.125,
                    ..default()
                },
                ..default()
            },
            ..default()
        }
    }
}

#[derive(Reflect, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TAASequence {
    Sample2,
    Sample4,
    #[default]
    Sample8,
}

impl TAASequence {
    fn get_offset(&self, frame_count: u32) -> Vec2 {
        match self {
            // https://advances.realtimerendering.com/s2017/DecimaSiggraph2017.pdf
            TAASequence::Sample2 => [vec2(0.0, -0.5), vec2(-0.5, 0.0)][frame_count as usize % 2],
            // RGGS https://blog.demofox.org/2015/04/23/4-rook-antialiasing-rgss/
            TAASequence::Sample4 => [
                vec2(1.0 / 8.0, 3.0 / 8.0),
                vec2(-3.0 / 8.0, 1.0 / 8.0),
                vec2(3.0 / 8.0, -1.0 / 8.0),
                vec2(-1.0 / 8.0, -3.0 / 8.0),
            ][frame_count as usize % 4],
            // Halton sequence (2, 3) - 0.5, skipping i = 0
            // https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/1680d1edd5c034f88ebbbb793d8b88f8842cf804/src/ffx-fsr2-api/ffx_fsr2.cpp#L1194
            TAASequence::Sample8 => [
                vec2(0.0, -0.16666666),
                vec2(-0.25, 0.16666669),
                vec2(0.25, -0.3888889),
                vec2(-0.375, -0.055555552),
                vec2(0.125, 0.2777778),
                vec2(-0.125, -0.2777778),
                vec2(0.375, 0.055555582),
                vec2(-0.4375, 0.3888889),
            ][frame_count as usize % 8],
        }
    }
}

/// Component to apply temporal anti-aliasing to a 3D perspective camera.
///
/// Temporal anti-aliasing (TAA) is a form of image filtering, like
/// multisample anti-aliasing (MSAA), or fast approximate anti-aliasing (FXAA).
/// TAA works by blending (averaging) each frame with the past few frames.
///
/// If no [`MipBias`] component is attached to the camera, TAA will add a MipBias(-1.0) component.
#[derive(Component, Reflect, Clone)]
pub struct TAASettings {
    sequence: TAASequence,
    parameters: TAAParameters,

    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false after one frame.
    pub reset: bool,
}

#[derive(Component, ShaderType, Reflect, Default, Copy, Clone)]
pub struct TAAParameters {
    default_history_blend_rate: f32,
    min_history_blend_rate: f32,
    future1: f32,
    future2: f32,
}

impl Default for TAASettings {
    fn default() -> Self {
        Self {
            sequence: TAASequence::Sample8,
            reset: true,
            parameters: TAAParameters::default(),
        }
    }
}

#[derive(Default)]
struct TAANode;

impl ViewNode for TAANode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static TAAHistoryTextures,
        &'static ViewPrepassTextures,
        &'static TAAPipelineId,
        &'static DynamicUniformIndex<TAAParameters>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            camera,
            view_target,
            taa_history_textures,
            prepass_textures,
            taa_pipeline_id,
            uniform_index,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let (Some(pipelines), Some(pipeline_cache)) = (
            world.get_resource::<TAAPipeline>(),
            world.get_resource::<PipelineCache>(),
        ) else {
            return Ok(());
        };
        let (Some(taa_pipeline), Some(prepass_motion_vectors_texture), Some(prepass_depth_texture)) = (
            pipeline_cache.get_render_pipeline(taa_pipeline_id.0),
            &prepass_textures.motion_vectors,
            &prepass_textures.depth,
        ) else {
            return Ok(());
        };
        let view_target = view_target.post_process_write();
        let uniforms = world.resource::<ComponentUniforms<TAAParameters>>();

        let Some(uniforms) = uniforms.binding() else {
            return Ok(());
        };

        let taa_bind_group =
            render_context
                .render_device()
                .create_bind_group(&BindGroupDescriptor {
                    label: Some("taa_bind_group"),
                    layout: &pipelines.taa_bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(view_target.source),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Sampler(&pipelines.linear_samplers[0]),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::TextureView(
                                &taa_history_textures.read.default_view,
                            ),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::Sampler(&pipelines.linear_samplers[1]),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: BindingResource::TextureView(
                                &taa_history_textures.motion_read.default_view,
                            ),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: BindingResource::TextureView(
                                &prepass_motion_vectors_texture.default_view,
                            ),
                        },
                        BindGroupEntry {
                            binding: 6,
                            resource: BindingResource::TextureView(
                                &prepass_depth_texture.default_view,
                            ),
                        },
                        BindGroupEntry {
                            binding: 7,
                            resource: uniforms,
                        },
                    ],
                });

        {
            let mut taa_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("taa_pass"),
                color_attachments: &[
                    Some(RenderPassColorAttachment {
                        view: view_target.destination,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                    Some(RenderPassColorAttachment {
                        view: &taa_history_textures.write.default_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                    Some(RenderPassColorAttachment {
                        view: &taa_history_textures.motion_write.default_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                ],
                depth_stencil_attachment: None,
            });
            taa_pass.set_render_pipeline(taa_pipeline);
            taa_pass.set_bind_group(0, &taa_bind_group, &[uniform_index.index()]);
            if let Some(viewport) = camera.viewport.as_ref() {
                taa_pass.set_camera_viewport(viewport);
            }
            taa_pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}

#[derive(Resource)]
struct TAAPipeline {
    taa_bind_group_layout: BindGroupLayout,
    linear_samplers: [Sampler; 2],
}

impl FromWorld for TAAPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let linear_discriptor = SamplerDescriptor {
            label: Some("taa_linear_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..SamplerDescriptor::default()
        };

        let linear_samplers = [
            render_device.create_sampler(&linear_discriptor),
            render_device.create_sampler(&linear_discriptor),
        ];

        let taa_bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("taa_bind_group_layout"),
                entries: &[
                    // View target (read)
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // View target Linear sampler
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    // TAA History (read)
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // TAA History Linear sampler
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    // TAA Motion History (read)
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Motion Vectors
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Depth
                    BindGroupLayoutEntry {
                        binding: 6,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Depth,
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // TAA Parameters
                    BindGroupLayoutEntry {
                        binding: 7,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: Some(TAAParameters::min_size()),
                        },
                        visibility: ShaderStages::FRAGMENT,
                        count: None,
                    },
                ],
            });

        TAAPipeline {
            taa_bind_group_layout,
            linear_samplers,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct TAAPipelineKey {
    hdr: bool,
    reset: bool,
    sequence: TAASequence,
}

impl SpecializedRenderPipeline for TAAPipeline {
    type Key = TAAPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = vec![];

        let format = if key.hdr {
            shader_defs.push("TONEMAP".into());
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        if key.reset {
            shader_defs.push("RESET".into());
        }

        // TODO webgl is not a bevy_mod_taa feature.
        // #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
        #[cfg(target_arch = "wasm32")]
        shader_defs.push("WEBGL2".into());

        match key.sequence {
            TAASequence::Sample2 => shader_defs.push("SAMPLE2".into()),
            TAASequence::Sample4 => shader_defs.push("SAMPLE4".into()),
            TAASequence::Sample8 => shader_defs.push("SAMPLE8".into()),
        }

        RenderPipelineDescriptor {
            label: Some("taa_pipeline".into()),
            layout: vec![self.taa_bind_group_layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: TAA_SHADER_HANDLE,
                shader_defs,
                entry_point: "taa".into(),
                targets: vec![
                    Some(ColorTargetState {
                        format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: Vec::new(),
        }
    }
}

fn extract_taa_settings(mut commands: Commands, mut main_world: ResMut<MainWorld>) {
    let mut cameras_3d = main_world
        .query_filtered::<(Entity, &Camera, &Projection, &mut TAASettings), (
            With<Camera3d>,
            With<TemporalJitter>,
            With<DepthPrepass>,
            With<MotionVectorPrepass>,
        )>();

    for (entity, camera, camera_projection, mut taa_settings) in
        cameras_3d.iter_mut(&mut main_world)
    {
        let has_perspective_projection = matches!(camera_projection, Projection::Perspective(_));
        if camera.is_active && has_perspective_projection {
            commands
                .get_or_spawn(entity)
                .insert(taa_settings.clone())
                .insert(taa_settings.parameters);
            taa_settings.reset = false;
        }
    }
}

fn prepare_taa_jitter_and_mip_bias(
    frame_count: Res<FrameCount>,
    mut query: Query<(Entity, &mut TemporalJitter, Option<&MipBias>, &TAASettings)>,
    mut commands: Commands,
) {
    for (entity, mut jitter, mip_bias, taa_settings) in &mut query {
        jitter.offset = taa_settings.sequence.get_offset(frame_count.0);

        if mip_bias.is_none() {
            commands.entity(entity).insert(MipBias(-1.0));
        }
    }
}

#[derive(Component)]
struct TAAHistoryTextures {
    write: CachedTexture,
    read: CachedTexture,
    motion_write: CachedTexture,
    motion_read: CachedTexture,
}

fn prepare_taa_history_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    frame_count: Res<FrameCount>,
    views: Query<(Entity, &ExtractedCamera, &ExtractedView), With<TAASettings>>,
) {
    for (entity, camera, view) in &views {
        if let Some(physical_viewport_size) = camera.physical_viewport_size {
            let mut texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: if view.hdr {
                    ViewTarget::TEXTURE_FORMAT_HDR
                } else {
                    TextureFormat::bevy_default()
                },
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            };

            let mut motion_texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            };

            texture_descriptor.label = Some("taa_history_1_texture");
            let history_1_texture = texture_cache.get(&render_device, texture_descriptor.clone());

            texture_descriptor.label = Some("taa_history_2_texture");
            let history_2_texture = texture_cache.get(&render_device, texture_descriptor);

            motion_texture_descriptor.label = Some("taa_history_1_texture");
            let history_motion_1_texture =
                texture_cache.get(&render_device, motion_texture_descriptor.clone());

            motion_texture_descriptor.label = Some("taa_history_2_texture");
            let history_motion_2_texture =
                texture_cache.get(&render_device, motion_texture_descriptor);

            let textures = if frame_count.0 % 2 == 0 {
                TAAHistoryTextures {
                    write: history_1_texture,
                    read: history_2_texture,
                    motion_write: history_motion_1_texture,
                    motion_read: history_motion_2_texture,
                }
            } else {
                TAAHistoryTextures {
                    write: history_2_texture,
                    read: history_1_texture,
                    motion_write: history_motion_2_texture,
                    motion_read: history_motion_1_texture,
                }
            };

            commands.entity(entity).insert(textures);
        }
    }
}

#[derive(Component)]
struct TAAPipelineId(CachedRenderPipelineId);

fn prepare_taa_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TAAPipeline>>,
    pipeline: Res<TAAPipeline>,
    views: Query<(Entity, &ExtractedView, &TAASettings)>,
) {
    for (entity, view, taa_settings) in &views {
        let mut pipeline_key = TAAPipelineKey {
            hdr: view.hdr,
            reset: taa_settings.reset,
            sequence: taa_settings.sequence,
        };
        let pipeline_id = pipelines.specialize(&pipeline_cache, &pipeline, pipeline_key.clone());

        // Prepare non-reset pipeline anyways - it will be necessary next frame
        if pipeline_key.reset {
            pipeline_key.reset = false;
            pipelines.specialize(&pipeline_cache, &pipeline, pipeline_key);
        }

        commands.entity(entity).insert(TAAPipelineId(pipeline_id));
    }
}
