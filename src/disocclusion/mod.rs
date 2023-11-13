use bevy::app::{App, Plugin};
use bevy::asset::{load_internal_asset, Handle};
use bevy::core::FrameCount;
use bevy::core_pipeline::core_3d::{self, CORE_3D};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::core_pipeline::prepass::{
    DepthPrepass, MotionVectorPrepass, ViewPrepassTextures, NORMAL_PREPASS_FORMAT,
};
use bevy::ecs::{
    prelude::Entity,
    query::{QueryItem, With},
    schedule::IntoSystemConfigs,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy::math::vec4;
use bevy::prelude::*;
use bevy::reflect::Reflect;
use bevy::render::extract_component::{
    ComponentUniforms, DynamicUniformIndex, UniformComponentPlugin,
};
use bevy::render::globals::{GlobalsBuffer, GlobalsUniform};
use bevy::render::render_resource::{BindGroupEntries, BufferBindingType, ShaderType};
use bevy::render::view::{ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{
    camera::{ExtractedCamera, TemporalJitter},
    prelude::{Camera, Projection},
    render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner},
    render_resource::{
        BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
        CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d, FilterMode, FragmentState,
        MultisampleState, Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment,
        RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
        SamplerDescriptor, Shader, ShaderStages, SpecializedRenderPipeline,
        SpecializedRenderPipelines, TextureDescriptor, TextureDimension, TextureFormat,
        TextureSampleType, TextureUsages, TextureViewDimension,
    },
    renderer::{RenderContext, RenderDevice},
    texture::{CachedTexture, TextureCache},
    view::ExtractedView,
    ExtractSchedule, MainWorld, Render, RenderApp, RenderSet,
};

mod draw_3d_graph {
    pub mod node {
        /// Label for the DISOCCLUSION render node.
        pub const DISOCCLUSION: &str = "disocclusion detection";
    }
}

const DISOCCLUSION_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(82390847572039487);

/// Generates a texture showing regions of disocclusion from prepass inputs
/// x: velocity_disocclusion
/// y: depth_disocclusion
/// z: normals_disocclusion
/// Each method generates some false positives and some false negatives.
/// One approach is to trust two of three:
/// `let two_of_three = min(min(max(d.x, d.y), max(d.y, d.z)), max(d.x, d.z));`
/// The range is from 0.0 to 1.0 where 1.0 is no disocclusion detected
pub struct DisocclusionPlugin;
impl Plugin for DisocclusionPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            DISOCCLUSION_SHADER_HANDLE,
            "disocclusion.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<DisocclusionSettings>()
            .add_plugins(UniformComponentPlugin::<DisocclusionUniforms>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<DisocclusionPipeline>>()
            .add_systems(ExtractSchedule, extract_disocclusion_settings)
            .add_systems(
                Render,
                (
                    prepare_disocclusion_pipelines.in_set(RenderSet::Prepare),
                    prepare_disocclusion_textures.in_set(RenderSet::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<DisocclusionNode>>(
                CORE_3D,
                draw_3d_graph::node::DISOCCLUSION,
            )
            .add_render_graph_edges(
                CORE_3D,
                &[
                    core_3d::graph::node::END_PREPASSES,
                    draw_3d_graph::node::DISOCCLUSION,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<DisocclusionPipeline>();
    }
}

#[derive(Component, ShaderType, Reflect, Default, Copy, Clone, Debug)]
pub struct DisocclusionUniforms {
    inverse_view_proj: Mat4,
    prev_inverse_view_proj: Mat4,
    velocity_disocclusion: f32,
    depth_disocclusion_px_radius: f32,
    normals_disocclusion_scale: f32,
}

#[derive(Component, Reflect, Clone, Debug)]
pub struct DisocclusionSettings {
    /// Higher values are more sensitive / more likely to detect disocclusion.
    pub velocity_disocclusion: Option<f32>,

    /// Reject history depths that are outside the neighborhood range by this px radius
    /// Lower values are more sensitive / more likely to detect disocclusion.
    pub depth_disocclusion_px_radius: Option<f32>,

    /// Higher values are more sensitive / more likely to detect disocclusion.
    pub normals_disocclusion_scale: Option<f32>,
}

impl Default for DisocclusionSettings {
    fn default() -> Self {
        Self {
            velocity_disocclusion: Some(40.0),
            depth_disocclusion_px_radius: Some(100.0),
            normals_disocclusion_scale: Some(1.0),
        }
    }
}

#[derive(Default)]
struct DisocclusionNode;

impl ViewNode for DisocclusionNode {
    type ViewQuery = (
        &'static ViewUniformOffset,
        &'static ExtractedCamera,
        &'static DisocclusionTextures,
        &'static ViewPrepassTextures,
        &'static DisocclusionPipelineId,
        &'static DynamicUniformIndex<DisocclusionUniforms>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            view_uniform_offset,
            camera,
            disocclusion_textures,
            prepass_textures,
            disocclusion_pipeline_id,
            uniform_index,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let (Some(pipelines), Some(pipeline_cache)) = (
            world.get_resource::<DisocclusionPipeline>(),
            world.get_resource::<PipelineCache>(),
        ) else {
            return Ok(());
        };

        let (
            Some(disocclusion_pipeline),
            Some(prepass_motion_vectors_texture),
            Some(prepass_depth_texture),
            Some(prepass_normal_texture),
        ) = (
            pipeline_cache.get_render_pipeline(disocclusion_pipeline_id.0),
            &prepass_textures.motion_vectors,
            &prepass_textures.depth,
            &prepass_textures.normal,
        )
        else {
            return Ok(());
        };

        let uniforms = world.resource::<ComponentUniforms<DisocclusionUniforms>>();

        let Some(uniforms) = uniforms.binding() else {
            return Ok(());
        };

        let view_uniforms = world.resource::<ViewUniforms>();
        let view_uniforms = view_uniforms.uniforms.binding().unwrap();
        let globals_buffer = world.resource::<GlobalsBuffer>();
        let globals_binding = globals_buffer.buffer.binding().unwrap();

        let disocclusion_bind_group = render_context.render_device().create_bind_group(
            "disocclusion_bind_group",
            &pipelines.disocclusion_bind_group_layout,
            &BindGroupEntries::with_indices((
                (0, view_uniforms.clone()),
                (9, globals_binding.clone()),
                (21, &pipelines.linear_samplers[0]),
                (22, &pipelines.linear_samplers[1]),
                (23, &disocclusion_textures.history_read.default_view),
                (24, &disocclusion_textures.history_normals_read.default_view),
                (25, &prepass_motion_vectors_texture.default_view),
                (26, uniforms),
                (27, &prepass_depth_texture.default_view),
                (28, &prepass_normal_texture.default_view),
            )),
        );

        {
            let mut disocclusion_pass =
                render_context.begin_tracked_render_pass(RenderPassDescriptor {
                    label: Some("disocclusion_pass"),
                    color_attachments: &[
                        Some(RenderPassColorAttachment {
                            view: &disocclusion_textures.output.default_view,
                            resolve_target: None,
                            ops: Operations::default(),
                        }),
                        Some(RenderPassColorAttachment {
                            view: &disocclusion_textures.history_write.default_view,
                            resolve_target: None,
                            ops: Operations::default(),
                        }),
                        Some(RenderPassColorAttachment {
                            view: &disocclusion_textures.history_normals_write.default_view,
                            resolve_target: None,
                            ops: Operations::default(),
                        }),
                    ],
                    depth_stencil_attachment: None,
                });
            disocclusion_pass.set_render_pipeline(disocclusion_pipeline);
            disocclusion_pass.set_bind_group(
                0,
                &disocclusion_bind_group,
                &[view_uniform_offset.offset, uniform_index.index()],
            );
            if let Some(viewport) = camera.viewport.as_ref() {
                disocclusion_pass.set_camera_viewport(viewport);
            }
            disocclusion_pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}

#[derive(Resource)]
struct DisocclusionPipeline {
    disocclusion_bind_group_layout: BindGroupLayout,
    linear_samplers: [Sampler; 2],
}

impl FromWorld for DisocclusionPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let linear_discriptor = SamplerDescriptor {
            label: Some("disocclusion_linear_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..SamplerDescriptor::default()
        };

        let linear_samplers = [
            render_device.create_sampler(&linear_discriptor),
            render_device.create_sampler(&linear_discriptor),
        ];

        let disocclusion_bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("disocclusion_bind_group_layout"),
                entries: &[
                    // View
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: Some(ViewUniform::min_size()),
                        },
                        count: None,
                    },
                    // Globals
                    BindGroupLayoutEntry {
                        binding: 9,
                        visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(GlobalsUniform::min_size()),
                        },
                        count: None,
                    },
                    // View target Linear sampler
                    BindGroupLayoutEntry {
                        binding: 21,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Disocclusion History Linear sampler
                    BindGroupLayoutEntry {
                        binding: 22,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Disocclusion Data History (read)
                    BindGroupLayoutEntry {
                        binding: 23,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Disocclusion Data Normals History (read)
                    BindGroupLayoutEntry {
                        binding: 24,
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
                        binding: 25,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Disocclusion Parameters
                    BindGroupLayoutEntry {
                        binding: 26,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: Some(DisocclusionUniforms::min_size()),
                        },
                        visibility: ShaderStages::FRAGMENT,
                        count: None,
                    },
                    // Depth
                    BindGroupLayoutEntry {
                        binding: 27,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Depth,
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Normals
                    BindGroupLayoutEntry {
                        binding: 28,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        DisocclusionPipeline {
            disocclusion_bind_group_layout,
            linear_samplers,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct DisocclusionPipelineKey {
    velocity_rejection: bool,
    depth_rejection: bool,
    normals_rejection: bool,
}

impl SpecializedRenderPipeline for DisocclusionPipeline {
    type Key = DisocclusionPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = vec![];

        if key.velocity_rejection {
            shader_defs.push("VELOCITY_REJECTION".into())
        }

        if key.depth_rejection {
            shader_defs.push("DEPTH_REJECTION".into())
        }

        if key.normals_rejection {
            shader_defs.push("NORMALS_REJECTION".into())
        }

        // #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
        #[cfg(target_arch = "wasm32")]
        shader_defs.push("WEBGL2".into());

        RenderPipelineDescriptor {
            label: Some("disocclusion_pipeline".into()),
            layout: vec![self.disocclusion_bind_group_layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: DISOCCLUSION_SHADER_HANDLE,
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: NORMAL_PREPASS_FORMAT,
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

fn extract_disocclusion_settings(
    mut commands: Commands,
    mut main_world: ResMut<MainWorld>,
    mut inverse_view_proj: Local<Mat4>,
) {
    let mut cameras_3d = main_world.query_filtered::<(
        Entity,
        &Camera,
        &GlobalTransform,
        &Projection,
        &mut DisocclusionSettings,
        &TemporalJitter,
    ), (
        With<Camera3d>,
        With<DepthPrepass>,
        With<MotionVectorPrepass>,
    )>();

    for (entity, camera, transform, camera_projection, disocclusion_settings, _temporal_jitter) in
        cameras_3d.iter_mut(&mut main_world)
    {
        if let (
            Some(URect {
                min: viewport_origin,
                ..
            }),
            Some(viewport_size),
        ) = (
            camera.physical_viewport_rect(),
            camera.physical_viewport_size(),
        ) {
            let _viewport = vec4(
                viewport_origin.x as f32,
                viewport_origin.y as f32,
                viewport_size.x as f32,
                viewport_size.y as f32,
            );
            let unjittered_projection = camera.projection_matrix();
            let projection = unjittered_projection;

            //temporal_jitter.jitter_projection(&mut projection, viewport.zw());

            let inverse_projection = projection.inverse();
            let view = transform.compute_matrix();
            //let inverse_view = view.inverse();

            //let view_proj = if temporal_jitter.is_some() {
            //    projection * inverse_view
            //} else {
            //    camera_view
            //        .view_projection
            //        .unwrap_or_else(|| projection * inverse_view)
            //};

            let has_perspective_projection =
                matches!(camera_projection, Projection::Perspective(_));

            let prev_inverse_view_proj = *inverse_view_proj;
            *inverse_view_proj = view * inverse_projection;

            if camera.is_active && has_perspective_projection {
                commands
                    .get_or_spawn(entity)
                    .insert(disocclusion_settings.clone())
                    .insert(DisocclusionUniforms {
                        inverse_view_proj: *inverse_view_proj,
                        prev_inverse_view_proj,
                        velocity_disocclusion: disocclusion_settings
                            .velocity_disocclusion
                            .unwrap_or(0.0),
                        depth_disocclusion_px_radius: disocclusion_settings
                            .depth_disocclusion_px_radius
                            .unwrap_or(0.0),
                        normals_disocclusion_scale: disocclusion_settings
                            .normals_disocclusion_scale
                            .unwrap_or(0.0),
                    });
            }
        }
    }
}

#[derive(Component)]
pub struct DisocclusionTextures {
    pub output: CachedTexture,
    history_write: CachedTexture,
    history_read: CachedTexture,
    history_normals_write: CachedTexture,
    history_normals_read: CachedTexture,
}

fn prepare_disocclusion_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    frame_count: Res<FrameCount>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        &ExtractedView,
        &DisocclusionSettings,
    )>,
) {
    for (entity, camera, _view, _settings) in &views {
        if let Some(physical_viewport_size) = camera.physical_viewport_size {
            let mut output_texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            };

            let mut history_texture_descriptor = TextureDescriptor {
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

            let mut history_normals_texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: NORMAL_PREPASS_FORMAT,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            };

            output_texture_descriptor.label = Some("disocclusion_output_texture");
            let output_texture =
                texture_cache.get(&render_device, output_texture_descriptor.clone());

            history_texture_descriptor.label = Some("disocclusion_history_1_texture");
            let history_1_texture =
                texture_cache.get(&render_device, history_texture_descriptor.clone());

            history_texture_descriptor.label = Some("disocclusion_history_2_texture");
            let history_2_texture = texture_cache.get(&render_device, history_texture_descriptor);

            history_normals_texture_descriptor.label =
                Some("disocclusion_history_normals_1_texture");
            let history_normals_1_texture =
                texture_cache.get(&render_device, history_normals_texture_descriptor.clone());

            history_normals_texture_descriptor.label =
                Some("disocclusion_history_normals_2_texture");
            let history_normals_2_texture =
                texture_cache.get(&render_device, history_normals_texture_descriptor);

            commands.entity(entity).insert(if frame_count.0 % 2 == 0 {
                DisocclusionTextures {
                    output: output_texture,
                    history_write: history_1_texture,
                    history_read: history_2_texture,
                    history_normals_write: history_normals_1_texture,
                    history_normals_read: history_normals_2_texture,
                }
            } else {
                DisocclusionTextures {
                    output: output_texture,
                    history_write: history_2_texture,
                    history_read: history_1_texture,
                    history_normals_write: history_normals_2_texture,
                    history_normals_read: history_normals_1_texture,
                }
            });
        }
    }
}

#[derive(Component)]
struct DisocclusionPipelineId(CachedRenderPipelineId);

fn prepare_disocclusion_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<DisocclusionPipeline>>,
    pipeline: Res<DisocclusionPipeline>,
    views: Query<(Entity, &ExtractedView, &DisocclusionSettings)>,
) {
    for (entity, _view, disocclusion_settings) in &views {
        let pipeline_key = DisocclusionPipelineKey {
            velocity_rejection: disocclusion_settings.velocity_disocclusion.is_some(),
            depth_rejection: disocclusion_settings.depth_disocclusion_px_radius.is_some(),
            normals_rejection: disocclusion_settings.normals_disocclusion_scale.is_some(),
        };
        let pipeline_id = pipelines.specialize(&pipeline_cache, &pipeline, pipeline_key.clone());

        commands
            .entity(entity)
            .insert(DisocclusionPipelineId(pipeline_id));
    }
}
