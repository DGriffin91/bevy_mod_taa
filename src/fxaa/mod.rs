use bevy::app::prelude::*;
use bevy::asset::{load_internal_asset, Handle};
use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::ecs::prelude::*;
use bevy::prelude::*;
use bevy::reflect::{std_traits::ReflectDefault, Reflect};
use bevy::render::render_graph::RenderLabel;
use bevy::render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_graph::RenderGraphApp,
    render_graph::ViewNodeRunner,
    render_resource::*,
    renderer::RenderDevice,
    texture::BevyDefault,
    view::{ExtractedView, ViewTarget},
    Render, RenderApp, RenderSet,
};

mod node;

pub use node::FxaaNode;

#[derive(Reflect, Eq, PartialEq, Hash, Clone, Copy)]
#[reflect(PartialEq, Hash)]
pub enum Sensitivity {
    UltraLow,
    Low,
    Medium,
    High,
    Ultra,
    Extreme,
}

impl Sensitivity {
    pub fn get_str(&self) -> &str {
        match self {
            Sensitivity::UltraLow => "ULTRA_LOW",
            Sensitivity::Low => "LOW",
            Sensitivity::Medium => "MEDIUM",
            Sensitivity::High => "HIGH",
            Sensitivity::Ultra => "ULTRA",
            Sensitivity::Extreme => "EXTREME",
        }
    }
}

#[derive(Reflect, Component, Clone, ExtractComponent)]
#[reflect(Component, Default)]
#[extract_component_filter(With<Camera>)]
pub struct FxaaPrepass {
    /// Enable render passes for FXAA.
    pub enabled: bool,

    /// Use lower sensitivity for a sharper, faster, result.
    /// Use higher sensitivity for a slower, smoother, result.
    /// [`Ultra`](`Sensitivity::Ultra`) and [`Extreme`](`Sensitivity::Extreme`)
    /// settings can result in significant smearing and loss of detail.

    /// The minimum amount of local contrast required to apply algorithm.
    pub edge_threshold: Sensitivity,

    /// Trims the algorithm from processing darks.
    pub edge_threshold_min: Sensitivity,
}

impl Default for FxaaPrepass {
    fn default() -> Self {
        FxaaPrepass {
            enabled: false,
            edge_threshold: Sensitivity::UltraLow,
            edge_threshold_min: Sensitivity::UltraLow,
        }
    }
}

impl FxaaPrepass {
    pub fn ultra_low() -> Self {
        FxaaPrepass {
            enabled: true,
            edge_threshold: Sensitivity::UltraLow,
            edge_threshold_min: Sensitivity::UltraLow,
        }
    }
}

const FXAA_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(923847520938471);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct FxaaPrepassLabel;

/// Adds support for Fast Approximate Anti-Aliasing (FXAA)
pub struct FxaaPrepassPlugin;
impl Plugin for FxaaPrepassPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, FXAA_SHADER_HANDLE, "fxaa.wgsl", Shader::from_wgsl);

        app.register_type::<FxaaPrepass>();
        app.add_plugins(ExtractComponentPlugin::<FxaaPrepass>::default());

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app
            .init_resource::<SpecializedRenderPipelines<FxaaPipeline>>()
            .add_systems(Render, prepare_fxaa_pipelines.in_set(RenderSet::Prepare))
            .add_render_graph_node::<ViewNodeRunner<FxaaNode>>(Core3d, FxaaPrepassLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::Tonemapping,
                    FxaaPrepassLabel,
                    Node3d::EndMainPassPostProcessing,
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<FxaaNode>>(Core2d, FxaaPrepassLabel)
            .add_render_graph_edges(
                Core2d,
                (
                    Node2d::Tonemapping,
                    FxaaPrepassLabel,
                    Node2d::EndMainPassPostProcessing,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app.init_resource::<FxaaPipeline>();
    }
}

#[derive(Resource, Deref)]
pub struct FxaaPipeline {
    texture_bind_group: BindGroupLayout,
}

impl FromWorld for FxaaPipeline {
    fn from_world(render_world: &mut World) -> Self {
        let texture_bind_group = render_world
            .resource::<RenderDevice>()
            .create_bind_group_layout(
                Some("fxaa_texture_bind_group_layout"),
                &[
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
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            );

        FxaaPipeline { texture_bind_group }
    }
}

#[derive(Component)]
pub struct CameraFxaaPipeline {
    pub pipeline_id: CachedRenderPipelineId,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct FxaaPipelineKey {
    edge_threshold: Sensitivity,
    edge_threshold_min: Sensitivity,
    texture_format: TextureFormat,
}

impl SpecializedRenderPipeline for FxaaPipeline {
    type Key = FxaaPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("fxaa".into()),
            layout: vec![self.texture_bind_group.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: FXAA_SHADER_HANDLE,
                shader_defs: vec![
                    format!("EDGE_THRESH_{}", key.edge_threshold.get_str()).into(),
                    format!("EDGE_THRESH_MIN_{}", key.edge_threshold_min.get_str()).into(),
                ],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: key.texture_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: Vec::new(),
        }
    }
}

pub fn prepare_fxaa_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<FxaaPipeline>>,
    fxaa_pipeline: Res<FxaaPipeline>,
    views: Query<(Entity, &ExtractedView, &FxaaPrepass)>,
) {
    for (entity, view, fxaa) in &views {
        if !fxaa.enabled {
            continue;
        }
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &fxaa_pipeline,
            FxaaPipelineKey {
                edge_threshold: fxaa.edge_threshold,
                edge_threshold_min: fxaa.edge_threshold_min,
                texture_format: if view.hdr {
                    ViewTarget::TEXTURE_FORMAT_HDR
                } else {
                    TextureFormat::bevy_default()
                },
            },
        );

        commands
            .entity(entity)
            .insert(CameraFxaaPipeline { pipeline_id });
    }
}
