use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use log::{error, info};
use peniko::Color;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;
use vello::kurbo::{Affine, Circle, Ellipse, Line, RoundedRect, Stroke};
use vello::util::{RenderContext, RenderSurface};
use vello::{AaConfig, Renderer, RendererOptions, Scene};
use wgpu::util::DeviceExt;
use wgpu::{Buffer, RenderPipeline};
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

struct State {
    last_update: Instant,
}

impl Default for State {
    fn default() -> Self {
        Self {
            last_update: Instant::now(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

struct Renderer3D {
    render_pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    num_vertices: u32,
    num_indices: u32,
}

impl Renderer3D {
    async fn new(render_context: &RenderContext, surface: &RenderSurface<'_>) -> Self {
        const VERTICES: &[Vertex] = &[
            Vertex {
                position: [-0.0868241, 0.49240386, 0.0],
                color: [0.5, 0.0, 0.5],
            }, // A
            Vertex {
                position: [-0.49513406, 0.06958647, 0.0],
                color: [0.5, 0.0, 0.5],
            }, // B
            Vertex {
                position: [-0.21918549, -0.44939706, 0.0],
                color: [0.5, 0.0, 0.5],
            }, // C
            Vertex {
                position: [0.35966998, -0.3473291, 0.0],
                color: [0.5, 0.0, 0.5],
            }, // D
            Vertex {
                position: [0.44147372, 0.2347359, 0.0],
                color: [0.5, 0.0, 0.5],
            }, // E
        ];
        const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

        let device = &render_context.devices[surface.dev_id].device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader/shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let num_vertices = VERTICES.len() as u32;

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            render_pipeline,
            vertex_buffer,
            num_vertices,
            index_buffer,
            num_indices,
        }
    }
}
struct Renderer2D {
    scene: Scene,
    vello: Renderer,
}

impl Renderer2D {
    fn new(render_cx: &RenderContext, surface: &RenderSurface<'_>) -> Self {
        Self {
            scene: Scene::new(),
            vello: Renderer::new(
                &render_cx.devices[surface.dev_id].device,
                RendererOptions {
                    surface_format: Some(surface.format),
                    use_cpu: false,
                    antialiasing_support: vello::AaSupport::all(),
                    num_init_threads: NonZeroUsize::new(1),
                },
            )
            .expect("Couldn't create renderer"),
        }
    }
}

#[derive(Default)]
pub struct App<'s> {
    window: Option<Arc<Window>>,
    renderer_3d: Option<Renderer3D>,
    renderer_2d: Option<Renderer2D>,
    render_context: Option<RenderContext>,
    surface: Option<RenderSurface<'s>>,
    state: Option<State>,
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = match event_loop.create_window(Window::default_attributes()) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                return error!("Failed to create window: {}", err);
            }
        };

        // create surface for 3d & 2d renderers
        let size = window.inner_size();
        let mut render_context = RenderContext::new();
        let surface_future = render_context.create_surface(
            window.clone(),
            size.width,
            size.height,
            wgpu::PresentMode::AutoVsync,
        );
        let surface = pollster::block_on(surface_future).expect("Error creating surface");

        self.window = Some(window);
        self.renderer_3d = Some(pollster::block_on(Renderer3D::new(
            &render_context,
            &surface,
        )));
        self.renderer_2d = Some(Renderer2D::new(&render_context, &surface));
        self.render_context = Some(render_context);
        self.surface = Some(surface);
        self.state = Some(State::default());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.as_ref() else {
            return info!("Skip window_event handling. We have no window");
        };
        let Some(render_context) = self.render_context.as_mut() else {
            return info!("Skip window_event handling. We have no render_context");
        };
        let Some(renderer_3d) = self.renderer_3d.as_mut() else {
            return info!("Skip window_event handling. We have no renderer_3d");
        };
        let Some(renderer_2d) = self.renderer_2d.as_mut() else {
            return info!("Skip window_event handling. We have no renderer_2d");
        };
        let Some(surface) = self.surface.as_mut() else {
            return info!("Skip window_event handling. We have no surface");
        };
        let Some(state) = self.state.as_mut() else {
            return info!("Skip window_event handling. We have no state");
        };

        match event {
            WindowEvent::CloseRequested => {
                info!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                info!("pressed key {:?}", event);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let _delta = now.duration_since(state.last_update).as_secs_f32();
                state.last_update = now;

                // Render 3D:
                {
                    // TODO add a 3d scene?
                }

                // Render 2D:
                {
                    let scene = &mut renderer_2d.scene;
                    scene.reset();

                    // Draw an outlined rectangle
                    let stroke = Stroke::new(6.0);
                    let rect = RoundedRect::new(10.0, 10.0, 240.0, 240.0, 20.0);
                    let rect_stroke_color = Color::new([0.9804, 0.702, 0.5294, 1.]);
                    scene.stroke(&stroke, Affine::IDENTITY, rect_stroke_color, None, &rect);

                    // Draw a filled circle
                    let circle = Circle::new((420.0, 200.0), 120.0);
                    let circle_fill_color = Color::new([0.9529, 0.5451, 0.6588, 1.]);
                    scene.fill(
                        vello::peniko::Fill::NonZero,
                        Affine::IDENTITY,
                        circle_fill_color,
                        None,
                        &circle,
                    );

                    // Draw a filled ellipse
                    let ellipse = Ellipse::new((250.0, 420.0), (100.0, 160.0), -90.0);
                    let ellipse_fill_color = Color::new([0.7961, 0.651, 0.9686, 0.5]);
                    scene.fill(
                        vello::peniko::Fill::NonZero,
                        Affine::IDENTITY,
                        ellipse_fill_color,
                        None,
                        &ellipse,
                    );

                    // Draw a straight line
                    let line = Line::new((260.0, 20.0), (620.0, 100.0));
                    let line_stroke_color = Color::new([0.5373, 0.7059, 0.9804, 1.]);
                    scene.stroke(&stroke, Affine::IDENTITY, line_stroke_color, None, &line);
                }

                render(render_context, surface, renderer_3d, renderer_2d);
                window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                render_context.resize_surface(surface, size.width, size.height);
            }
            _ => (),
        }
    }
}

fn render(
    render_context: &RenderContext,
    surface: &RenderSurface,
    renderer_3d: &mut Renderer3D,
    renderer_2d: &mut Renderer2D,
) {
    let width = surface.config.width;
    let height = surface.config.height;
    let device_handle = &render_context.devices[surface.dev_id];
    let surface_texture = surface
        .surface
        .get_current_texture()
        .expect("failed to get surface texture");
    let view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // Create 3D texture
    let texture_3d = device_handle
        .device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("3D Render Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
    let texture_3d_view = texture_3d.create_view(&wgpu::TextureViewDescriptor::default());

    // Create 2D texture
    let texture_2d = device_handle
        .device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("2D Render Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
    let texture_2d_view = texture_2d.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder =
        device_handle
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

    // renderer_3d.render_to_texture
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("3D Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_3d_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1, // Dark background for 3D scene
                        g: 0.1,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&renderer_3d.render_pipeline);
        render_pass.set_vertex_buffer(0, renderer_3d.vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            renderer_3d.index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        ); // 1.
        render_pass.draw_indexed(0..renderer_3d.num_indices, 0, 0..1);
    }

    // renderer_2d.render_to_texture
    {
        renderer_2d
            .vello
            .render_to_texture(
                &device_handle.device,
                &device_handle.queue,
                &renderer_2d.scene,
                &texture_2d_view,
                &vello::RenderParams {
                    // the 2d render_texture has to have a transparent background,
                    // because we overlay the 2d texture above the 3d rendering texture later.
                    base_color: Color::new([0.0, 0.0, 0.0, 0.0]),
                    width,
                    height,
                    antialiasing_method: AaConfig::Msaa16,
                },
            )
            .expect("failed to render to vello texture");
    }

    // Combine the 3D and 2D render textures into the surface_texture
    {
        let sampler = device_handle
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group_layout =
            device_handle
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Texture Bind Group Layout"),
                    entries: &[
                        // 3D texture binding
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // 2D texture binding
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Sampler binding
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let bind_group = device_handle
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Texture Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_3d_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_2d_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        let pipeline_layout =
            device_handle
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Composition Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let shader = device_handle
            .device
            .create_shader_module(wgpu::include_wgsl!("shader/blend_textures.wgsl"));

        let composition_pipeline =
            device_handle
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Composition Pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: surface.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

        let mut comp_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Composition Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        comp_pass.set_pipeline(&composition_pipeline);
        comp_pass.set_bind_group(0, &bind_group, &[]);
        comp_pass.draw(0..3, 0..1); // The comp_pass shader draws a single fullscreen triangle
    }

    device_handle
        .queue
        .submit(std::iter::once(encoder.finish()));
    surface_texture.present();
}

fn main() -> Result<(), EventLoopError> {
    env_logger::init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)
}
