use anyhow::Result;
use cgmath::{Matrix4, Rad};
use std::num::NonZeroUsize;
use std::sync::Arc;
use vello::kurbo::{Affine, Circle, Ellipse, Line, RoundedRect, Stroke};
use vello::peniko::Color;
use vello::util::{RenderContext, RenderSurface};
use vello::{AaConfig, Renderer, RendererOptions, Scene};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

use bytemuck::{Pod, Zeroable};
use vello::wgpu;

/// Simple struct to hold the state of the renderer
#[derive(Debug)]
pub struct ActiveRenderState<'s> {
    // The fields MUST be in this order, so that the surface is dropped before the window
    surface: RenderSurface<'s>,
    window: Arc<Window>,
}

enum RenderState<'s> {
    Active(ActiveRenderState<'s>),
    // Cache a window so that it can be reused when the app is resumed after being suspended
    Suspended(Option<Arc<Window>>),
}

// 3D rendering structs
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// 3D rendering resources
struct Render3D {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    transform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct App<'s> {
    // The vello RenderContext which is a global context that lasts for the
    // lifetime of the application
    context: RenderContext,

    // An array of renderers, one per wgpu device
    renderers: Vec<Option<Renderer>>,

    // State for our example where we store the winit Window and the wgpu Surface
    state: RenderState<'s>,

    // A vello Scene which is a data structure which allows one to build up a
    // description a scene to be drawn (with paths, fills, images, text, etc)
    // which is then passed to a renderer for rendering
    scene: Scene,

    // 3D rendering resources
    render_3d: Option<Render3D>,

    // Rotation angle for animation
    rotation: f32,
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let RenderState::Suspended(cached_window) = &mut self.state else {
            return;
        };

        // Get the winit window cached in a previous Suspended event or else create a new window
        let window = cached_window
            .take()
            .unwrap_or_else(|| create_winit_window(event_loop));

        // Create a vello Surface
        let size = window.inner_size();
        let surface_future = self.context.create_surface(
            window.clone(),
            size.width,
            size.height,
            wgpu::PresentMode::AutoVsync,
        );
        let surface = pollster::block_on(surface_future).expect("Error creating surface");

        // Create a vello Renderer for the surface (using its device id)
        self.renderers
            .resize_with(self.context.devices.len(), || None);
        self.renderers[surface.dev_id]
            .get_or_insert_with(|| create_vello_renderer(&self.context, &surface));

        // Initialize 3D rendering resources
        self.render_3d = Some(create_3d_pipeline(
            &self.context.devices[surface.dev_id].device,
            surface.format,
        ));

        println!("Surface format: {:?}", surface.format);

        // Save the Window and Surface to a state variable
        self.state = RenderState::Active(ActiveRenderState { window, surface });
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let RenderState::Active(state) = &self.state {
            self.state = RenderState::Suspended(Some(state.window.clone()));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Ignore the event (return from the function) if
        //   - we have no render_state
        //   - OR the window id of the event doesn't match the window id of our render_state
        //
        // Else extract a mutable reference to the render state from its containing option for use below
        let render_state = match &mut self.state {
            RenderState::Active(state) if state.window.id() == window_id => state,
            _ => return,
        };

        match event {
            // Exit the event loop when a close is requested (e.g. window's close button is pressed)
            WindowEvent::CloseRequested => event_loop.exit(),

            // Resize the surface when the window is resized
            WindowEvent::Resized(size) => {
                self.context
                    .resize_surface(&mut render_state.surface, size.width, size.height);
            }

            // This is where all the rendering happens
            WindowEvent::RedrawRequested => {
                // Empty the scene of objects to draw
                self.scene.reset();

                // Re-add the objects to draw to the scene
                add_shapes_to_scene(&mut self.scene);

                // Get the RenderSurface (surface + config)
                let surface = &render_state.surface;

                // Get the window size
                let width = surface.config.width;
                let height = surface.config.height;

                // Get a handle to the device
                let device_handle = &self.context.devices[surface.dev_id];

                // Get the surface's texture
                let surface_texture = surface
                    .surface
                    .get_current_texture()
                    .expect("failed to get surface texture");

                // Create view for surface texture
                let view = surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                // 1. Create separate textures for 3D and Vello rendering
                let texture_descriptor = wgpu::TextureDescriptor {
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
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                };

                // Create 3D texture
                let texture_3d = device_handle.device.create_texture(&texture_descriptor);
                let texture_3d_view =
                    texture_3d.create_view(&wgpu::TextureViewDescriptor::default());

                // Create Vello texture
                let texture_vello = device_handle
                    .device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("Vello Render Texture"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm, // Use Rgba8Unorm format for Vello
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::STORAGE_BINDING, // Add STORAGE_BINDING flag
                        view_formats: &[],
                    });
                let texture_vello_view =
                    texture_vello.create_view(&wgpu::TextureViewDescriptor::default());

                // Create a command encoder for all rendering
                let mut encoder =
                    device_handle
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Combined Render Encoder"),
                        });

                // 2. Render 3D scene to its texture
                if let Some(render_3d) = &self.render_3d {
                    // Update rotation for animation
                    self.rotation += 0.01;

                    let transform: Matrix4<f32> = Matrix4::from_angle_z(Rad(self.rotation));
                    let transform_array: [[f32; 4]; 4] = transform.into();
                    device_handle.queue.write_buffer(
                        &render_3d.transform_buffer,
                        0,
                        bytemuck::cast_slice(&transform_array),
                    );

                    // Begin 3D render pass to the 3D texture
                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

                        // Draw 3D content
                        render_pass.set_pipeline(&render_3d.pipeline);
                        render_pass.set_bind_group(0, &render_3d.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, render_3d.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            render_3d.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint16,
                        );
                        render_pass.draw_indexed(0..render_3d.num_indices, 0, 0..1);
                    }
                }

                // Now render the 2D content with Vello
                self.renderers[surface.dev_id]
                    .as_mut()
                    .unwrap()
                    .render_to_texture(
                        &device_handle.device,
                        &device_handle.queue,
                        &self.scene,
                        &texture_vello_view,
                        &vello::RenderParams {
                            base_color: Color::new([0.0, 0.0, 0.0, 0.0]), // Transparent background
                            width,
                            height,
                            antialiasing_method: AaConfig::Msaa16,
                        },
                    )
                    .expect("failed to render to vello texture");

                // 4. Create the final composition pass to combine both textures
                {
                    // Create a bind group layout for our textures
                    let bind_group_layout = device_handle.device.create_bind_group_layout(
                        &wgpu::BindGroupLayoutDescriptor {
                            label: Some("Texture Bind Group Layout"),
                            entries: &[
                                // 3D texture binding
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Texture {
                                        sample_type: wgpu::TextureSampleType::Float {
                                            filterable: true,
                                        },
                                        view_dimension: wgpu::TextureViewDimension::D2,
                                        multisampled: false,
                                    },
                                    count: None,
                                },
                                // Vello texture binding
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Texture {
                                        sample_type: wgpu::TextureSampleType::Float {
                                            filterable: true,
                                        },
                                        view_dimension: wgpu::TextureViewDimension::D2,
                                        multisampled: false,
                                    },
                                    count: None,
                                },
                                // Sampler
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Sampler(
                                        wgpu::SamplerBindingType::Filtering,
                                    ),
                                    count: None,
                                },
                            ],
                        },
                    );

                    // Create the sampler for texture sampling
                    let sampler = device_handle
                        .device
                        .create_sampler(&wgpu::SamplerDescriptor {
                            address_mode_u: wgpu::AddressMode::ClampToEdge,
                            address_mode_v: wgpu::AddressMode::ClampToEdge,
                            address_mode_w: wgpu::AddressMode::ClampToEdge,
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Linear,
                            mipmap_filter: wgpu::FilterMode::Linear,
                            ..Default::default()
                        });

                    // Create the bind group for our textures
                    let bind_group =
                        device_handle
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("Texture Bind Group"),
                                layout: &bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            &texture_3d_view,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::TextureView(
                                            &texture_vello_view,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: wgpu::BindingResource::Sampler(&sampler),
                                    },
                                ],
                            });

                    // Create pipeline layout
                    let pipeline_layout = device_handle.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("Composition Pipeline Layout"),
                            bind_group_layouts: &[&bind_group_layout],
                            push_constant_ranges: &[],
                        },
                    );

                    // Create the composition shader
                    let comp_shader =
                        device_handle
                            .device
                            .create_shader_module(wgpu::ShaderModuleDescriptor {
                                label: Some("Composition Shader"),
                                source: wgpu::ShaderSource::Wgsl(
                                    include_str!("shader/blend_textures.wgsl").into(),
                                ),
                            });

                    // Create the composition pipeline
                    let composition_pipeline = device_handle.device.create_render_pipeline(
                        &wgpu::RenderPipelineDescriptor {
                            label: Some("Composition Pipeline"),
                            layout: Some(&pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &comp_shader,
                                entry_point: Some("vs_main"),
                                buffers: &[],
                                compilation_options: wgpu::PipelineCompilationOptions::default(),
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &comp_shader,
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
                        },
                    );

                    // Execute composition pass to the surface
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
                    comp_pass.draw(0..3, 0..1); // Draw a fullscreen triangle
                }

                // Submit all the work
                device_handle
                    .queue
                    .submit(std::iter::once(encoder.finish()));

                // Present the surface texture
                surface_texture.present();
                device_handle.device.poll(wgpu::Maintain::Poll); // TODO: what does this do? https://github.com/linebender/vello/blob/v0.4.x/examples/simple/src/main.rs#L161

                // Request another frame to keep animation going
                render_state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    // Setup a bunch of state:
    let mut app = App {
        context: RenderContext::new(),
        renderers: vec![],
        state: RenderState::Suspended(None),
        scene: Scene::new(),
        render_3d: None,
        rotation: 0.0,
    };

    // Create and run a winit event loop
    let event_loop = EventLoop::new()?;
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
    Ok(())
}

/// Helper function that creates a Winit window and returns it (wrapped in an Arc for sharing between threads)
fn create_winit_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    let attr = Window::default_attributes()
        .with_inner_size(LogicalSize::new(1044, 800))
        .with_resizable(true)
        .with_title("Vello + 3D Example");
    Arc::new(event_loop.create_window(attr).unwrap())
}

/// Helper function that creates a vello `Renderer` for a given `RenderContext` and `RenderSurface`
fn create_vello_renderer(render_cx: &RenderContext, surface: &RenderSurface<'_>) -> Renderer {
    Renderer::new(
        &render_cx.devices[surface.dev_id].device,
        RendererOptions {
            surface_format: Some(surface.format),
            use_cpu: false,
            antialiasing_support: vello::AaSupport::all(),
            num_init_threads: NonZeroUsize::new(1),
        },
    )
    .expect("Couldn't create renderer")
}

/// Add shapes to a vello scene. This does not actually render the shapes, but adds them
/// to the Scene data structure which represents a set of objects to draw.
fn add_shapes_to_scene(scene: &mut Scene) {
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

fn create_3d_pipeline(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Render3D {
    // Shader for 3D rendering
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("3D Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader/shader.wgsl").into()),
    });

    // Create a uniform buffer for the transform matrix
    let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Transform Buffer"),
        size: 64, // 4x4 matrix (16 floats)
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create a bind group layout for the transform matrix
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Transform Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    // Create a bind group for the transform matrix
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Transform Bind Group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: transform_buffer.as_entire_binding(),
        }],
    });

    // Create pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("3D Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create render pipeline
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("3D Render Pipeline"),
        layout: Some(&pipeline_layout),
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
                format: surface_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
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

    // Create vertex and index buffers for a simple cube
    let vertices = [
        // Front face (z = 0.5)
        Vertex {
            position: [-0.5, -0.5, 0.5],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
            color: [1.0, 1.0, 0.0],
        },
        // Back face (z = -0.5)
        Vertex {
            position: [-0.5, -0.5, -0.5],
            color: [1.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.5, -0.5, -0.5],
            color: [0.0, 1.0, 1.0],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
            color: [0.5, 0.5, 1.0],
        },
        Vertex {
            position: [-0.5, 0.5, -0.5],
            color: [1.0, 0.5, 0.5],
        },
    ];

    let indices: &[u16] = &[
        // Front face
        0, 1, 2, 2, 3, 0, // Back face
        4, 5, 6, 6, 7, 4, // Left face
        0, 3, 7, 7, 4, 0, // Right face
        1, 5, 6, 6, 2, 1, // Top face
        3, 2, 6, 6, 7, 3, // Bottom face
        0, 4, 5, 5, 1, 0,
    ];

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    Render3D {
        pipeline,
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
        transform_buffer,
        bind_group,
    }
}
