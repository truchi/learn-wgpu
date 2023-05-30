//! https://codelabs.developers.google.com/your-first-webgpu-app

#![allow(unused)]

use std::{
    mem::size_of,
    time::{Duration, Instant},
};
use wgpu::{
    include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferAddress, BufferBinding,
    BufferBindingType, BufferUsages, Color, ColorTargetState, ComputePipeline,
    ComputePipelineDescriptor, Device, FragmentState, Instance, LoadOp, Operations,
    PipelineLayoutDescriptor, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, ShaderStages, Surface,
    SurfaceConfiguration, TextureFormat, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexState,
};
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn default<T: Default>() -> T {
    T::default()
}

// =================================================================================================

pub fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = State::new(window).unwrap();
    let mut last_redraw = Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window().id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    _ => {}
                }
            }
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            state.window().request_redraw();
        }
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            let now = Instant::now();

            if now - last_redraw > Duration::from_millis(500) {
                last_redraw = now;

                state.update();
                match state.render() {
                    Some(()) => {}
                    None => *control_flow = ControlFlow::Exit,
                }
            }
        }
        _ => {}
    });
}

// =================================================================================================

const WORKGROUP: usize = 8;

const CLEAR: Color = Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};

const GRID_SIZE: u32 = 32;

const VERTICES: &[f32] = &[
    // X, Y
    -0.8, -0.8, // Triangle 1 (Blue)
    0.8, -0.8, //
    0.8, 0.8, //
    -0.8, -0.8, // Triangle 2 (Red)
    0.8, 0.8, //
    -0.8, 0.8, //
];

struct State {
    window: Window,
    surface: Surface,
    config: SurfaceConfiguration,
    device: Device,
    queue: Queue,
    vertex_buffer: Buffer,
    bind_groups: [BindGroup; 2],
    render_pipeline: RenderPipeline,
    compute_pipeline: ComputePipeline,
    step: usize,
}

impl State {
    fn new(window: Window) -> Option<Self> {
        let size = window.inner_size();

        // WGPU instance
        let instance = Instance::new(default());

        // Surface (window/canvas)
        //
        // SAFETY:
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.ok()?;

        // Request adapter (device handle), device (gpu connection) and queue (handle to command queue)
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..default()
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&default(), None)).ok()?;

        // Configure surface
        let config = surface.get_default_config(&adapter, size.width, size.height)?;
        assert!(config.format == TextureFormat::Rgba8UnormSrgb);
        surface.configure(&device, &config);

        // Vertices buffer
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: BufferUsages::VERTEX,
        });
        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: 2 * size_of::<f32>() as BufferAddress,
            step_mode: default(),
            attributes: &[VertexAttribute {
                format: VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            }],
        };

        // Grid uniform buffer
        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Grid uniforms"),
            contents: bytemuck::cast_slice(&[GRID_SIZE as f32, GRID_SIZE as f32]),
            usage: BufferUsages::UNIFORM,
        });

        // Cell activation storage buffers
        let activation_storages = {
            let activation_storage = |modulo| {
                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Activation storage"),
                    contents: bytemuck::cast_slice(
                        (0..GRID_SIZE * GRID_SIZE)
                            .into_iter()
                            .map(|i| if i % modulo == 0 { 1 } else { 0 })
                            .collect::<Vec<u32>>()
                            .as_slice(),
                    ),
                    usage: BufferUsages::STORAGE,
                })
            };
            [activation_storage(1), activation_storage(2)]
        };

        // Bind groups
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Cell bind group layout"),
            entries: &[
                // Grid uniform buffer
                // @binding(0)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX
                        | ShaderStages::COMPUTE
                        | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input activation
                // @binding(1)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output activation
                // @binding(2)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_groups = {
            let bind_group = |activation1, activation2| {
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Cell bind group"),
                    layout: &bind_group_layout,
                    entries: &[
                        // @binding(0)
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: &uniform_buffer,
                                offset: default(),
                                size: default(),
                            }),
                        },
                        // @binding(1)
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: activation1,
                                offset: default(),
                                size: default(),
                            }),
                        },
                        // @binding(2)
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: activation2,
                                offset: default(),
                                size: default(),
                            }),
                        },
                    ],
                })
            };
            [
                bind_group(&activation_storages[0], &activation_storages[1]),
                bind_group(&activation_storages[1], &activation_storages[0]),
            ]
        };

        // Shaders and pipelines
        let cell_shader_module = device.create_shader_module(include_wgsl!("cell.wgsl"));
        let simulation_shader_module =
            device.create_shader_module(include_wgsl!("simulation.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline layout"),
            bind_group_layouts: &[
                // @group(0)
                &bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &cell_shader_module,
                entry_point: "vertex_main",
                buffers: &[vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                module: &cell_shader_module,
                entry_point: "fragment_main",
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: default(), // Some(BlendState::REPLACE)?
                    write_mask: default(),
                })],
            }),
            primitive: default(),
            depth_stencil: default(),
            multisample: default(),
            multiview: default(),
        });
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &simulation_shader_module,
            entry_point: "compute_main",
        });

        Some(Self {
            window,
            surface,
            config,
            device,
            queue,
            vertex_buffer,
            bind_groups,
            render_pipeline,
            compute_pipeline,
            step: 0,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Option<()> {
        let mut encoder = self.device.create_command_encoder(&default());

        {
            let mut compute_pass = encoder.begin_compute_pass(&default());
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_groups[self.step], &[]);
            compute_pass.dispatch_workgroups(
                (GRID_SIZE as f32 / WORKGROUP as f32).ceil() as u32,
                (GRID_SIZE as f32 / WORKGROUP as f32).ceil() as u32,
                0,
            );
        }

        self.step = if self.step == 0 { 1 } else { 0 };

        let output = self.surface.get_current_texture().ok()?;
        let view = output.texture.create_view(&default());

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(CLEAR),
                        store: true,
                    },
                })],
                ..default()
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_groups[self.step], &[]);
            render_pass.draw(0..VERTICES.len() as u32 / 2, 0..GRID_SIZE * GRID_SIZE);
        }

        self.queue.submit([encoder.finish()]);
        output.present();

        Some(())
    }
}
