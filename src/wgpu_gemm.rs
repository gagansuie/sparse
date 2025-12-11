//! WGPU-based GEMM kernels for cross-platform GPU inference.
//!
//! Supports AMD, Intel, NVIDIA, and Apple Silicon GPUs via Vulkan/Metal/DX12.

use wgpu::util::DeviceExt;

/// G8 GEMM context for wgpu compute
pub struct G8GemmContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl G8GemmContext {
    /// Create a new G8 GEMM context
    pub async fn new() -> Result<Self, String> {
        // Request adapter (works on AMD, Intel, NVIDIA, Apple)
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find GPU adapter")?;

        let adapter_info = adapter.get_info();
        println!(
            "[tenpak] Using GPU: {} ({:?})",
            adapter_info.name, adapter_info.backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("tenpak"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        // Create compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("g8_gemm_shader"),
            source: wgpu::ShaderSource::Wgsl(G8_GEMM_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("g8_gemm_bind_group_layout"),
            entries: &[
                // X input [M, K]
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // W weights [N, K/2] packed int4
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Scales [N, K/8]
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Offsets [N, K/8]
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Y output [M, N]
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniforms (M, N, K)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("g8_gemm_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("g8_gemm_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Perform G8 GEMM: Y = X @ W^T
    ///
    /// X: [M, K] f32
    /// W: [N, K/2] u8 (packed int4)
    /// scales: [N, K/8] f32
    /// offsets: [N, K/8] f32
    /// Y: [M, N] f32
    pub fn gemm(
        &self,
        x: &[f32],
        w: &[u8],
        scales: &[f32],
        offsets: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Vec<f32> {
        // Create buffers
        let x_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("x_buffer"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let w_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w_buffer"),
                contents: w,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let scales_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scales_buffer"),
                contents: bytemuck::cast_slice(scales),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let offsets_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("offsets_buffer"),
                contents: bytemuck::cast_slice(offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let y_size = (m * n) as usize * std::mem::size_of::<f32>();
        let y_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y_buffer"),
            size: y_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniforms = [m, n, k, 0]; // padding for alignment
        let uniforms_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms_buffer"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g8_gemm_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scales_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniforms_buffer.as_entire_binding(),
                },
            ],
        });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: y_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Encode and submit
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("g8_gemm_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("g8_gemm_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (16x16 threads per workgroup)
            let workgroup_x = (n + 15) / 16;
            let workgroup_y = (m + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        encoder.copy_buffer_to_buffer(&y_buffer, 0, &staging_buffer, 0, y_size as u64);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }
}

/// WGSL compute shader for G8 GEMM
const G8_GEMM_SHADER: &str = r#"
// G8 GEMM Shader: Y = X @ W^T
// W is int4 quantized with group size 8

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;  // packed as u32 for efficiency
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

// Workgroup size
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const GROUP_SIZE: u32 = 8u;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;  // M dimension
    let col = global_id.x;  // N dimension
    
    if (row >= uniforms.M || col >= uniforms.N) {
        return;
    }
    
    let K = uniforms.K;
    let num_groups = K / GROUP_SIZE;
    
    var acc: f32 = 0.0;
    
    // Process K dimension in groups of 8
    for (var g: u32 = 0u; g < num_groups; g = g + 1u) {
        // Load scale and offset for this group
        let scale = scales[col * num_groups + g];
        let offset = offsets[col * num_groups + g];
        
        let k_base = g * GROUP_SIZE;
        
        // Load 4 bytes = 8 int4 values (one full group)
        // W is stored as [N, K/2] bytes, but we read as u32 for efficiency
        let w_idx = col * (K / 2u) + k_base / 2u;
        let packed = W[w_idx / 4u];  // Read 4 bytes as u32
        let byte_offset = (w_idx % 4u) * 8u;
        
        // Unpack and process 8 weights
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            // Get byte from packed u32
            let byte_idx = k_base / 2u + i;
            let packed_byte = (W[byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu;
            
            // Low nibble
            let w0 = f32(packed_byte & 0xFu) * scale + offset;
            let x0 = X[row * K + k_base + i * 2u];
            acc = acc + x0 * w0;
            
            // High nibble
            let w1 = f32((packed_byte >> 4u) & 0xFu) * scale + offset;
            let x1 = X[row * K + k_base + i * 2u + 1u];
            acc = acc + x1 * w1;
        }
    }
    
    Y[row * uniforms.N + col] = acc;
}
"#;

/// G16 GEMM context for wgpu compute (matches int4_opt_v1)
pub struct G16GemmContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl G16GemmContext {
    /// Create a new G16 GEMM context
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find GPU adapter")?;

        let adapter_info = adapter.get_info();
        println!(
            "[tenpak] G16 GEMM using GPU: {} ({:?})",
            adapter_info.name, adapter_info.backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("tenpak_g16"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("g16_gemm_shader"),
            source: wgpu::ShaderSource::Wgsl(G16_GEMM_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("g16_gemm_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("g16_gemm_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("g16_gemm_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Compute Y = X @ W^T where W is int4 quantized with group size 16
    pub fn gemm(
        &self,
        x: &[f32],
        w: &[u8],
        scales: &[f32],
        offsets: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Vec<f32> {
        let x_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("x_buffer"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let w_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w_buffer"),
                contents: w,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let scales_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scales_buffer"),
                contents: bytemuck::cast_slice(scales),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let offsets_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("offsets_buffer"),
                contents: bytemuck::cast_slice(offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let y_size = (m * n) as usize * std::mem::size_of::<f32>();
        let y_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y_buffer"),
            size: y_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniforms = [m, n, k, 0];
        let uniforms_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms_buffer"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g16_gemm_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scales_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniforms_buffer.as_entire_binding(),
                },
            ],
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: y_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("g16_gemm_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("g16_gemm_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_x = (n + 15) / 16;
            let workgroup_y = (m + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        encoder.copy_buffer_to_buffer(&y_buffer, 0, &staging_buffer, 0, y_size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }
}

/// WGSL compute shader for G16 GEMM (matches int4_opt_v1)
const G16_GEMM_SHADER: &str = r#"
// G16 GEMM Shader: Y = X @ W^T
// W is int4 quantized with group size 16 (matches int4_opt_v1)

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

const GROUP_SIZE: u32 = 16u;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= uniforms.M || col >= uniforms.N) {
        return;
    }
    
    let K = uniforms.K;
    let num_groups = K / GROUP_SIZE;
    
    var acc: f32 = 0.0;
    
    // Process K dimension in groups of 16
    for (var g: u32 = 0u; g < num_groups; g = g + 1u) {
        let scale = scales[col * num_groups + g];
        let offset = offsets[col * num_groups + g];
        let k_base = g * GROUP_SIZE;
        
        // Process 16 weights (8 bytes)
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            let byte_idx = col * (K / 2u) + k_base / 2u + i;
            let packed_byte = (W[byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu;
            
            let w0 = f32(packed_byte & 0xFu) * scale + offset;
            let x0 = X[row * K + k_base + i * 2u];
            acc = acc + x0 * w0;
            
            let w1 = f32((packed_byte >> 4u) & 0xFu) * scale + offset;
            let x1 = X[row * K + k_base + i * 2u + 1u];
            acc = acc + x1 * w1;
        }
    }
    
    Y[row * uniforms.N + col] = acc;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_g8_gemm() {
        let ctx = G8GemmContext::new()
            .await
            .expect("Failed to create context");

        // Simple test: 2x4 @ 4x3 = 2x3
        let m = 2u32;
        let n = 3u32;
        let k = 8u32; // Must be multiple of 8

        // X: [2, 8]
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();

        // W: [3, 8] -> quantized to [3, 4] packed
        let w_fp: Vec<f32> = (0..24).map(|i| i as f32 * 0.05).collect();

        // Quantize W
        let mut w_packed = vec![0u8; 12]; // 3 * 4 bytes
        let mut scales = vec![0f32; 3]; // 3 output channels, 1 group each
        let mut offsets = vec![0f32; 3];

        for out_ch in 0..3 {
            let group = &w_fp[out_ch * 8..(out_ch + 1) * 8];
            let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = (max_val - min_val) / 15.0;
            let scale = if scale < 1e-8 { 1.0 } else { scale };

            scales[out_ch] = scale;
            offsets[out_ch] = min_val;

            for i in 0..4 {
                let v0 = ((group[i * 2] - min_val) / scale).round().clamp(0.0, 15.0) as u8;
                let v1 = ((group[i * 2 + 1] - min_val) / scale)
                    .round()
                    .clamp(0.0, 15.0) as u8;
                w_packed[out_ch * 4 + i] = v0 | (v1 << 4);
            }
        }

        let y = ctx.gemm(&x, &w_packed, &scales, &offsets, m, n, k);

        println!("Y = {:?}", y);
        assert_eq!(y.len(), (m * n) as usize);
    }
}
