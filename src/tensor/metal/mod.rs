// Copyright 2026 Mentat AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::tensor::backend::{Backend, Device};
use crate::tensor::DataType;
use metal::{
    CommandQueue, ComputePipelineDescriptor, Device as MetalDevice, MTLResourceOptions, MTLSize,
};
use std::mem;
use std::sync::{Arc, OnceLock};
const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}

kernel void mul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] * b[id];
}

kernel void matmul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint row = id.y;
    uint col = id.x;
    
    if (row < M && col < N) {
        float sum = 0.0;
        for (uint i = 0; i < K; ++i) {
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}
"#;

#[derive(Debug)]
struct MetalBackendInner {
    device: MetalDevice,
    queue: CommandQueue,
    add_pipeline: metal::ComputePipelineState,
    mul_pipeline: metal::ComputePipelineState,
    matmul_pipeline: metal::ComputePipelineState,
}

#[derive(Debug, Clone)]
pub struct MetalBackend {
    inner: Arc<MetalBackendInner>,
}

static METAL_BACKEND_INNER: OnceLock<Arc<MetalBackendInner>> = OnceLock::new();

impl Default for MetalBackend {
    fn default() -> Self {
        let inner = METAL_BACKEND_INNER.get_or_init(|| {
            let device = MetalDevice::system_default().expect("No Metal device found");
            let queue = device.new_command_queue();
            
            let compile_options = metal::CompileOptions::new();
            let library = device
                .new_library_with_source(MSL_SOURCE, &compile_options)
                .expect("Failed to compile Metal library");

            let add_function = library.get_function("add_kernel", None).unwrap();
            let add_desc = ComputePipelineDescriptor::new();
            add_desc.set_compute_function(Some(&add_function));
            let add_pipeline = device.new_compute_pipeline_state_with_function(&add_desc.compute_function().unwrap()).unwrap();

            let mul_function = library.get_function("mul_kernel", None).unwrap();
            let mul_desc = ComputePipelineDescriptor::new();
            mul_desc.set_compute_function(Some(&mul_function));
            let mul_pipeline = device.new_compute_pipeline_state_with_function(&mul_desc.compute_function().unwrap()).unwrap();

            let matmul_function = library.get_function("matmul_kernel", None).unwrap();
            let matmul_desc = ComputePipelineDescriptor::new();
            matmul_desc.set_compute_function(Some(&matmul_function));
            let matmul_pipeline = device.new_compute_pipeline_state_with_function(&matmul_desc.compute_function().unwrap()).unwrap();

            Arc::new(MetalBackendInner {
                device,
                queue,
                add_pipeline,
                mul_pipeline,
                matmul_pipeline,
            })
        });

        MetalBackend {
            inner: inner.clone(),
        }
    }
}

impl Backend for MetalBackend {
    fn device(&self) -> Device {
        Device::Metal(0)
    }

    fn add(&self, _shape: &[usize], _dtype: &DataType, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        let size = a.len();
        let byte_size = (size * mem::size_of::<f32>()) as u64;

        let buffer_a = self.inner.device.new_buffer_with_data(
            unsafe { mem::transmute(a.as_ptr()) },
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_b = self.inner.device.new_buffer_with_data(
            unsafe { mem::transmute(b.as_ptr()) },
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_c = self.inner.device.new_buffer(byte_size, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.inner.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.inner.add_pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_c), 0);

        let grid_size = MTLSize::new(size as u64, 1, 1);
        let threadgroup_size = MTLSize::new(self.inner.add_pipeline.max_total_threads_per_threadgroup(), 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut c = vec![0.0f32; size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer_c.contents() as *const f32,
                c.as_mut_ptr(),
                size,
            );
        }

        Ok(c)
    }

    fn mul(&self, _shape: &[usize], _dtype: &DataType, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        let size = a.len();
        let byte_size = (size * mem::size_of::<f32>()) as u64;

        let buffer_a = self.inner.device.new_buffer_with_data(
            unsafe { mem::transmute(a.as_ptr()) },
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_b = self.inner.device.new_buffer_with_data(
            unsafe { mem::transmute(b.as_ptr()) },
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_c = self.inner.device.new_buffer(byte_size, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.inner.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.inner.mul_pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_c), 0);

        let grid_size = MTLSize::new(size as u64, 1, 1);
        let threadgroup_size = MTLSize::new(self.inner.mul_pipeline.max_total_threads_per_threadgroup(), 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut c = vec![0.0f32; size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer_c.contents() as *const f32,
                c.as_mut_ptr(),
                size,
            );
        }

        Ok(c)
    }

    fn matmul(
        &self,
        shape_a: &[usize],
        shape_b: &[usize],
        _dtype: &DataType,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String> {
        let m = shape_a[0];
        let k = shape_a[1];
        let n = shape_b[1];

        let byte_size_a = (m * k * mem::size_of::<f32>()) as u64;
        let byte_size_b = (k * n * mem::size_of::<f32>()) as u64;
        let byte_size_c = (m * n * mem::size_of::<f32>()) as u64;

        let buffer_a = self.inner.device.new_buffer_with_data(
            unsafe { mem::transmute(a.as_ptr()) },
            byte_size_a,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_b = self.inner.device.new_buffer_with_data(
            unsafe { mem::transmute(b.as_ptr()) },
            byte_size_b,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_c = self.inner.device.new_buffer(byte_size_c, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.inner.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.inner.matmul_pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_c), 0);
        
        encoder.set_bytes(3, mem::size_of::<u32>() as u64, unsafe { mem::transmute(&(m as u32)) });
        encoder.set_bytes(4, mem::size_of::<u32>() as u64, unsafe { mem::transmute(&(k as u32)) });
        encoder.set_bytes(5, mem::size_of::<u32>() as u64, unsafe { mem::transmute(&(n as u32)) });

        let grid_size = MTLSize::new(n as u64, m as u64, 1);
        let _w = self.inner.matmul_pipeline.max_total_threads_per_threadgroup();
        let threadgroup_size = MTLSize::new(16, 16, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut c = vec![0.0f32; m * n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer_c.contents() as *const f32,
                c.as_mut_ptr(),
                m * n,
            );
        }

        Ok(c)
    }
}
