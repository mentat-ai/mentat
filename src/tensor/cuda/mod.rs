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
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const CU_SOURCE: &str = r#"
extern "C" __global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        c[id] = a[id] + b[id];
    }
}

extern "C" __global__ void mul_kernel(const float* a, const float* b, float* c, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        c[id] = a[id] * b[id];
    }
}

extern "C" __global__ void matmul_kernel(const float* a, const float* b, float* c, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0;
        for (int i = 0; i < K; ++i) {
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}
"#;

#[derive(Debug, Clone)]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
}

use std::sync::OnceLock;

static CUDA_BACKEND_DEVICE: OnceLock<Arc<CudaDevice>> = OnceLock::new();

impl Default for CudaBackend {
    fn default() -> Self {
        let device = CUDA_BACKEND_DEVICE.get_or_init(|| {
            let device = CudaDevice::new(0).expect("No CUDA device found");
            
            let ptx = compile_ptx(CU_SOURCE).expect("Failed to compile PTX");
            device.load_ptx(ptx, "kernels", &["add_kernel", "mul_kernel", "matmul_kernel"])
                  .expect("Failed to load PTX modules");
            device
        });

        CudaBackend { device: device.clone() }
    }
}

impl Backend for CudaBackend {
    fn device(&self) -> Device {
        Device::Cuda(0)
    }

    fn add(&self, _shape: &[usize], _dtype: &DataType, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        let size = a.len();
        
        let d_a = self.device.htod_copy(a.to_vec()).unwrap();
        let d_b = self.device.htod_copy(b.to_vec()).unwrap();
        let mut d_c = self.device.alloc_zeros::<f32>(size).unwrap();

        let f = self.device.get_func("kernels", "add_kernel").unwrap();
        let cfg = LaunchConfig::for_num_elems(size as u32);
        
        unsafe { f.launch(cfg, (&d_a, &d_b, &mut d_c, size as i32)) }.unwrap();

        let c = self.device.sync_reclaim(d_c).unwrap();
        Ok(c)
    }

    fn mul(&self, _shape: &[usize], _dtype: &DataType, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        let size = a.len();
        
        let d_a = self.device.htod_copy(a.to_vec()).unwrap();
        let d_b = self.device.htod_copy(b.to_vec()).unwrap();
        let mut d_c = self.device.alloc_zeros::<f32>(size).unwrap();

        let f = self.device.get_func("kernels", "mul_kernel").unwrap();
        let cfg = LaunchConfig::for_num_elems(size as u32);
        
        unsafe { f.launch(cfg, (&d_a, &d_b, &mut d_c, size as i32)) }.unwrap();

        let c = self.device.sync_reclaim(d_c).unwrap();
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

        let d_a = self.device.htod_copy(a.to_vec()).unwrap();
        let d_b = self.device.htod_copy(b.to_vec()).unwrap();
        let mut d_c = self.device.alloc_zeros::<f32>(m * n).unwrap();

        let f = self.device.get_func("kernels", "matmul_kernel").unwrap();
        
        let block_size = (16, 16, 1);
        let grid_size = (
            (n as u32 + block_size.0 - 1) / block_size.0,
            (m as u32 + block_size.1 - 1) / block_size.1,
            1,
        );
        let cfg = LaunchConfig {
            grid_dim: grid_size,
            block_dim: block_size,
            shared_mem_bytes: 0,
        };
        
        unsafe { f.launch(cfg, (&d_a, &d_b, &mut d_c, m as i32, k as i32, n as i32)) }.unwrap();

        let c = self.device.sync_reclaim(d_c).unwrap();
        Ok(c)
    }
}
