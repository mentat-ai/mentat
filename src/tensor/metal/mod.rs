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

#[derive(Debug, Clone)]
pub struct MetalBackend;

impl Backend for MetalBackend {
    fn device(&self) -> Device {
        Device::Metal(0)
    }

    fn add(&self, _shape: &[usize], _dtype: &DataType, _a: &[f32], _b: &[f32]) -> Result<Vec<f32>, String> {
        unimplemented!("Metal backend add is not yet implemented")
    }

    fn mul(&self, _shape: &[usize], _dtype: &DataType, _a: &[f32], _b: &[f32]) -> Result<Vec<f32>, String> {
        unimplemented!("Metal backend mul is not yet implemented")
    }

    fn matmul(
        &self,
        _shape_a: &[usize],
        _shape_b: &[usize],
        _dtype: &DataType,
        _a: &[f32],
        _b: &[f32],
    ) -> Result<Vec<f32>, String> {
        unimplemented!("Metal backend matmul is not yet implemented")
    }
}
