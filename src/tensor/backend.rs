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

use crate::tensor::DataType;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
    #[cfg(feature = "metal")]
    Metal(usize),
}

pub trait Backend: Debug {
    fn device(&self) -> Device;
    fn add(&self, shape: &[usize], dtype: &DataType, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String>;
    fn mul(&self, shape: &[usize], dtype: &DataType, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String>;
    fn matmul(
        &self,
        shape_a: &[usize],
        shape_b: &[usize],
        dtype: &DataType,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String>;
}
