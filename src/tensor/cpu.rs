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
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn add(&self, _shape: &[usize], _dtype: &DataType, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        Ok(a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x + y).collect())
    }

    fn mul(&self, _shape: &[usize], _dtype: &DataType, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        Ok(a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x * y).collect())
    }

    fn matmul(
        &self,
        shape_a: &[usize],
        shape_b: &[usize],
        _dtype: &DataType,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String> {
        let rows_a = shape_a[0];
        let cols_a = shape_a[1];
        let cols_b = shape_b[1];

        Ok((0..rows_a)
            .into_par_iter()
            .flat_map(|i| {
                let mut row_result = vec![0.0; cols_b];
                for j in 0..cols_b {
                    let mut sum: f32 = 0.0;
                    for k in 0..cols_a {
                        sum += a[i * cols_a + k] * b[k * cols_b + j];
                    }
                    row_result[j] = sum;
                }
                row_result
            })
            .collect())
    }
}
