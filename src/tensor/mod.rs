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

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Float32,
    Float16,
    Int32,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub data: Vec<f32>, // Using f32 as the base for now
}

impl Tensor {
    pub fn new(shape: Vec<usize>, dtype: DataType) -> Result<Self, String> {
        if dtype != DataType::Float32 {
            return Err("currently only Float32 is supported".to_string());
        }

        let size: usize = shape.iter().product();
        Ok(Self {
            shape,
            dtype,
            data: vec![0.0; size],
        })
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err(format!(
                "shape mismatch: {:?} and {:?}",
                self.shape, other.shape
            ));
        }

        let mut result = Tensor::new(self.shape.clone(), self.dtype.clone())?;
        for (i, (&a, &b)) in self.data.iter().zip(other.data.iter()).enumerate() {
            result.data[i] = a + b;
        }
        Ok(result)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err(format!(
                "shape mismatch: {:?} and {:?}",
                self.shape, other.shape
            ));
        }

        let mut result = Tensor::new(self.shape.clone(), self.dtype.clone())?;
        for (i, (&a, &b)) in self.data.iter().zip(other.data.iter()).enumerate() {
            result.data[i] = a * b;
        }
        Ok(result)
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err("MatMul currently only supports 2D tensors".to_string());
        }
        if self.shape[1] != other.shape[0] {
            return Err(format!(
                "incompatible dimensions for MatMul: {:?} and {:?}",
                self.shape, other.shape
            ));
        }

        let rows_a = self.shape[0];
        let cols_a = self.shape[1];
        let cols_b = other.shape[1];

        let mut result = Tensor::new(vec![rows_a, cols_b], self.dtype.clone())?;

        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum: f32 = 0.0;
                for k in 0..cols_a {
                    sum += self.data[i * cols_a + k] * other.data[k * cols_b + j];
                }
                result.data[i * cols_b + j] = sum;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let mut a = Tensor::new(vec![2, 2], DataType::Float32).unwrap();
        let mut b = Tensor::new(vec![2, 2], DataType::Float32).unwrap();

        a.data = vec![1.0, 2.0, 3.0, 4.0];
        b.data = vec![5.0, 6.0, 7.0, 8.0];

        let result = a.add(&b).unwrap();
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matmul() {
        let mut a = Tensor::new(vec![2, 3], DataType::Float32).unwrap();
        let mut b = Tensor::new(vec![3, 2], DataType::Float32).unwrap();

        a.data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        b.data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }
}
