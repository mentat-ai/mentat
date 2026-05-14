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

use crate::tensor::Tensor;

/// A standard fully connected Linear layer: y = xA^T + b
#[derive(Debug)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    /// Forward pass for the Linear layer.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        // x shape: [batch_size, in_features]
        // weight shape: [out_features, in_features]
        // We need x * weight^T to get [batch_size, out_features]
        // For simplicity in our current Tensor implementation, let's assume
        // we have a matmul that handles this, or we expect weights already transposed.

        // Assuming weights are stored as [in_features, out_features] for our basic matmul
        let mut output = x.matmul(&self.weight)?;

        if let Some(b) = &self.bias {
            // Broadcasting addition: output + bias
            // Since our current Add doesn't support broadcasting, we simulate it for now
            // by adding the bias directly to each row.
            let out_cols = output.shape[1];
            for i in 0..output.shape[0] {
                for j in 0..out_cols {
                    output.data[i * out_cols + j] += b.data[j];
                }
            }
        }

        Ok(output)
    }
}

/// Root Mean Square Normalization (RMSNorm)
/// Commonly used in modern architectures like Llama and GPT-OSS instead of LayerNorm.
#[derive(Debug)]
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward pass for RMSNorm.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        let mut output = Tensor::new(x.shape.clone(), x.dtype.clone())?;

        let batch_size = x.shape[0];
        let hidden_size = x.shape[1];

        for i in 0..batch_size {
            let row_start = i * hidden_size;
            let row_end = row_start + hidden_size;
            let row = &x.data[row_start..row_end];

            // Calculate RMS
            let mut sum_sq: f32 = 0.0;
            for &val in row {
                sum_sq += val * val;
            }
            let rms = ((sum_sq / hidden_size as f32) + self.eps).sqrt();

            // Normalize and scale by weight
            for j in 0..hidden_size {
                output.data[row_start + j] = (row[j] / rms) * self.weight.data[j];
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DataType;

    #[test]
    fn test_linear_forward() {
        let mut x = Tensor::new(vec![1, 2], DataType::Float32).unwrap();
        x.data = vec![1.0, 2.0];

        let mut w = Tensor::new(vec![2, 2], DataType::Float32).unwrap();
        w.data = vec![1.0, 2.0, 3.0, 4.0];

        let mut b = Tensor::new(vec![2], DataType::Float32).unwrap();
        b.data = vec![0.5, 0.5];

        let linear = Linear::new(w, Some(b));
        let result = linear.forward(&x).unwrap();

        // [1, 2] * [[1, 2], [3, 4]] = [7, 10]
        // [7, 10] + [0.5, 0.5] = [7.5, 10.5]
        assert_eq!(result.data, vec![7.5, 10.5]);
    }
}
