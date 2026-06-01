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

use crate::model::layers::Linear;
use crate::tensor::Tensor;

/// Low-Rank Adaptation (LoRA) Linear Layer
/// Replaces W with W + (A * B) * (alpha / r)
pub struct LoraLinear {
    pub base: Linear,
    pub lora_a: Tensor, // shape: [in_features, r]
    pub lora_b: Tensor, // shape: [r, out_features]
    pub r: usize,
    pub alpha: f32,
}

impl LoraLinear {
    /// Creates a new LoraLinear by wrapping an existing Linear layer.
    pub fn new(base: Linear, r: usize, alpha: f32) -> Result<Self, String> {
        let in_features = base.weight.shape[0]; // weight is [in_features, out_features]
        let out_features = base.weight.shape[1];

        // In a real implementation, lora_a is initialized with random normal
        // and lora_b is initialized with zeros. We use dummy initialization here.
        let mut lora_a = Tensor::new(vec![in_features, r], base.weight.dtype.clone())?;
        lora_a.data.fill(0.01);
        
        let mut lora_b = Tensor::new(vec![r, out_features], base.weight.dtype.clone())?;
        lora_b.data.fill(0.0);

        Ok(Self {
            base,
            lora_a,
            lora_b,
            r,
            alpha,
        })
    }

    /// Forward pass: Y = X * W^T + X * A * B * (alpha/r)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        let base_out = self.base.forward(x)?;

        let scaling = self.alpha / (self.r as f32);

        // Compute X * A
        let xa = x.matmul(&self.lora_a)?;
        // Compute (X * A) * B
        let mut xab = xa.matmul(&self.lora_b)?;

        // Scale xab by (alpha/r)
        for val in xab.data.iter_mut() {
            *val *= scaling;
        }

        // Add to base output
        base_out.add(&xab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DataType;

    #[test]
    fn test_lora_linear_forward() {
        let mut weight = Tensor::new(vec![3, 4], DataType::Float32).unwrap();
        weight.data.fill(1.0);
        
        let base_layer = Linear {
            weight,
            bias: None,
        };

        let lora_layer = LoraLinear::new(base_layer, 2, 1.0).unwrap();
        
        let mut x = Tensor::new(vec![2, 3], DataType::Float32).unwrap();
        x.data.fill(1.0);

        let out = lora_layer.forward(&x).unwrap();
        assert_eq!(out.shape, vec![2, 4]);
    }
}
