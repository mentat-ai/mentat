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

/// An individual Expert network.
/// Typically, this is a multi-layer perceptron (Feed-Forward Network)
/// using SwiGLU or Gelu activations.
#[derive(Debug)]
pub struct Expert {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

impl Expert {
    pub fn new(w1: Linear, w2: Linear, w3: Linear) -> Self {
        Self { w1, w2, w3 }
    }

    /// Forward pass through the expert.
    /// Standard SwiGLU FFN: output = (Swish(x * w1) * (x * w3)) * w2
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        let mut h1 = self.w1.forward(x)?;
        let h3 = self.w3.forward(x)?;

        // Apply Swish/SiLU on h1 and multiply by h3: (x * w1) * sigmoid(x * w1) * (x * w3)
        for i in 0..h1.data.len() {
            let val = h1.data[i];
            let sigmoid = 1.0 / (1.0 + (-val).exp());
            h1.data[i] = val * sigmoid * h3.data[i];
        }

        // Output projection
        let output = self.w2.forward(&h1)?;
        Ok(output)
    }
}

/// Mixture of Experts (MoE) block.
/// Replaces the standard Feed-Forward Network in a Transformer block.
#[derive(Debug)]
pub struct MoE {
    /// The router linear layer that outputs logits for each expert.
    pub gate: Linear,
    /// The pool of experts.
    pub experts: Vec<Expert>,
    /// Number of experts to select per token (e.g., top-2).
    pub top_k: usize,
}

impl MoE {
    pub fn new(gate: Linear, experts: Vec<Expert>, top_k: usize) -> Self {
        Self {
            gate,
            experts,
            top_k,
        }
    }

    /// Forward pass for the MoE block.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        let seq_len = x.shape[0];
        let hidden_size = x.shape[1];
        let num_experts = self.experts.len();

        let mut output = Tensor::new(vec![seq_len, hidden_size], x.dtype.clone())?;
        if num_experts == 0 {
            return Ok(output);
        }

        // 1. Calculate routing logits: shape [seq_len, num_experts]
        let logits = self.gate.forward(x)?;

        // 2. Process each token individually
        for i in 0..seq_len {
            let logit_offset = i * num_experts;
            let mut probs = vec![0.0; num_experts];

            // Softmax over experts for token i
            let mut max_logit = logits.data[logit_offset];
            for e in 1..num_experts {
                let l = logits.data[logit_offset + e];
                if l > max_logit {
                    max_logit = l;
                }
            }

            let mut sum_exp = 0.0;
            for e in 0..num_experts {
                let p = (logits.data[logit_offset + e] - max_logit).exp();
                probs[e] = p;
                sum_exp += p;
            }

            if sum_exp > 0.0 {
                for e in 0..num_experts {
                    probs[e] /= sum_exp;
                }
            }

            // Find top-k experts
            let mut expert_probs: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
            expert_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_k_experts = &expert_probs[0..self.top_k.min(num_experts)];

            // Renormalize top-k probabilities
            let top_k_sum: f32 = top_k_experts.iter().map(|(_, p)| p).sum();

            // Prepare token input tensor of shape [1, hidden_size]
            let mut token_tensor = Tensor::new(vec![1, hidden_size], x.dtype.clone())?;
            let token_offset = i * hidden_size;
            token_tensor.data.copy_from_slice(&x.data[token_offset..token_offset + hidden_size]);

            // Route to selected experts and sum outputs
            for &(expert_idx, prob) in top_k_experts {
                if top_k_sum > 0.0 {
                    let weight = prob / top_k_sum;
                    let expert = &self.experts[expert_idx];
                    let expert_out = expert.forward(&token_tensor)?;

                    // Accumulate scaled output
                    let out_offset = i * hidden_size;
                    for d in 0..hidden_size {
                        output.data[out_offset + d] += expert_out.data[d] * weight;
                    }
                }
            }
        }

        Ok(output)
    }
}
