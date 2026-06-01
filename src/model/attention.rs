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

/// Multi-Head Attention (MHA) or Grouped-Query Attention (GQA) base structure.
/// Modern models (like Llama 3 / GPT-OSS) often use GQA for efficiency,
/// but the forward pass signature remains similar.
#[derive(Debug)]
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        // 1. Project input to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let seq_len = x.shape[0];
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let group_size = if self.num_kv_heads > 0 { self.num_heads / self.num_kv_heads } else { 1 };

        if q.data.len() < seq_len * q_dim || k.data.len() < seq_len * kv_dim || v.data.len() < seq_len * kv_dim {
            return Err("Attention projection output shape mismatch with num_heads/num_kv_heads/head_dim parameters".to_string());
        }

        // Output tensor shape: [seq_len, q_dim]
        let mut attn_out = Tensor::new(vec![seq_len, q_dim], x.dtype.clone())?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // 2. Loop over each head
        for h in 0..self.num_heads {
            let kv_h = h / group_size;

            for i in 0..seq_len {
                // Compute attention scores for head h, query token i, key token j
                let mut scores = vec![0.0; seq_len];
                let q_offset = i * q_dim + h * self.head_dim;

                for j in 0..seq_len {
                    if j > i {
                        scores[j] = f32::NEG_INFINITY;
                    } else {
                        let mut sum = 0.0;
                        let k_offset = j * kv_dim + kv_h * self.head_dim;
                        for d in 0..self.head_dim {
                            sum += q.data[q_offset + d] * k.data[k_offset + d];
                        }
                        scores[j] = sum * scale;
                    }
                }

                // Softmax over j
                let mut max_val = scores[0];
                for j in 1..seq_len {
                    if scores[j] > max_val {
                        max_val = scores[j];
                    }
                }

                let mut sum_exp = 0.0;
                for j in 0..seq_len {
                    let val = (scores[j] - max_val).exp();
                    scores[j] = val;
                    sum_exp += val;
                }

                if sum_exp > 0.0 {
                    for j in 0..seq_len {
                        scores[j] /= sum_exp;
                    }
                }

                // Weighted sum of V
                let out_offset = i * q_dim + h * self.head_dim;
                for d in 0..self.head_dim {
                    let mut sum = 0.0;
                    for j in 0..seq_len {
                        let v_offset = j * kv_dim + kv_h * self.head_dim;
                        sum += scores[j] * v.data[v_offset + d];
                    }
                    attn_out.data[out_offset + d] = sum;
                }
            }
        }

        // 3. Final output projection
        let out = self.o_proj.forward(&attn_out)?;

        Ok(out)
    }
}
