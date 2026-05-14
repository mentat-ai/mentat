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

use crate::model::attention::Attention;
use crate::model::layers::RmsNorm;
use crate::model::moe::MoE;
use crate::tensor::Tensor;

/// A single Transformer Block combining Attention and Mixture of Experts (MoE).
/// Modern architectures use Pre-Normalization, meaning RMSNorm is applied
/// before the Attention and MoE blocks, and the original input is added back
/// via residual connections.
#[derive(Debug)]
pub struct TransformerBlock {
    pub input_layernorm: RmsNorm,
    pub attention: Attention,
    pub post_attention_layernorm: RmsNorm,
    pub moe: MoE,
}

impl TransformerBlock {
    pub fn new(
        input_layernorm: RmsNorm,
        attention: Attention,
        post_attention_layernorm: RmsNorm,
        moe: MoE,
    ) -> Self {
        Self {
            input_layernorm,
            attention,
            post_attention_layernorm,
            moe,
        }
    }

    /// Forward pass through the Transformer block.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        // 1. Attention path with Pre-Norm
        // norm_x = RMSNorm(x)
        let norm_x = self.input_layernorm.forward(x)?;

        // attn_out = Attention(norm_x)
        let attn_out = self.attention.forward(&norm_x)?;

        // Residual Connection: h = x + attn_out
        let h = x.add(&attn_out)?;

        // 2. MoE path with Pre-Norm
        // norm_h = RMSNorm(h)
        let norm_h = self.post_attention_layernorm.forward(&h)?;

        // moe_out = MoE(norm_h)
        let moe_out = self.moe.forward(&norm_h)?;

        // Residual Connection: out = h + moe_out
        let out = h.add(&moe_out)?;

        Ok(out)
    }
}
