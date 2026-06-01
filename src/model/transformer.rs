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
use crate::model::block::TransformerBlock;
use crate::model::layers::{Embedding, Linear, RmsNorm};
use crate::model::moe::{Expert, MoE};
use crate::tensor::{DataType, Tensor};
use std::collections::HashMap;

/// The overall Transformer architecture.
#[derive(Debug)]
pub struct Transformer {
    pub token_embeddings: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
}

impl Transformer {
    pub fn new(
        token_embeddings: Embedding,
        layers: Vec<TransformerBlock>,
        norm: RmsNorm,
        lm_head: Linear,
    ) -> Self {
        Self {
            token_embeddings,
            layers,
            norm,
            lm_head,
        }
    }

    /// Constructs a Transformer model from loaded Safetensors weights.
    pub fn from_weights(mut weights: HashMap<String, Tensor>) -> Result<Self, String> {
        // 1. Embeddings
        let embed_weight = weights
            .remove("model.embed_tokens.weight")
            .or_else(|| weights.remove("embed_tokens.weight"))
            .ok_reachable("model.embed_tokens.weight tensor not found")?;
        let token_embeddings = Embedding::new(embed_weight);

        // 2. Count layers
        let mut layer_idx = 0;
        let mut layers = Vec::new();

        loop {
            let input_ln_key = format!("model.layers.{}.input_layernorm.weight", layer_idx);
            if !weights.contains_key(&input_ln_key) {
                break;
            }

            // Load layernorms
            let input_ln_weight = weights
                .remove(&input_ln_key)
                .ok_reachable("input_layernorm weight missing")?;
            let input_layernorm = RmsNorm::new(input_ln_weight, 1e-5);

            let post_attn_ln_key = format!(
                "model.layers.{}.post_attention_layernorm.weight",
                layer_idx
            );
            let post_attn_ln_weight = weights
                .remove(&post_attn_ln_key)
                .ok_reachable("post_attention_layernorm weight missing")?;
            let post_attention_layernorm = RmsNorm::new(post_attn_ln_weight, 1e-5);

            // Load Self Attention
            let q_proj_weight = weights
                .remove(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx))
                .ok_reachable("q_proj weight missing")?;
            let k_proj_weight = weights
                .remove(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx))
                .ok_reachable("k_proj weight missing")?;
            let v_proj_weight = weights
                .remove(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx))
                .ok_reachable("v_proj weight missing")?;
            let o_proj_weight = weights
                .remove(&format!("model.layers.{}.self_attn.o_proj.weight", layer_idx))
                .ok_reachable("o_proj weight missing")?;

            // Transpose linear weights from safetensors format [out_features, in_features] to our internal format [in_features, out_features]
            let q_proj_weight = transpose_2d(&q_proj_weight)?;
            let k_proj_weight = transpose_2d(&k_proj_weight)?;
            let v_proj_weight = transpose_2d(&v_proj_weight)?;
            let o_proj_weight = transpose_2d(&o_proj_weight)?;

            // Estimate parameters for Attention
            let hidden_size = q_proj_weight.shape[0]; // q_proj is now [in_features, out_features]
            let num_heads = 32; // Default fallback estimate
            let head_dim = hidden_size / num_heads;
            let num_kv_heads = k_proj_weight.shape[1] / head_dim;

            let attention = Attention::new(
                Linear::new(q_proj_weight, None),
                Linear::new(k_proj_weight, None),
                Linear::new(v_proj_weight, None),
                Linear::new(o_proj_weight, None),
                num_heads,
                num_kv_heads,
                head_dim,
            );

            // Load MLP / MoE
            let moe_gate_key = format!("model.layers.{}.block_sparse_moe.gate.weight", layer_idx);
            let moe = if weights.contains_key(&moe_gate_key) {
                // Real MoE
                let gate_weight = weights
                    .remove(&moe_gate_key)
                    .ok_reachable("gate weight missing")?;
                let gate = Linear::new(transpose_2d(&gate_weight)?, None);

                let mut experts = Vec::new();
                let mut expert_idx = 0;
                loop {
                    let w1_key = format!(
                        "model.layers.{}.block_sparse_moe.experts.{}.w1.weight",
                        layer_idx, expert_idx
                    );
                    if !weights.contains_key(&w1_key) {
                        break;
                    }
                    let w1 = Linear::new(transpose_2d(&weights.remove(&w1_key).unwrap())?, None);
                    let w2 = Linear::new(
                        transpose_2d(
                            &weights
                                .remove(&format!(
                                    "model.layers.{}.block_sparse_moe.experts.{}.w2.weight",
                                    layer_idx, expert_idx
                                ))
                                .unwrap(),
                        )?,
                        None,
                    );
                    let w3 = Linear::new(
                        transpose_2d(
                            &weights
                                .remove(&format!(
                                    "model.layers.{}.block_sparse_moe.experts.{}.w3.weight",
                                    layer_idx, expert_idx
                                ))
                                .unwrap(),
                        )?,
                        None,
                    );

                    experts.push(Expert::new(w1, w2, w3));
                    expert_idx += 1;
                }

                MoE::new(gate, experts, 2) // Default top_k = 2 for Mixtral
            } else {
                // Standard MLP mapped to single-expert MoE
                let gate_proj_weight = weights
                    .remove(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))
                    .ok_reachable("mlp.gate_proj weight missing")?;
                let down_proj_weight = weights
                    .remove(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx))
                    .ok_reachable("mlp.down_proj weight missing")?;
                let up_proj_weight = weights
                    .remove(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx))
                    .ok_reachable("mlp.up_proj weight missing")?;

                let expert = Expert::new(
                    Linear::new(transpose_2d(&gate_proj_weight)?, None),
                    Linear::new(transpose_2d(&down_proj_weight)?, None),
                    Linear::new(transpose_2d(&up_proj_weight)?, None),
                );

                // Dummy gate with zeros, top_k = 1
                let dummy_gate_weight = Tensor::new(vec![hidden_size, 1], DataType::Float32)?;
                let gate = Linear::new(dummy_gate_weight, None);

                MoE::new(gate, vec![expert], 1)
            };

            layers.push(TransformerBlock::new(
                input_layernorm,
                attention,
                post_attention_layernorm,
                moe,
            ));

            layer_idx += 1;
        }

        // 3. Final normalization
        let norm_weight = weights
            .remove("model.norm.weight")
            .or_else(|| weights.remove("norm.weight"))
            .ok_reachable("model.norm.weight tensor not found")?;
        let norm = RmsNorm::new(norm_weight, 1e-5);

        // 4. LM Head
        let lm_head_weight = weights
            .remove("lm_head.weight")
            .ok_reachable("lm_head.weight tensor not found")?;
        let lm_head = Linear::new(transpose_2d(&lm_head_weight)?, None);

        Ok(Self::new(token_embeddings, layers, norm, lm_head))
    }

    /// Forward pass through the Transformer model.
    /// Returns the logits for the last token.
    pub fn forward(&self, tokens: &[u32]) -> Result<Tensor, String> {
        if tokens.is_empty() {
            return Err("Cannot run forward pass on empty token list".to_string());
        }

        // 1. Retrieve initial embeddings
        let mut x = self.token_embeddings.forward(tokens)?;

        // 2. Pass through each TransformerBlock layer
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // 3. Final RMSNorm
        x = self.norm.forward(&x)?;

        // 4. Extract last token representation to compute next-token logits
        let seq_len = x.shape[0];
        let hidden_size = x.shape[1];

        let mut x_last = Tensor::new(vec![1, hidden_size], x.dtype.clone())?;
        let offset = (seq_len - 1) * hidden_size;
        x_last.data.copy_from_slice(&x.data[offset..offset + hidden_size]);

        // 5. Predict logits via LM Head
        let logits = self.lm_head.forward(&x_last)?;
        Ok(logits)
    }

    /// Moves all model weights to the specified device.
    pub fn to_device(&mut self, device: crate::tensor::backend::Device) {
        self.token_embeddings.weight.to_device(device);
        for layer in &mut self.layers {
            layer.input_layernorm.weight.to_device(device);
            layer.attention.q_proj.weight.to_device(device);
            layer.attention.k_proj.weight.to_device(device);
            layer.attention.v_proj.weight.to_device(device);
            layer.attention.o_proj.weight.to_device(device);
            layer.post_attention_layernorm.weight.to_device(device);
            layer.moe.gate.weight.to_device(device);
            for expert in &mut layer.moe.experts {
                expert.w1.weight.to_device(device);
                expert.w2.weight.to_device(device);
                expert.w3.weight.to_device(device);
            }
        }
        self.norm.weight.to_device(device);
        self.lm_head.weight.to_device(device);
    }

    /// Generates new tokens from a prompt using greedy sampling.
    pub fn generate<F>(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        stop_token_id: Option<u32>,
        mut on_token: F,
    ) -> Result<Vec<u32>, String>
    where
        F: FnMut(u32),
    {
        let mut generated = Vec::new();
        let mut context = prompt_tokens.to_vec();

        for _ in 0..max_tokens {
            let logits = self.forward(&context)?;

            // Greedy sampling: find index of maximum logit
            let mut max_idx = 0;
            let mut max_val = logits.data[0];
            for idx in 1..logits.data.len() {
                if logits.data[idx] > max_val {
                    max_val = logits.data[idx];
                    max_idx = idx;
                }
            }

            let next_token = max_idx as u32;
            generated.push(next_token);
            on_token(next_token);

            if Some(next_token) == stop_token_id {
                break;
            }

            context.push(next_token);
        }

        Ok(generated)
    }
}

/// Helper extension trait for Option mapping
trait OptionExt<T> {
    fn ok_reachable(self, err_msg: &str) -> Result<T, String>;
}

impl<T> OptionExt<T> for Option<T> {
    fn ok_reachable(self, err_msg: &str) -> Result<T, String> {
        self.ok_or_else(|| err_msg.to_string())
    }
}

fn transpose_2d(t: &Tensor) -> Result<Tensor, String> {
    if t.shape.len() != 2 {
        return Err("Only 2D tensors can be transposed".to_string());
    }
    let rows = t.shape[0];
    let cols = t.shape[1];
    let mut transposed = Tensor::new(vec![cols, rows], t.dtype.clone())?;
    for i in 0..rows {
        for j in 0..cols {
            transposed.data[j * rows + i] = t.data[i * cols + j];
        }
    }
    Ok(transposed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DataType;

    #[test]
    fn test_transformer_compiles_and_runs() {
        // Create a mini 1-layer dummy model
        let embed = Embedding::new(Tensor::new(vec![10, 4], DataType::Float32).unwrap());

        let input_ln = RmsNorm::new(Tensor::new(vec![4], DataType::Float32).unwrap(), 1e-5);
        let post_ln = RmsNorm::new(Tensor::new(vec![4], DataType::Float32).unwrap(), 1e-5);

        let attn = Attention::new(
            Linear::new(Tensor::new(vec![4, 4], DataType::Float32).unwrap(), None),
            Linear::new(Tensor::new(vec![4, 4], DataType::Float32).unwrap(), None),
            Linear::new(Tensor::new(vec![4, 4], DataType::Float32).unwrap(), None),
            Linear::new(Tensor::new(vec![4, 4], DataType::Float32).unwrap(), None),
            2,
            2,
            2,
        );

        let expert = Expert::new(
            Linear::new(Tensor::new(vec![4, 8], DataType::Float32).unwrap(), None),
            Linear::new(Tensor::new(vec![8, 4], DataType::Float32).unwrap(), None),
            Linear::new(Tensor::new(vec![4, 8], DataType::Float32).unwrap(), None),
        );
        let moe = MoE::new(
            Linear::new(Tensor::new(vec![4, 1], DataType::Float32).unwrap(), None),
            vec![expert],
            1,
        );

        let block = TransformerBlock::new(input_ln, attn, post_ln, moe);

        let norm = RmsNorm::new(Tensor::new(vec![4], DataType::Float32).unwrap(), 1e-5);
        let lm_head = Linear::new(Tensor::new(vec![4, 10], DataType::Float32).unwrap(), None);

        let transformer = Transformer::new(embed, vec![block], norm, lm_head);

        let prompt = vec![1, 2, 3];
        let logits = transformer.forward(&prompt).unwrap();
        assert_eq!(logits.shape, vec![1, 10]);
    }
}
