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

/// A single entry in the KV Cache for one layer.
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    pub k: Tensor,
    pub v: Tensor,
}

/// KV Cache stores Key and Value tensors for all previous tokens across all layers.
/// This avoids re-calculating the representations of past tokens during autoregressive generation.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Vector of cache entries, one per Transformer layer.
    pub layers: Vec<Option<KVCacheEntry>>,
    /// Maximum sequence length supported by the cache.
    pub max_seq_len: usize,
    /// Current position (number of tokens already in cache).
    pub current_pos: usize,
}

impl KVCache {
    /// Creates a new KV Cache for a model with `num_layers`.
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
            max_seq_len,
            current_pos: 0,
        }
    }

    /// Updates the cache for a specific layer with new K and V tensors.
    pub fn update(&mut self, layer_idx: usize, new_k: Tensor, new_v: Tensor) -> Result<(), String> {
        if layer_idx >= self.layers.len() {
            return Err(format!("layer index {} out of bounds", layer_idx));
        }

        // In a real implementation, we would append to existing tensors or 
        // write into a pre-allocated buffer at `self.current_pos`.
        // For now, we store them as entries.
        self.layers[layer_idx] = Some(KVCacheEntry { k: new_k, v: new_v });
        
        Ok(())
    }

    /// Resets the cache (e.g., for a new prompt).
    pub fn clear(&mut self) {
        for entry in self.layers.iter_mut() {
            *entry = None;
        }
        self.current_pos = 0;
    }
}
