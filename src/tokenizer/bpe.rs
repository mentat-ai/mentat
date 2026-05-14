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

use std::collections::HashMap;

/// A high-performance Byte Pair Encoding (BPE) tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Maps a string token to its ID.
    pub vocab: HashMap<String, u32>,
    /// Maps an ID to its string representation (reverse of vocab).
    pub id_to_token: HashMap<u32, String>,
    /// Defines the merge rules: (token_id_1, token_id_2) -> merged_token_id.
    pub merges: HashMap<(u32, u32), u32>,
    /// Special tokens for agentic interactions and reasoning (e.g., "<think>").
    pub special_tokens: HashMap<String, u32>,
}

impl BpeTokenizer {
    /// Creates a new, empty BPE tokenizer.
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: HashMap::new(),
            special_tokens: HashMap::new(),
        }
    }

    /// Adds a special token to the vocabulary.
    pub fn add_special_token(&mut self, token: &str, id: u32) {
        self.special_tokens.insert(token.to_string(), id);
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
    }

    /// Encodes a raw string into a list of token IDs.
    /// This is a simplified implementation of the BPE algorithm.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Handle special tokens very naively for the initial implementation.
        // A production version would use a regex or Aho-Corasick automaton to extract them.

        let mut tokens = Vec::new();

        // For this baseline, we treat each char as a base token if not merging.
        // In real BPE, this starts as raw bytes.
        let mut current_ids: Vec<u32> = text
            .chars()
            .map(|c| {
                let s = c.to_string();
                *self.vocab.get(&s).unwrap_or(&0) // 0 as UNK token for now
            })
            .collect();

        // Apply merges iteratively
        loop {
            if current_ids.len() < 2 {
                break;
            }

            // Find the best pair to merge. In real BPE, you'd track ranks.
            // Here, we just find the first available merge.
            let mut merged_once = false;
            let mut i = 0;
            let mut next_ids = Vec::new();

            while i < current_ids.len() {
                if i + 1 < current_ids.len() {
                    let pair = (current_ids[i], current_ids[i + 1]);
                    if let Some(&merged_id) = self.merges.get(&pair) {
                        next_ids.push(merged_id);
                        i += 2;
                        merged_once = true;
                        continue;
                    }
                }
                next_ids.push(current_ids[i]);
                i += 1;
            }

            current_ids = next_ids;
            if !merged_once {
                break;
            }
        }

        tokens.extend(current_ids);
        tokens
    }

    /// Decodes a list of token IDs back into a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for id in ids {
            if let Some(token) = self.id_to_token.get(id) {
                text.push_str(token);
            }
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_basic_encode_decode() {
        let mut tokenizer = BpeTokenizer::new();

        // Seed basic chars
        tokenizer.vocab.insert("h".to_string(), 1);
        tokenizer.id_to_token.insert(1, "h".to_string());
        tokenizer.vocab.insert("e".to_string(), 2);
        tokenizer.id_to_token.insert(2, "e".to_string());
        tokenizer.vocab.insert("l".to_string(), 3);
        tokenizer.id_to_token.insert(3, "l".to_string());
        tokenizer.vocab.insert("o".to_string(), 4);
        tokenizer.id_to_token.insert(4, "o".to_string());

        // Seed merged tokens
        tokenizer.vocab.insert("he".to_string(), 5);
        tokenizer.id_to_token.insert(5, "he".to_string());
        tokenizer.merges.insert((1, 2), 5); // 'h' + 'e' -> 'he'

        tokenizer.vocab.insert("ll".to_string(), 6);
        tokenizer.id_to_token.insert(6, "ll".to_string());
        tokenizer.merges.insert((3, 3), 6); // 'l' + 'l' -> 'll'

        let encoded = tokenizer.encode("hello");
        // Expected: 'he' (5), 'll' (6), 'o' (4)
        assert_eq!(encoded, vec![5, 6, 4]);

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_special_tokens() {
        let mut tokenizer = BpeTokenizer::new();
        tokenizer.add_special_token("<think>", 100);
        tokenizer.add_special_token("</think>", 101);

        assert_eq!(tokenizer.vocab.get("<think>"), Some(&100));
        assert_eq!(tokenizer.decode(&[100, 101]), "<think></think>");
    }
}
