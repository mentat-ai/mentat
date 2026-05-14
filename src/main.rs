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

use clap::Parser;
use tracing::{debug, error, info, Level};
use tracing_subscriber::FmtSubscriber;

use mentat::config::{Commands, Config};
use mentat::model::loader::Loader;
use mentat::tokenizer::bpe::BpeTokenizer;
use mentat::tokenizer::parser::HarmonyParser;

fn main() {
    let config = Config::parse();

    let log_level = if config.debug {
        Level::DEBUG
    } else {
        Level::INFO
    };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    info!("Mentat Inference Engine starting...");

    match &config.command {
        Commands::Run { model, prompt } => {
            info!("Initializing 'run' mode");
            debug!(
                "Configuration loaded - model: {}, prompt: {:?}",
                model, prompt
            );
            info!("Starting inference (Not implemented yet)");
        }
        Commands::Serve { port } => {
            info!("Initializing 'serve' mode");
            info!("Starting API server on port {} (Not implemented yet)", port);
        }
        Commands::Tokenize { text } => {
            info!("Initializing 'tokenize' test mode");
            let mut tokenizer = BpeTokenizer::new();

            // Dynamically seed base vocabulary with all unique characters from the input
            // so we don't get 'UNK' (0) for characters like 'a', 'c', ' ', etc.
            let mut current_max_id = 1;
            for c in text.chars() {
                let s = c.to_string();
                if !tokenizer.vocab.contains_key(&s) {
                    tokenizer.vocab.insert(s.clone(), current_max_id);
                    tokenizer.id_to_token.insert(current_max_id, s);
                    current_max_id += 1;
                }
            }

            // Seed a few artificial merge rules for demonstration
            // Only add them if the characters exist in our dynamic vocab
            if let (Some(&h), Some(&e)) = (tokenizer.vocab.get("h"), tokenizer.vocab.get("e")) {
                tokenizer.vocab.insert("he".to_string(), current_max_id);
                tokenizer
                    .id_to_token
                    .insert(current_max_id, "he".to_string());
                tokenizer.merges.insert((h, e), current_max_id);
                current_max_id += 1;
            }

            if let Some(&l) = tokenizer.vocab.get("l") {
                tokenizer.vocab.insert("ll".to_string(), current_max_id);
                tokenizer
                    .id_to_token
                    .insert(current_max_id, "ll".to_string());
                tokenizer.merges.insert((l, l), current_max_id);
            }

            // Add special harmony token
            tokenizer.add_special_token("<think>", 100);

            println!("\n--- Tokenizer Interactive Test ---");
            println!("Input Text: '{}'", text);

            let encoded = tokenizer.encode(text);
            println!("Encoded IDs: {:?}", encoded);

            let decoded = tokenizer.decode(&encoded);
            println!("Decoded Text: '{}'", decoded);
            println!("----------------------------------");
        }
        Commands::Parse { text } => {
            info!("Initializing 'parse' test mode");
            println!("\n--- Parser Interactive Test ---");
            println!("Input Text:\n{}\n", text);

            let blocks = HarmonyParser::parse(text);

            for (i, block) in blocks.iter().enumerate() {
                println!("Block {}: {:#?}", i + 1, block);
            }
            println!("----------------------------------");
        }
        Commands::Inspect { model } => {
            info!("Initializing 'inspect' mode");
            println!("\n--- Model Inspection: {} ---", model);

            match Loader::load_safetensors(model) {
                Ok(weights) => {
                    println!("Successfully loaded {} tensors.\n", weights.len());
                    println!("{:<50} | {:<20} | {:<10}", "Tensor Name", "Shape", "Data Type");
                    println!("{:-<50}-|-{:-<20}-|-{:-<10}", "", "", "");

                    let mut sorted_names: Vec<_> = weights.keys().collect();
                    sorted_names.sort();

                    for name in sorted_names {
                        let t = &weights[name];
                        println!("{:<50} | {:<20?} | {:?}", name, t.shape, t.dtype);
                    }
                }
                Err(e) => {
                    error!("Failed to inspect model: {}", e);
                }
            }
            println!("----------------------------------");
        }
    }
}

