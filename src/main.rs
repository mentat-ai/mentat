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
use tracing::{Level, debug, error, info};
use tracing_subscriber::FmtSubscriber;
use std::path::Path;
use std::fs;

use mentat::config::{Commands, Config};
use mentat::model::loader::Loader;
use mentat::model::transformer::Transformer;
use mentat::tokenizer::bpe::BpeTokenizer;
use mentat::tokenizer::parser::{HarmonyParser, ParsedBlock};
use mentat::tools::{Tool, browser::BrowserTool, fs::FilePatcherTool, python::PythonTool};
use mentat::telemetry::DataCollector;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
#[tokio::main]
async fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let config = Config::parse();
    
    let collector = DataCollector::new(config.opt_in_data_collection);

    let log_level = if config.debug {
        Level::DEBUG
    } else {
        Level::INFO
    };
    let subscriber = FmtSubscriber::builder().with_max_level(log_level).finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    info!("Mentat Inference Engine starting...");

    match &config.command {
        Commands::Run { model, prompt } => {
            info!("Initializing 'run' mode");
            debug!(
                "Configuration loaded - model: {}, prompt: {:?}",
                model, prompt
            );

            if model == "mock" {
                run_mock_chat(prompt.as_deref());
            } else {
                let path = Path::new(model);
                if !path.exists() {
                    error!("Model file '{}' does not exist.", model);
                    println!("Error: Model file '{}' does not exist. Use --model mock for testing without weights.", model);
                    std::process::exit(1);
                }

                info!("Loading model weights from {}...", model);
                let start_time = std::time::Instant::now();
                let weights = match Loader::load_safetensors(path) {
                    Ok(w) => w,
                    Err(e) => {
                        error!("Failed to load safetensors: {}", e);
                        std::process::exit(1);
                    }
                };
                info!("Loaded model weights in {:?}", start_time.elapsed());

                info!("Building Transformer computation graph...");
                let mut transformer = match Transformer::from_weights(weights) {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Failed to build Transformer: {}", e);
                        std::process::exit(1);
                    }
                };
                info!("Transformer initialized successfully.");

                let device = default_device();
                if device != mentat::tensor::backend::Device::Cpu {
                    info!("Moving model weights to GPU device: {:?}", device);
                    transformer.to_device(device);
                }

                let tokenizer = load_or_create_tokenizer(model);

                if let Some(user_prompt) = prompt {
                    println!("\nGenerating response for prompt: {}\n", user_prompt);
                    let formatted_prompt = format_llama2_chat(&[], user_prompt);
                    let prompt_tokens = tokenizer.encode(&formatted_prompt);
                    let _res = transformer.generate(
                        &prompt_tokens,
                        100,
                        tokenizer.vocab.get("</s>").copied().or(Some(257)),
                        |token_id| {
                            if let Some(token) = tokenizer.id_to_token.get(&token_id) {
                                print!("{}", clean_token(token));
                            } else {
                                print!("[{}]", token_id);
                            }
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                        },
                    );
                    println!("\n");
                } else {
                    run_real_chat(&transformer, &tokenizer);
                }
            }
        }
        Commands::Serve { port } => {
            info!("Initializing 'serve' mode");
            if let Err(e) = mentat::api::start_server(*port).await {
                error!("API server failed: {}", e);
            }
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
            
            collector.record_interaction(text, &decoded);
        }
        Commands::Parse { text } => {
            info!("Initializing 'parse' test mode");
            println!("\n--- Parser Interactive Test ---");
            println!("Input Text:\n{}\n", text);

            let blocks = HarmonyParser::parse(text);

            for (i, block) in blocks.iter().enumerate() {
                println!("Block {}: {:#?}", i + 1, block);

                // If the block is a python tool call, execute it!
                if let ParsedBlock::ToolCall {
                    tool_name,
                    arguments,
                } = block
                {
                    if tool_name == "python" {
                        println!(">> ⚙️ Executing Python Tool...");
                        let python_tool = PythonTool;
                        match python_tool.execute(arguments) {
                            Ok(output) => {
                                println!(">> ✅ Output:\n{}", output);
                            }
                            Err(err) => {
                                println!(">> ❌ Error:\n{}", err);
                            }
                        }
                    } else if tool_name == "file_patcher" {
                        println!(">> 📂 Executing File Patcher Tool...");
                        let fs_tool = FilePatcherTool;
                        match fs_tool.execute(arguments) {
                            Ok(output) => {
                                println!(">> ✅ Output:\n{}", output);
                            }
                            Err(err) => {
                                println!(">> ❌ Error:\n{}", err);
                            }
                        }
                    } else if tool_name == "browser" {
                        println!(">> 🌐 Executing Browser Tool...");
                        let browser_tool = BrowserTool;
                        match browser_tool.execute(arguments) {
                            Ok(output) => {
                                println!(">> ✅ Output:\n{}", output);
                            }
                            Err(err) => {
                                println!(">> ❌ Error:\n{}", err);
                            }
                        }
                    }
                }
            }
            println!("----------------------------------");
        }
        Commands::Inspect { model } => {
            info!("Initializing 'inspect' mode");
            println!("\n--- Model Inspection: {} ---", model);

            match Loader::load_safetensors(model) {
                Ok(weights) => {
                    println!("Successfully loaded {} tensors.\n", weights.len());
                    println!(
                        "{:<50} | {:<20} | {:<10}",
                        "Tensor Name", "Shape", "Data Type"
                    );
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

fn load_or_create_tokenizer(model_path: &str) -> BpeTokenizer {
    let mut tokenizer = BpeTokenizer::new();
    
    // Check if tokenizer.json is next to the model file
    let model_dir = Path::new(model_path).parent().unwrap_or_else(|| Path::new("."));
    let tokenizer_path = model_dir.join("tokenizer.json");
    
    let mut loaded = false;
    if tokenizer_path.exists() {
        if let Ok(content) = fs::read_to_string(&tokenizer_path) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                let vocab_opt = val.get("model")
                    .and_then(|m| m.get("vocab"))
                    .or_else(|| val.get("vocab"))
                    .and_then(|v| v.as_object());
                
                if let Some(vocab_obj) = vocab_opt {
                    for (token, id_val) in vocab_obj {
                        if let Some(id) = id_val.as_u64() {
                            let id_u32 = id as u32;
                            tokenizer.vocab.insert(token.clone(), id_u32);
                            tokenizer.id_to_token.insert(id_u32, token.clone());
                        }
                    }
                    loaded = true;
                    info!("Loaded {} tokens from {}", tokenizer.vocab.len(), tokenizer_path.display());
                }
            }
        }
    }
    
    if !loaded {
        info!("tokenizer.json not found or failed to parse. Creating fallback ASCII tokenizer.");
        // Seed standard ASCII printables and bytes
        for i in 0..256 {
            let byte = i as u8;
            if byte.is_ascii() {
                let s = (byte as char).to_string();
                tokenizer.vocab.insert(s.clone(), i);
                tokenizer.id_to_token.insert(i, s);
            } else {
                let s = format!("\\x{:02x}", byte);
                tokenizer.vocab.insert(s.clone(), i);
                tokenizer.id_to_token.insert(i, s);
            }
        }
        // Add special tokens
        tokenizer.add_special_token("<unk>", 0);
        tokenizer.add_special_token("<s>", 256);
        tokenizer.add_special_token("</s>", 257);
        tokenizer.add_special_token("<think>", 258);
        tokenizer.add_special_token("</think>", 259);
    }
    
    tokenizer
}

fn run_mock_chat(initial_prompt: Option<&str>) {
    println!("\n==================================================");
    println!("   Mentat Sovereign Chat - Interactive Mock Mode   ");
    println!("   Type 'exit' or 'quit' to end the session.      ");
    println!("==================================================\n");

    if let Some(prompt) = initial_prompt {
        print!("User: {}\nMentat: ", prompt);
        respond_mock(prompt);
        println!("\n");
    }

    use std::io::{self, Write};
    let stdin = io::stdin();
    loop {
        print!("User > ");
        let _ = io::stdout().flush();
        let mut input = String::new();
        if stdin.read_line(&mut input).is_err() {
            break;
        }
        let trimmed = input.trim();
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }
        if trimmed.is_empty() {
            continue;
        }
        print!("Mentat > ");
        respond_mock(trimmed);
        println!("\n");
    }
}

fn respond_mock(prompt: &str) {
    let lower = prompt.to_lowercase();
    let response = if lower.contains("hello") || lower.contains("hi") {
        "Hello! I am Mentat, a sovereign Rust-native reasoning assistant. How can I help you today?"
    } else if lower.contains("who are you") || lower.contains("name") {
        "I am Mentat, an AI model running fully locally on your hardware. No external APIs used."
    } else if lower.contains("rust") {
        "Rust is excellent for AI! It gives me maximum memory control, safety, and performance without garbage collection."
    } else if lower.contains("agent") || lower.contains("tool") {
        "I have access to local tools like a secure Python sandbox, a headless browser, and an atomic file patcher."
    } else {
        "Interesting point. Since we are running in mock mode, I am simulating this conversation. Load a real Safetensors model to run mathematical inference!"
    };

    use std::io::{self, Write};
    for c in response.chars() {
        print!("{}", c);
        let _ = io::stdout().flush();
        std::thread::sleep(std::time::Duration::from_millis(15));
    }
}

fn format_llama2_chat(history: &[(String, String)], current_user_msg: &str) -> String {
    let mut prompt = String::new();
    let system_prompt = "You are a helpful, respectful and honest assistant.";
    if history.is_empty() {
        prompt.push_str(&format!(
            "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
            system_prompt, current_user_msg
        ));
    } else {
        for (i, (user, assistant)) in history.iter().enumerate() {
            if i == 0 {
                prompt.push_str(&format!(
                    "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST] {} </s>",
                    system_prompt, user, assistant
                ));
            } else {
                prompt.push_str(&format!(
                    "<s>[INST] {} [/INST] {} </s>",
                    user, assistant
                ));
            }
        }
        prompt.push_str(&format!("<s>[INST] {} [/INST]", current_user_msg));
    }
    prompt
}

fn run_real_chat(transformer: &Transformer, tokenizer: &BpeTokenizer) {
    println!("\n==================================================");
    println!("   Mentat Sovereign Chat - Real Inference Mode   ");
    println!("   Type 'exit' or 'quit' to end the session.      ");
    println!("==================================================\n");

    use std::io::{self, Write};
    let stdin = io::stdin();
    let mut history: Vec<(String, String)> = Vec::new();

    loop {
        print!("User > ");
        let _ = io::stdout().flush();
        let mut input = String::new();
        if stdin.read_line(&mut input).is_err() {
            break;
        }
        let trimmed = input.trim();
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }
        if trimmed.is_empty() {
            continue;
        }
        print!("Mentat > ");
        let _ = io::stdout().flush();

        let formatted_prompt = format_llama2_chat(&history, trimmed);
        let prompt_tokens = tokenizer.encode(&formatted_prompt);
        let stop_token = tokenizer.vocab.get("</s>").copied().or(Some(257));
        
        let mut response_tokens = Vec::new();
        match transformer.generate(&prompt_tokens, 150, stop_token, |token_id| {
            response_tokens.push(token_id);
            if let Some(token) = tokenizer.id_to_token.get(&token_id) {
                print!("{}", clean_token(token));
            } else {
                print!("[{}]", token_id);
            }
            let _ = io::stdout().flush();
        }) {
            Ok(_) => {
                let mut response_text = String::new();
                for id in &response_tokens {
                    if let Some(token) = tokenizer.id_to_token.get(id) {
                        response_text.push_str(&clean_token(token));
                    }
                }
                history.push((trimmed.to_string(), response_text));
            }
            Err(e) => {
                println!("\n>> Generation Error: {}", e);
            }
        }
        println!("\n");
    }
}

fn default_device() -> mentat::tensor::backend::Device {
    #[cfg(feature = "metal")]
    {
        mentat::tensor::backend::Device::Metal(0)
    }
    #[cfg(feature = "cuda")]
    {
        mentat::tensor::backend::Device::Cuda(0)
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        mentat::tensor::backend::Device::Cpu
    }
}

fn clean_token(token: &str) -> String {
    if token.starts_with("<0x") && token.ends_with(">") && token.len() == 6 {
        if let Ok(byte_val) = u8::from_str_radix(&token[3..5], 16) {
            return (byte_val as char).to_string();
        }
    }
    token.replace("\u{2581}", " ")
}
