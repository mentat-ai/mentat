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

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "mentat")]
#[command(about = "Mentat Sovereign Inference Engine", long_about = None)]
pub struct Config {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable debug logging
    #[arg(long, global = true)]
    pub debug: bool,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run a model directly
    Run {
        /// Path to the model weights file (e.g., .safetensors)
        #[arg(long)]
        model: String,

        /// The initial prompt for the model
        #[arg(long)]
        prompt: Option<String>,
    },
    /// Start an API server
    Serve {
        /// Port for the API server
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}
