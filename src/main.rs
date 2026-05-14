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
use tracing::{Level, debug, info};
use tracing_subscriber::FmtSubscriber;

use mentat::config::{Commands, Config};

fn main() {
    let config = Config::parse();

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
            info!("Starting inference (Not implemented yet)");
        }
        Commands::Serve { port } => {
            info!("Initializing 'serve' mode");
            info!("Starting API server on port {} (Not implemented yet)", port);
        }
    }
}
