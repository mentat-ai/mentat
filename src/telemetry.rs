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

use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use tracing::{error, info};

#[derive(Debug, Serialize, Deserialize)]
pub struct InteractionRecord {
    pub prompt: String,
    pub response: String,
    pub timestamp: String,
}

pub struct DataCollector {
    enabled: bool,
    filepath: PathBuf,
}

impl DataCollector {
    pub fn new(enabled: bool) -> Self {
        let home_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let mentat_dir = home_dir.join(".mentat");
        
        if enabled {
            if let Err(e) = std::fs::create_dir_all(&mentat_dir) {
                error!("Failed to create telemetry directory: {}", e);
            }
        }

        Self {
            enabled,
            filepath: mentat_dir.join("training_data.jsonl"),
        }
    }

    pub fn record_interaction(&self, prompt: &str, response: &str) {
        if !self.enabled {
            return;
        }

        let record = InteractionRecord {
            prompt: prompt.to_string(),
            response: response.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        match serde_json::to_string(&record) {
            Ok(json_line) => {
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.filepath);

                match file {
                    Ok(mut f) => {
                        if let Err(e) = writeln!(f, "{}", json_line) {
                            error!("Failed to write telemetry data: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to open telemetry file: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to serialize interaction record: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs::File;
    use std::io::Read;

    #[test]
    fn test_record_interaction_disabled() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("training_data.jsonl");
        
        let mut collector = DataCollector::new(false);
        collector.filepath = filepath.clone(); // Override path for testing

        collector.record_interaction("hello", "world");
        assert!(!filepath.exists());
    }

    #[test]
    fn test_record_interaction_enabled() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("training_data.jsonl");
        
        let mut collector = DataCollector::new(true);
        collector.filepath = filepath.clone(); // Override path for testing

        collector.record_interaction("hello", "world");
        
        assert!(filepath.exists());
        let mut file = File::open(&filepath).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        
        assert!(contents.contains(r#""prompt":"hello""#));
        assert!(contents.contains(r#""response":"world""#));
    }
}
