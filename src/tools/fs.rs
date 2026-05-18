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

use super::Tool;
use std::fs;
use std::path::Path;

/// Represents a filesystem operation tool (e.g., creating/reading files).
/// Currently supports simple READ and WRITE commands.
pub struct FilePatcherTool;

impl Tool for FilePatcherTool {
    fn name(&self) -> &str {
        "file_patcher"
    }

    fn execute(&self, arguments: &str) -> Result<String, String> {
        let lines: Vec<&str> = arguments.lines().collect();
        if lines.is_empty() {
            return Err("Empty arguments for file_patcher. Expected 'READ <path>' or 'WRITE <path>\\n<content>'".to_string());
        }

        let command_line = lines[0].trim();
        let parts: Vec<&str> = command_line.splitn(2, ' ').collect();

        if parts.is_empty() {
            return Err("Invalid command format".to_string());
        }

        let cmd = parts[0];

        match cmd {
            "READ" => {
                if parts.len() < 2 {
                    return Err("READ command requires a file path".to_string());
                }
                let path = parts[1].trim();
                match fs::read_to_string(path) {
                    Ok(content) => Ok(content),
                    Err(e) => Err(format!("Failed to read file '{}': {}", path, e)),
                }
            }
            "WRITE" => {
                if parts.len() < 2 {
                    return Err("WRITE command requires a file path".to_string());
                }
                let path = parts[1].trim();
                let content = if lines.len() > 1 {
                    lines[1..].join("\n")
                } else {
                    String::new()
                };

                // Basic security check: prevent simple path traversal
                if path.contains("..") {
                    return Err(
                        "Path traversal (..) is not allowed for security reasons".to_string()
                    );
                }

                // Ensure parent directories exist
                if let Some(parent) = Path::new(path).parent() {
                    if !parent.as_os_str().is_empty() {
                        if let Err(e) = fs::create_dir_all(parent) {
                            return Err(format!(
                                "Failed to create parent directories for '{}': {}",
                                path, e
                            ));
                        }
                    }
                }

                match fs::write(path, content) {
                    Ok(_) => Ok(format!("Successfully wrote to '{}'", path)),
                    Err(e) => Err(format!("Failed to write to file '{}': {}", path, e)),
                }
            }
            _ => Err(format!(
                "Unknown file_patcher command: '{}'. Supported commands are READ and WRITE",
                cmd
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_file_patcher_write_and_read() {
        let tool = FilePatcherTool;
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let file_path_str = file_path.to_str().unwrap();

        // Test WRITE
        let write_args = format!("WRITE {}\nHello, Mentat!", file_path_str);
        let write_result = tool.execute(&write_args);
        assert!(write_result.is_ok());

        // Test READ
        let read_args = format!("READ {}", file_path_str);
        let read_result = tool.execute(&read_args);
        assert!(read_result.is_ok());
        assert_eq!(read_result.unwrap(), "Hello, Mentat!");
    }

    #[test]
    fn test_file_patcher_security() {
        let tool = FilePatcherTool;
        let result = tool.execute("WRITE ../../../etc/passwd\nhacked");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Path traversal"));
    }
}
