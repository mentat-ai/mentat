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

/// Represents a headless browser tool for the model to search the web.
pub struct BrowserTool;

impl Tool for BrowserTool {
    fn name(&self) -> &str {
        "browser"
    }

    fn execute(&self, arguments: &str) -> Result<String, String> {
        let url = arguments.trim();
        if url.is_empty() {
            return Err("BrowserTool requires a URL argument".to_string());
        }

        // Fetch the URL content using ureq
        match ureq::get(url).call() {
            Ok(response) => {
                match response.into_string() {
                    Ok(body) => {
                        // Return the first 500 characters to simulate reading the page safely
                        let preview: String = body.chars().take(500).collect();
                        Ok(format!("Page content preview:\n{}...", preview))
                    }
                    Err(e) => Err(format!("Failed to read response body: {}", e)),
                }
            }
            Err(e) => Err(format!("Failed to fetch URL '{}': {}", url, e)),
        }
    }
}
