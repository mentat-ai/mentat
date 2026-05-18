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

pub mod browser;
pub mod fs;
pub mod python;

/// A common trait for all agentic tools.
pub trait Tool {
    /// Returns the name of the tool (e.g., "python", "browser").
    fn name(&self) -> &str;

    /// Executes the tool with the given arguments.
    /// Returns the standard output or an error message.
    fn execute(&self, arguments: &str) -> Result<String, String>;
}
