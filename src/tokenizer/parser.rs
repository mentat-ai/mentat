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

/// Represents a parsed block from the model's output in the Harmony format.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedBlock {
    /// Standard text output from the model.
    Text(String),
    /// A reasoning chain encapsulated by `<think>` tags.
    Thought(String),
    /// A tool invocation block, e.g., `<python>print('hello')</python>`.
    ToolCall {
        tool_name: String,
        arguments: String,
    },
}

/// A rudimentary parser for the Harmony response format.
/// In a production scenario, this would likely be a state machine or use a crate like `nom`.
pub struct HarmonyParser;

impl HarmonyParser {
    /// Parses a raw string decoded from the model into a sequence of structured blocks.
    pub fn parse(input: &str) -> Vec<ParsedBlock> {
        let mut blocks = Vec::new();
        let mut current_text = String::new();

        // This is a naive, split-based approach for demonstration.
        // It looks for <think> and </think> tags specifically.
        // It can be easily extended to `<python>` or `<browser>` tags.

        let mut chars = input.chars().peekable();
        let mut in_think_block = false;
        let mut current_block_content = String::new();

        while let Some(c) = chars.next() {
            if c == '<' {
                // Peek ahead to see if it's a known tag
                let rest: String = chars.clone().collect();

                if !in_think_block && rest.starts_with("think>") {
                    // Flush accumulated text
                    if !current_text.trim().is_empty() {
                        blocks.push(ParsedBlock::Text(current_text.clone()));
                    }
                    current_text.clear();

                    // Consume "think>"
                    for _ in 0..6 {
                        chars.next();
                    }
                    in_think_block = true;
                    continue;
                } else if in_think_block && rest.starts_with("/think>") {
                    // Flush accumulated thought
                    blocks.push(ParsedBlock::Thought(current_block_content.clone()));
                    current_block_content.clear();

                    // Consume "/think>"
                    for _ in 0..7 {
                        chars.next();
                    }
                    in_think_block = false;
                    continue;
                }

                // For tool calls (e.g., <python> or <file_patcher>)
                if !in_think_block && rest.starts_with("python>") {
                    if !current_text.trim().is_empty() {
                        blocks.push(ParsedBlock::Text(current_text.clone()));
                    }
                    current_text.clear();

                    for _ in 0..7 {
                        chars.next();
                    }

                    // Read until </python>
                    let mut tool_content = String::new();
                    while let Some(tc) = chars.next() {
                        if tc == '<' {
                            let trest: String = chars.clone().collect();
                            if trest.starts_with("/python>") {
                                for _ in 0..8 {
                                    chars.next();
                                }
                                break;
                            }
                        }
                        tool_content.push(tc);
                    }

                    blocks.push(ParsedBlock::ToolCall {
                        tool_name: "python".to_string(),
                        arguments: tool_content,
                    });
                    continue;
                } else if !in_think_block && rest.starts_with("file_patcher>") {
                    if !current_text.trim().is_empty() {
                        blocks.push(ParsedBlock::Text(current_text.clone()));
                    }
                    current_text.clear();

                    for _ in 0..13 {
                        chars.next();
                    }

                    // Read until </file_patcher>
                    let mut tool_content = String::new();
                    while let Some(tc) = chars.next() {
                        if tc == '<' {
                            let trest: String = chars.clone().collect();
                            if trest.starts_with("/file_patcher>") {
                                for _ in 0..14 {
                                    chars.next();
                                }
                                break;
                            }
                        }
                        tool_content.push(tc);
                    }

                    blocks.push(ParsedBlock::ToolCall {
                        tool_name: "file_patcher".to_string(),
                        arguments: tool_content,
                    });
                    continue;
                } else if !in_think_block && rest.starts_with("browser>") {
                    if !current_text.trim().is_empty() {
                        blocks.push(ParsedBlock::Text(current_text.clone()));
                    }
                    current_text.clear();

                    for _ in 0..8 {
                        chars.next();
                    }

                    // Read until </browser>
                    let mut tool_content = String::new();
                    while let Some(tc) = chars.next() {
                        if tc == '<' {
                            let trest: String = chars.clone().collect();
                            if trest.starts_with("/browser>") {
                                for _ in 0..9 {
                                    chars.next();
                                }
                                break;
                            }
                        }
                        tool_content.push(tc);
                    }

                    blocks.push(ParsedBlock::ToolCall {
                        tool_name: "browser".to_string(),
                        arguments: tool_content,
                    });
                    continue;
                }
            }

            if in_think_block {
                current_block_content.push(c);
            } else {
                current_text.push(c);
            }
        }

        // Flush any remaining text
        if !current_text.trim().is_empty() {
            blocks.push(ParsedBlock::Text(current_text));
        }

        blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_text() {
        let input = "Hello, how can I help you today?";
        let blocks = HarmonyParser::parse(input);

        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0],
            ParsedBlock::Text("Hello, how can I help you today?".to_string())
        );
    }

    #[test]
    fn test_parse_think_block() {
        let input = "<think>I need to add 2 and 2.</think>The answer is 4.";
        let blocks = HarmonyParser::parse(input);

        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[0],
            ParsedBlock::Thought("I need to add 2 and 2.".to_string())
        );
        assert_eq!(blocks[1], ParsedBlock::Text("The answer is 4.".to_string()));
    }

    #[test]
    fn test_parse_tool_call() {
        let input = "Let me calculate that for you.\n<python>print(2 + 2)</python>\nDone.";
        let blocks = HarmonyParser::parse(input);

        assert_eq!(blocks.len(), 3);
        assert_eq!(
            blocks[0],
            ParsedBlock::Text("Let me calculate that for you.\n".to_string())
        );
        assert_eq!(
            blocks[1],
            ParsedBlock::ToolCall {
                tool_name: "python".to_string(),
                arguments: "print(2 + 2)".to_string(),
            }
        );
        assert_eq!(blocks[2], ParsedBlock::Text("\nDone.".to_string()));
    }
}
