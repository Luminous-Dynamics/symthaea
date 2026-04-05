// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tool-use framework for on-device LLM (Gemma 4 E2B function calling).
//!
//! Defines tool schemas, parses `<tool_call>` blocks from LLM output,
//! executes tools via SomaEngine dispatch, and re-prompts with results.
//!
//! Works both on-device (LiteRT) and desktop (Ollama) — same ToolRegistry.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Parameter type for tool definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamType {
    String,
    Number,
    Boolean,
}

/// A single parameter in a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam {
    pub name: String,
    pub param_type: ParamType,
    pub description: String,
    pub required: bool,
}

/// Tool definition — describes what a tool does and its parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParam>,
}

/// Parsed tool call extracted from LLM output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of executing a tool, fed back to LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub success: bool,
    pub output: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Registry
// ═══════════════════════════════════════════════════════════════════════════════

/// Registry of available tools for LLM function calling.
pub struct ToolRegistry {
    tools: Vec<ToolDef>,
}

impl ToolRegistry {
    /// Create a registry with the default Soma tools.
    pub fn default_tools() -> Self {
        Self {
            tools: vec![
                ToolDef {
                    name: "web_search".into(),
                    description: "Search the web for current information".into(),
                    parameters: vec![ToolParam {
                        name: "query".into(),
                        param_type: ParamType::String,
                        description: "Search query".into(),
                        required: true,
                    }],
                },
                ToolDef {
                    name: "calculate".into(),
                    description: "Evaluate a mathematical expression".into(),
                    parameters: vec![ToolParam {
                        name: "expression".into(),
                        param_type: ParamType::String,
                        description: "Math expression (e.g., '2 + 3 * 4')".into(),
                        required: true,
                    }],
                },
                ToolDef {
                    name: "get_time".into(),
                    description: "Get the current date and time".into(),
                    parameters: vec![],
                },
                ToolDef {
                    name: "memory_recall".into(),
                    description: "Recall information from consciousness memory".into(),
                    parameters: vec![ToolParam {
                        name: "topic".into(),
                        param_type: ParamType::String,
                        description: "Topic to recall".into(),
                        required: true,
                    }],
                },
            ],
        }
    }

    /// Generate the tool schema block for the system prompt (Gemma 4 format).
    pub fn system_prompt_block(&self) -> String {
        let mut block = String::from("Available tools:\n");
        for tool in &self.tools {
            let params: Vec<String> = tool
                .parameters
                .iter()
                .map(|p| format!("{}: {:?}", p.name, p.param_type))
                .collect();
            if params.is_empty() {
                block.push_str(&format!("- {}(): {}\n", tool.name, tool.description));
            } else {
                block.push_str(&format!(
                    "- {}({}): {}\n",
                    tool.name,
                    params.join(", "),
                    tool.description
                ));
            }
        }
        block.push_str(
            "\nTo use a tool, respond with: <tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>\n",
        );
        block
    }

    /// Look up a tool by name.
    pub fn find(&self, name: &str) -> Option<&ToolDef> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Parser
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse `<tool_call>...</tool_call>` blocks from LLM output.
///
/// Returns the text with tool calls removed, and a list of parsed tool calls.
pub fn parse_tool_calls(output: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut clean_text = String::new();
    let mut remaining = output;

    while let Some(start) = remaining.find("<tool_call>") {
        // Add text before the tool call
        clean_text.push_str(&remaining[..start]);

        let after_tag = &remaining[start + "<tool_call>".len()..];
        if let Some(end) = after_tag.find("</tool_call>") {
            let json_str = after_tag[..end].trim();
            if let Ok(call) = serde_json::from_str::<ToolCall>(json_str) {
                calls.push(call);
            }
            remaining = &after_tag[end + "</tool_call>".len()..];
        } else {
            // Unclosed tag — include as text
            clean_text.push_str(&remaining[..start + "<tool_call>".len()]);
            remaining = after_tag;
        }
    }
    clean_text.push_str(remaining);

    (clean_text.trim().to_string(), calls)
}

/// Format tool results for re-prompting the LLM.
pub fn format_tool_results(results: &[ToolResult]) -> String {
    let mut block = String::from("Tool results:\n");
    for result in results {
        if result.success {
            block.push_str(&format!("[{}]: {}\n", result.name, result.output));
        } else {
            block.push_str(&format!("[{} FAILED]: {}\n", result.name, result.output));
        }
    }
    block.push_str("\nNow answer the user's question using the tool results above.\n");
    block
}

/// Execute the built-in `calculate` tool (simple math evaluation).
pub fn execute_calculate(expression: &str) -> ToolResult {
    // Simple evaluator: supports +, -, *, / with proper precedence
    let result = evaluate_simple_math(expression);
    match result {
        Some(value) => ToolResult {
            name: "calculate".into(),
            success: true,
            output: format!("{expression} = {value}"),
        },
        None => ToolResult {
            name: "calculate".into(),
            success: false,
            output: format!("Could not evaluate: {expression}"),
        },
    }
}

/// Execute the built-in `get_time` tool.
pub fn execute_get_time() -> ToolResult {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Format as ISO-ish (no chrono dependency — keep lightweight)
    let secs_per_day = 86400u64;
    let days_since_epoch = now / secs_per_day;
    let time_of_day = now % secs_per_day;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    ToolResult {
        name: "get_time".into(),
        success: true,
        output: format!(
            "Unix timestamp: {now}, day {days_since_epoch} since epoch, time {hours:02}:{minutes:02}:{seconds:02} UTC"
        ),
    }
}

/// Simple math evaluator (no dependencies).
/// Supports: +, -, *, /, parentheses, decimal numbers.
fn evaluate_simple_math(expr: &str) -> Option<f64> {
    let expr = expr.trim();
    if expr.is_empty() {
        return None;
    }

    // Tokenize
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in expr.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            current.push(ch);
        } else if "+-*/()".contains(ch) {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            tokens.push(ch.to_string());
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        } else {
            return None; // unsupported character
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    // Shunting-yard → RPN → evaluate
    let rpn = shunting_yard(&tokens)?;
    evaluate_rpn(&rpn)
}

fn precedence(op: &str) -> u8 {
    match op {
        "+" | "-" => 1,
        "*" | "/" => 2,
        _ => 0,
    }
}

fn shunting_yard(tokens: &[String]) -> Option<Vec<String>> {
    let mut output = Vec::new();
    let mut ops: Vec<String> = Vec::new();

    for token in tokens {
        if token.parse::<f64>().is_ok() {
            output.push(token.clone());
        } else if "+-*/".contains(token.as_str()) {
            while let Some(top) = ops.last() {
                if top != "(" && precedence(top) >= precedence(token) {
                    output.push(ops.pop().unwrap());
                } else {
                    break;
                }
            }
            ops.push(token.clone());
        } else if token == "(" {
            ops.push(token.clone());
        } else if token == ")" {
            while let Some(top) = ops.pop() {
                if top == "(" {
                    break;
                }
                output.push(top);
            }
        }
    }

    while let Some(op) = ops.pop() {
        if op == "(" {
            return None; // mismatched
        }
        output.push(op);
    }

    Some(output)
}

fn evaluate_rpn(rpn: &[String]) -> Option<f64> {
    let mut stack: Vec<f64> = Vec::new();

    for token in rpn {
        if let Ok(num) = token.parse::<f64>() {
            stack.push(num);
        } else {
            let b = stack.pop()?;
            let a = stack.pop()?;
            let result = match token.as_str() {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => {
                    if b.abs() < f64::EPSILON {
                        return None;
                    }
                    a / b
                }
                _ => return None,
            };
            stack.push(result);
        }
    }

    if stack.len() == 1 {
        Some(stack[0])
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_calls_basic() {
        let output = r#"Let me search for that. <tool_call>{"name":"web_search","arguments":{"query":"NixOS flakes"}}</tool_call>"#;
        let (text, calls) = parse_tool_calls(output);
        assert_eq!(text, "Let me search for that.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["query"], "NixOS flakes");
    }

    #[test]
    fn test_parse_tool_calls_multiple() {
        let output = r#"<tool_call>{"name":"get_time","arguments":{}}</tool_call> and <tool_call>{"name":"calculate","arguments":{"expression":"2+3"}}</tool_call>"#;
        let (text, calls) = parse_tool_calls(output);
        assert_eq!(text, "and");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_time");
        assert_eq!(calls[1].name, "calculate");
    }

    #[test]
    fn test_parse_tool_calls_none() {
        let output = "Just a normal response with no tools.";
        let (text, calls) = parse_tool_calls(output);
        assert_eq!(text, output);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_tool_calls_malformed() {
        let output = "<tool_call>not json</tool_call>";
        let (_, calls) = parse_tool_calls(output);
        assert!(calls.is_empty()); // invalid JSON → skipped
    }

    #[test]
    fn test_tool_registry_default() {
        let reg = ToolRegistry::default_tools();
        assert_eq!(reg.len(), 4);
        assert!(reg.find("web_search").is_some());
        assert!(reg.find("calculate").is_some());
        assert!(reg.find("get_time").is_some());
        assert!(reg.find("memory_recall").is_some());
        assert!(reg.find("nonexistent").is_none());
    }

    #[test]
    fn test_system_prompt_block() {
        let reg = ToolRegistry::default_tools();
        let block = reg.system_prompt_block();
        assert!(block.contains("web_search"));
        assert!(block.contains("<tool_call>"));
    }

    #[test]
    fn test_format_tool_results() {
        let results = vec![
            ToolResult {
                name: "calculate".into(),
                success: true,
                output: "2 + 3 = 5".into(),
            },
            ToolResult {
                name: "web_search".into(),
                success: false,
                output: "timeout".into(),
            },
        ];
        let formatted = format_tool_results(&results);
        assert!(formatted.contains("[calculate]: 2 + 3 = 5"));
        assert!(formatted.contains("[web_search FAILED]: timeout"));
    }

    #[test]
    fn test_calculate_basic() {
        let r = execute_calculate("2 + 3");
        assert!(r.success);
        assert!(r.output.contains("5"));
    }

    #[test]
    fn test_calculate_precedence() {
        let r = execute_calculate("2 + 3 * 4");
        assert!(r.success);
        assert!(r.output.contains("14"));
    }

    #[test]
    fn test_calculate_parentheses() {
        let r = execute_calculate("(2 + 3) * 4");
        assert!(r.success);
        assert!(r.output.contains("20"));
    }

    #[test]
    fn test_calculate_division_by_zero() {
        let r = execute_calculate("5 / 0");
        assert!(!r.success);
    }

    #[test]
    fn test_get_time() {
        let r = execute_get_time();
        assert!(r.success);
        assert!(r.output.contains("UTC"));
    }
}
