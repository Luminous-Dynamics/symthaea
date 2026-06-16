// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Syntax Highlighting for Nix Expressions
//!
//! Provides terminal-friendly syntax highlighting for:
//! - Nix expressions
//! - Shell commands
//! - Configuration paths
//!
//! Uses ratatui-compatible spans for TUI rendering.
//!
//! Requires the `ratatui` feature to be enabled.

#![cfg(feature = "shell")]

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Token types for syntax highlighting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// Keywords: let, in, if, then, else, inherit, with, rec, assert
    Keyword,
    /// Builtin functions: import, throw, builtins, etc.
    Builtin,
    /// String literals (including multiline)
    String,
    /// Interpolation markers: ${ }
    Interpolation,
    /// Comments: # ...
    Comment,
    /// Numbers
    Number,
    /// Boolean: true, false
    Boolean,
    /// Null
    Null,
    /// Attribute path: foo.bar.baz
    AttrPath,
    /// Operators: = ++ // ? @ :
    Operator,
    /// Brackets: { } [ ] ( )
    Bracket,
    /// Function arrow: ->
    Arrow,
    /// Path: ./path or /nix/store/...
    Path,
    /// URI: https://...
    Uri,
    /// Identifier (variables, attributes)
    Identifier,
    /// Whitespace
    Whitespace,
    /// Plain text (no highlighting)
    Plain,
}

impl TokenType {
    /// Get the ratatui style for this token type
    pub fn style(&self) -> Style {
        match self {
            Self::Keyword => Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
            Self::Builtin => Style::default().fg(Color::Cyan),
            Self::String => Style::default().fg(Color::Green),
            Self::Interpolation => Style::default().fg(Color::Yellow),
            Self::Comment => Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
            Self::Number => Style::default().fg(Color::LightBlue),
            Self::Boolean => Style::default().fg(Color::LightYellow),
            Self::Null => Style::default().fg(Color::Red),
            Self::AttrPath => Style::default().fg(Color::LightCyan),
            Self::Operator => Style::default().fg(Color::Yellow),
            Self::Bracket => Style::default().fg(Color::White),
            Self::Arrow => Style::default().fg(Color::Magenta),
            Self::Path => Style::default().fg(Color::Blue),
            Self::Uri => Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::UNDERLINED),
            Self::Identifier => Style::default().fg(Color::White),
            Self::Whitespace => Style::default(),
            Self::Plain => Style::default(),
        }
    }
}

/// A highlighted token
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub token_type: TokenType,
}

impl Token {
    pub fn new(text: impl Into<String>, token_type: TokenType) -> Self {
        Self {
            text: text.into(),
            token_type,
        }
    }

    /// Convert to ratatui Span
    pub fn to_span(&self) -> Span<'_> {
        Span::styled(&self.text, self.token_type.style())
    }
}

/// Nix-aware syntax highlighter
pub struct NixHighlighter {
    /// Keywords to highlight
    keywords: Vec<&'static str>,
    /// Builtin functions
    builtins: Vec<&'static str>,
}

impl Default for NixHighlighter {
    fn default() -> Self {
        Self::new()
    }
}

impl NixHighlighter {
    pub fn new() -> Self {
        Self {
            keywords: vec![
                "let", "in", "if", "then", "else", "inherit", "with", "rec", "assert", "or", "and",
            ],
            builtins: vec![
                "import",
                "throw",
                "abort",
                "builtins",
                "derivation",
                "fetchurl",
                "fetchTarball",
                "fetchGit",
                "null",
                "true",
                "false",
                "map",
                "filter",
                "fold",
                "head",
                "tail",
                "length",
                "toString",
                "toJSON",
                "fromJSON",
                "trace",
                "seq",
                "deepSeq",
            ],
        }
    }

    /// Highlight a line of Nix code
    pub fn highlight_line(&self, line: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = line.chars().collect();
        let mut pos = 0;

        while pos < chars.len() {
            let remaining: String = chars[pos..].iter().collect();

            // Skip whitespace
            if chars[pos].is_whitespace() {
                let mut ws = String::new();
                while pos < chars.len() && chars[pos].is_whitespace() {
                    ws.push(chars[pos]);
                    pos += 1;
                }
                tokens.push(Token::new(ws, TokenType::Whitespace));
                continue;
            }

            // Comment: # ...
            if chars[pos] == '#' {
                let comment: String = chars[pos..].iter().collect();
                tokens.push(Token::new(comment, TokenType::Comment));
                break;
            }

            // String: "..." or ''...''
            if chars[pos] == '"' {
                tokens.push(self.parse_double_string(&chars, &mut pos));
                continue;
            }

            if pos + 1 < chars.len() && chars[pos] == '\'' && chars[pos + 1] == '\'' {
                tokens.push(self.parse_multiline_string(&chars, &mut pos));
                continue;
            }

            // Interpolation: ${
            if pos + 1 < chars.len() && chars[pos] == '$' && chars[pos + 1] == '{' {
                tokens.push(Token::new("${", TokenType::Interpolation));
                pos += 2;
                continue;
            }

            // Arrow: ->
            if pos + 1 < chars.len() && chars[pos] == '-' && chars[pos + 1] == '>' {
                tokens.push(Token::new("->", TokenType::Arrow));
                pos += 2;
                continue;
            }

            // Operators: ++ // ? @ : = == != <= >= < > && ||
            if let Some(op) = self.try_parse_operator(&chars, &mut pos) {
                tokens.push(Token::new(op, TokenType::Operator));
                continue;
            }

            // Brackets
            if matches!(chars[pos], '{' | '}' | '[' | ']' | '(' | ')') {
                tokens.push(Token::new(chars[pos].to_string(), TokenType::Bracket));
                pos += 1;
                continue;
            }

            // Path: ./... or /...
            if chars[pos] == '/'
                || (chars[pos] == '.' && pos + 1 < chars.len() && chars[pos + 1] == '/')
            {
                if let Some(path) = self.try_parse_path(&chars, &mut pos) {
                    tokens.push(Token::new(path, TokenType::Path));
                    continue;
                }
            }

            // URI: http:// https:// etc.
            if remaining.starts_with("http://") || remaining.starts_with("https://") {
                let uri = self.parse_uri(&chars, &mut pos);
                tokens.push(Token::new(uri, TokenType::Uri));
                continue;
            }

            // Number
            if chars[pos].is_ascii_digit() {
                let num = self.parse_number(&chars, &mut pos);
                tokens.push(Token::new(num, TokenType::Number));
                continue;
            }

            // Identifier or keyword
            if chars[pos].is_alphabetic() || chars[pos] == '_' {
                let ident = self.parse_identifier(&chars, &mut pos);

                // Check token type
                let token_type = if self.keywords.contains(&ident.as_str()) {
                    TokenType::Keyword
                } else if ident == "true" || ident == "false" {
                    TokenType::Boolean
                } else if ident == "null" {
                    TokenType::Null
                } else if self.builtins.contains(&ident.as_str()) {
                    TokenType::Builtin
                } else {
                    TokenType::Identifier
                };

                tokens.push(Token::new(ident, token_type));
                continue;
            }

            // Semicolon and comma
            if chars[pos] == ';' || chars[pos] == ',' {
                tokens.push(Token::new(chars[pos].to_string(), TokenType::Operator));
                pos += 1;
                continue;
            }

            // Unknown - just pass through
            tokens.push(Token::new(chars[pos].to_string(), TokenType::Plain));
            pos += 1;
        }

        tokens
    }

    /// Highlight and convert to ratatui Line
    pub fn highlight_to_line(&self, text: &str) -> Line<'static> {
        let tokens = self.highlight_line(text);
        let spans: Vec<Span<'static>> = tokens
            .into_iter()
            .map(|t| Span::styled(t.text, t.token_type.style()))
            .collect();
        Line::from(spans)
    }

    fn parse_double_string(&self, chars: &[char], pos: &mut usize) -> Token {
        let mut s = String::new();
        s.push(chars[*pos]); // Opening "
        *pos += 1;

        let mut escaped = false;
        while *pos < chars.len() {
            let c = chars[*pos];
            s.push(c);
            *pos += 1;

            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                break;
            }
        }

        Token::new(s, TokenType::String)
    }

    fn parse_multiline_string(&self, chars: &[char], pos: &mut usize) -> Token {
        let mut s = String::new();
        s.push_str("''"); // Opening ''
        *pos += 2;

        while *pos + 1 < chars.len() {
            if chars[*pos] == '\'' && chars[*pos + 1] == '\'' {
                s.push_str("''");
                *pos += 2;
                break;
            }
            s.push(chars[*pos]);
            *pos += 1;
        }

        Token::new(s, TokenType::String)
    }

    fn try_parse_operator(&self, chars: &[char], pos: &mut usize) -> Option<String> {
        let two_char_ops = ["++", "//", "==", "!=", "<=", ">=", "&&", "||", "->"];
        let one_char_ops = ['+', '-', '*', '/', '?', '@', ':', '=', '<', '>', '!'];

        // Try two-char operators first
        if *pos + 1 < chars.len() {
            let two: String = chars[*pos..=*pos + 1].iter().collect();
            if two_char_ops.contains(&two.as_str()) {
                *pos += 2;
                return Some(two);
            }
        }

        // Single char operators
        if one_char_ops.contains(&chars[*pos]) {
            let op = chars[*pos].to_string();
            *pos += 1;
            return Some(op);
        }

        None
    }

    fn try_parse_path(&self, chars: &[char], pos: &mut usize) -> Option<String> {
        let start = *pos;
        let mut path = String::new();

        while *pos < chars.len() {
            let c = chars[*pos];
            if c.is_alphanumeric() || matches!(c, '/' | '.' | '-' | '_') {
                path.push(c);
                *pos += 1;
            } else {
                break;
            }
        }

        // Must contain at least one /
        if path.contains('/') && path.len() > 1 {
            Some(path)
        } else {
            *pos = start;
            None
        }
    }

    fn parse_uri(&self, chars: &[char], pos: &mut usize) -> String {
        let mut uri = String::new();

        while *pos < chars.len() {
            let c = chars[*pos];
            if c.is_whitespace() || matches!(c, '"' | '\'' | ')' | ']' | '}' | ';' | ',') {
                break;
            }
            uri.push(c);
            *pos += 1;
        }

        uri
    }

    fn parse_number(&self, chars: &[char], pos: &mut usize) -> String {
        let mut num = String::new();

        while *pos < chars.len() && (chars[*pos].is_ascii_digit() || chars[*pos] == '.') {
            num.push(chars[*pos]);
            *pos += 1;
        }

        num
    }

    fn parse_identifier(&self, chars: &[char], pos: &mut usize) -> String {
        let mut ident = String::new();

        while *pos < chars.len() {
            let c = chars[*pos];
            if c.is_alphanumeric() || c == '_' || c == '-' || c == '\'' {
                ident.push(c);
                *pos += 1;
            } else {
                break;
            }
        }

        ident
    }
}

/// Shell command highlighter (simpler than Nix)
pub struct ShellHighlighter;

impl ShellHighlighter {
    /// Highlight a shell command line
    pub fn highlight_line(line: &str) -> Line<'static> {
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty() {
            return Line::from(line.to_string());
        }

        let mut spans = Vec::new();

        // First word is the command
        let cmd = parts[0];
        let cmd_style = if cmd.starts_with("nix") || cmd.starts_with("nixos") {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else if cmd.starts_with("sudo") {
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Yellow)
        };

        let mut remaining = line.to_string();
        if let Some(idx) = remaining.find(cmd) {
            if idx > 0 {
                spans.push(Span::raw(remaining[..idx].to_string()));
            }
            spans.push(Span::styled(cmd.to_string(), cmd_style));
            remaining = remaining[idx + cmd.len()..].to_string();
        }

        // Highlight flags and arguments
        for word in remaining.split_whitespace() {
            spans.push(Span::raw(" "));

            let style = if word.starts_with("--") {
                Style::default().fg(Color::Green)
            } else if word.starts_with('-') {
                Style::default().fg(Color::Green)
            } else if word.starts_with('/') || word.starts_with("./") {
                Style::default().fg(Color::Blue)
            } else if word.starts_with("http") {
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::UNDERLINED)
            } else {
                Style::default()
            };

            spans.push(Span::styled(word.to_string(), style));
        }

        Line::from(spans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_keyword() {
        let highlighter = NixHighlighter::new();
        let tokens = highlighter.highlight_line("let x = 5 in x");

        let let_token = tokens.iter().find(|t| t.text == "let").unwrap();
        assert_eq!(let_token.token_type, TokenType::Keyword);

        let in_token = tokens.iter().find(|t| t.text == "in").unwrap();
        assert_eq!(in_token.token_type, TokenType::Keyword);
    }

    #[test]
    fn test_highlight_string() {
        let highlighter = NixHighlighter::new();
        let tokens = highlighter.highlight_line("\"hello world\"");

        let string_token = tokens.iter().find(|t| t.text.contains("hello")).unwrap();
        assert_eq!(string_token.token_type, TokenType::String);
    }

    #[test]
    fn test_highlight_comment() {
        let highlighter = NixHighlighter::new();
        let tokens = highlighter.highlight_line("x = 5 # this is a comment");

        let comment_token = tokens.iter().find(|t| t.text.contains("comment")).unwrap();
        assert_eq!(comment_token.token_type, TokenType::Comment);
    }

    #[test]
    fn test_highlight_builtin() {
        let highlighter = NixHighlighter::new();
        let tokens = highlighter.highlight_line("import ./default.nix");

        let import_token = tokens.iter().find(|t| t.text == "import").unwrap();
        assert_eq!(import_token.token_type, TokenType::Builtin);
    }

    #[test]
    fn test_highlight_path() {
        let highlighter = NixHighlighter::new();
        let tokens = highlighter.highlight_line("./some/path");

        let path_token = tokens.iter().find(|t| t.text.contains("some")).unwrap();
        assert_eq!(path_token.token_type, TokenType::Path);
    }

    #[test]
    fn test_highlight_boolean() {
        let highlighter = NixHighlighter::new();
        let tokens = highlighter.highlight_line("enable = true;");

        let bool_token = tokens.iter().find(|t| t.text == "true").unwrap();
        assert_eq!(bool_token.token_type, TokenType::Boolean);
    }

    #[test]
    fn test_shell_highlight() {
        let line = ShellHighlighter::highlight_line("nix-env -iA nixpkgs.firefox");
        assert!(!line.spans.is_empty());
    }
}
