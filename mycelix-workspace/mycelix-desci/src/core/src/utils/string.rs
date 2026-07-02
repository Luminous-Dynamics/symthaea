// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! String utilities
//!
//! Functions for string manipulation, formatting, and display

/// Truncate a string to a maximum length, adding ellipsis if needed
pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        "...".to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Truncate a string in the middle, preserving start and end
pub fn truncate_middle(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }

    if max_len <= 3 {
        return "...".to_string();
    }

    let ellipsis_len = 3;
    let available = max_len - ellipsis_len;
    let start_len = (available + 1) / 2;
    let end_len = available / 2;

    format!(
        "{}...{}",
        &s[..start_len],
        &s[s.len() - end_len..]
    )
}

/// Pluralize a word based on count
pub fn pluralize(word: &str, count: usize) -> String {
    if count == 1 {
        word.to_string()
    } else {
        format!("{}s", word)
    }
}

/// Format a count with singular/plural word
pub fn count_with_word(count: usize, word: &str) -> String {
    format!("{} {}", count, pluralize(word, count))
}

/// Capitalize the first letter of a string
pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().chain(chars).collect(),
    }
}

/// Convert to title case (capitalize each word)
pub fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(capitalize)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Convert camelCase or PascalCase to snake_case
pub fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    let mut prev_is_lower = false;

    for ch in s.chars() {
        if ch.is_uppercase() && prev_is_lower {
            result.push('_');
        }
        result.push(ch.to_ascii_lowercase());
        prev_is_lower = ch.is_lowercase();
    }

    result
}

/// Convert snake_case or kebab-case to camelCase
pub fn to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;
    let mut is_first = true;

    for ch in s.chars() {
        if ch == '_' || ch == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else if is_first {
            result.push(ch.to_ascii_lowercase());
            is_first = false;
        } else {
            result.push(ch);
        }
    }

    result
}

/// Convert to kebab-case
pub fn to_kebab_case(s: &str) -> String {
    to_snake_case(s).replace('_', "-")
}

/// Pad a string to a minimum width with spaces
pub fn pad_left(s: &str, width: usize) -> String {
    format!("{:>width$}", s, width = width)
}

/// Pad a string to a minimum width with spaces (right-aligned)
pub fn pad_right(s: &str, width: usize) -> String {
    format!("{:<width$}", s, width = width)
}

/// Center a string within a given width
pub fn center(s: &str, width: usize) -> String {
    if s.len() >= width {
        return s.to_string();
    }

    let padding = width - s.len();
    let left_pad = padding / 2;
    let right_pad = padding - left_pad;

    format!("{}{}{}", " ".repeat(left_pad), s, " ".repeat(right_pad))
}

/// Join a list of items with a separator, handling the last item specially
pub fn join_with_and(items: &[impl AsRef<str>]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].as_ref().to_string(),
        2 => format!("{} and {}", items[0].as_ref(), items[1].as_ref()),
        _ => {
            let all_but_last = items[..items.len() - 1]
                .iter()
                .map(|s| s.as_ref())
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}, and {}", all_but_last, items[items.len() - 1].as_ref())
        }
    }
}

/// Join a list of items with "or"
pub fn join_with_or(items: &[impl AsRef<str>]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].as_ref().to_string(),
        2 => format!("{} or {}", items[0].as_ref(), items[1].as_ref()),
        _ => {
            let all_but_last = items[..items.len() - 1]
                .iter()
                .map(|s| s.as_ref())
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}, or {}", all_but_last, items[items.len() - 1].as_ref())
        }
    }
}

/// Indent each line of text
pub fn indent(text: &str, spaces: usize) -> String {
    let indent_str = " ".repeat(spaces);
    text.lines()
        .map(|line| format!("{}{}", indent_str, line))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Remove common leading whitespace from each line
pub fn dedent(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    // Find minimum indentation (excluding empty lines)
    let min_indent = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    lines
        .iter()
        .map(|line| {
            if line.len() >= min_indent {
                &line[min_indent..]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Wrap text to a maximum line width
pub fn wrap(text: &str, width: usize) -> String {
    let mut result = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.is_empty() {
            current_line.push_str(word);
        } else if current_line.len() + 1 + word.len() <= width {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            result.push(current_line.clone());
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        result.push(current_line);
    }

    result.join("\n")
}

/// Escape special characters for shell/command line use
pub fn shell_escape(s: &str) -> String {
    if s.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.') {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// Remove ANSI color codes from a string
pub fn strip_ansi(s: &str) -> String {
    let mut result = String::new();
    let mut in_escape = false;

    for ch in s.chars() {
        if ch == '\x1b' {
            in_escape = true;
        } else if in_escape && ch == 'm' {
            in_escape = false;
        } else if !in_escape {
            result.push(ch);
        }
    }

    result
}

/// Check if a string contains only ASCII characters
pub fn is_ascii(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii())
}

/// Count the number of lines in a string
pub fn line_count(s: &str) -> usize {
    if s.is_empty() {
        0
    } else {
        s.lines().count()
    }
}

/// Extract the first line of a string
pub fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

/// Extract the last line of a string
pub fn last_line(s: &str) -> &str {
    s.lines().last().unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_exact() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_long() {
        assert_eq!(truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_truncate_very_short() {
        assert_eq!(truncate("hello", 3), "...");
        assert_eq!(truncate("hello", 2), "...");
    }

    #[test]
    fn test_truncate_middle() {
        assert_eq!(truncate_middle("hello", 10), "hello");
        assert_eq!(truncate_middle("hello world", 8), "hel...ld");
        assert_eq!(truncate_middle("1234567890", 7), "12...90");
    }

    #[test]
    fn test_pluralize() {
        assert_eq!(pluralize("item", 0), "items");
        assert_eq!(pluralize("item", 1), "item");
        assert_eq!(pluralize("item", 2), "items");
        assert_eq!(pluralize("item", 10), "items");
    }

    #[test]
    fn test_count_with_word() {
        assert_eq!(count_with_word(0, "item"), "0 items");
        assert_eq!(count_with_word(1, "item"), "1 item");
        assert_eq!(count_with_word(5, "item"), "5 items");
    }

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("hello"), "Hello");
        assert_eq!(capitalize("HELLO"), "HELLO");
        assert_eq!(capitalize("h"), "H");
        assert_eq!(capitalize(""), "");
    }

    #[test]
    fn test_title_case() {
        assert_eq!(title_case("hello world"), "Hello World");
        assert_eq!(title_case("the quick brown fox"), "The Quick Brown Fox");
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("camelCase"), "camel_case");
        assert_eq!(to_snake_case("PascalCase"), "pascal_case");
        assert_eq!(to_snake_case("simpleword"), "simpleword");
        assert_eq!(to_snake_case("HTTPResponse"), "httpresponse");
    }

    #[test]
    fn test_to_camel_case() {
        assert_eq!(to_camel_case("snake_case"), "snakeCase");
        assert_eq!(to_camel_case("kebab-case"), "kebabCase");
        assert_eq!(to_camel_case("simple"), "simple");
    }

    #[test]
    fn test_to_kebab_case() {
        assert_eq!(to_kebab_case("camelCase"), "camel-case");
        assert_eq!(to_kebab_case("PascalCase"), "pascal-case");
    }

    #[test]
    fn test_pad_left() {
        assert_eq!(pad_left("hello", 10), "     hello");
        assert_eq!(pad_left("hello", 5), "hello");
        assert_eq!(pad_left("hello", 3), "hello");
    }

    #[test]
    fn test_pad_right() {
        assert_eq!(pad_right("hello", 10), "hello     ");
        assert_eq!(pad_right("hello", 5), "hello");
    }

    #[test]
    fn test_center() {
        assert_eq!(center("hi", 6), "  hi  ");
        assert_eq!(center("hi", 5), " hi  ");
        assert_eq!(center("hello", 5), "hello");
    }

    #[test]
    fn test_join_with_and() {
        assert_eq!(join_with_and(&[] as &[&str]), "");
        assert_eq!(join_with_and(&["one"]), "one");
        assert_eq!(join_with_and(&["one", "two"]), "one and two");
        assert_eq!(join_with_and(&["one", "two", "three"]), "one, two, and three");
        assert_eq!(join_with_and(&["a", "b", "c", "d"]), "a, b, c, and d");
    }

    #[test]
    fn test_join_with_or() {
        assert_eq!(join_with_or(&[] as &[&str]), "");
        assert_eq!(join_with_or(&["one"]), "one");
        assert_eq!(join_with_or(&["one", "two"]), "one or two");
        assert_eq!(join_with_or(&["one", "two", "three"]), "one, two, or three");
    }

    #[test]
    fn test_indent() {
        let text = "line1\nline2\nline3";
        let indented = indent(text, 4);
        assert_eq!(indented, "    line1\n    line2\n    line3");
    }

    #[test]
    fn test_dedent() {
        let text = "    line1\n    line2\n      line3";
        let dedented = dedent(text);
        assert_eq!(dedented, "line1\nline2\n  line3");
    }

    #[test]
    fn test_dedent_with_empty_lines() {
        let text = "    line1\n\n    line2";
        let dedented = dedent(text);
        assert_eq!(dedented, "line1\n\nline2");
    }

    #[test]
    fn test_wrap() {
        let text = "the quick brown fox jumps over the lazy dog";
        let wrapped = wrap(text, 20);
        assert!(wrapped.lines().all(|line| line.len() <= 20));
    }

    #[test]
    fn test_wrap_short() {
        let text = "short";
        assert_eq!(wrap(text, 20), "short");
    }

    #[test]
    fn test_shell_escape() {
        assert_eq!(shell_escape("simple"), "simple");
        assert_eq!(shell_escape("with spaces"), "'with spaces'");
        assert_eq!(shell_escape("with'quote"), "'with'\\''quote'");
    }

    #[test]
    fn test_strip_ansi() {
        let colored = "\x1b[31mred text\x1b[0m normal";
        assert_eq!(strip_ansi(colored), "red text normal");

        let no_color = "plain text";
        assert_eq!(strip_ansi(no_color), "plain text");
    }

    #[test]
    fn test_is_ascii() {
        assert!(is_ascii("hello"));
        assert!(is_ascii("123"));
        assert!(is_ascii("hello123!@#"));
        assert!(!is_ascii("hello 世界"));
        assert!(!is_ascii("café"));
    }

    #[test]
    fn test_line_count() {
        assert_eq!(line_count(""), 0);
        assert_eq!(line_count("single line"), 1);
        assert_eq!(line_count("line1\nline2"), 2);
        assert_eq!(line_count("line1\nline2\nline3"), 3);
    }

    #[test]
    fn test_first_line() {
        assert_eq!(first_line(""), "");
        assert_eq!(first_line("single"), "single");
        assert_eq!(first_line("first\nsecond"), "first");
    }

    #[test]
    fn test_last_line() {
        assert_eq!(last_line(""), "");
        assert_eq!(last_line("single"), "single");
        assert_eq!(last_line("first\nsecond"), "second");
        assert_eq!(last_line("first\nsecond\nthird"), "third");
    }
}
