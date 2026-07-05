// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code transformation utilities for the coding agent.

use super::*;

impl CodingAgent {
    /// Determine the target file for code generation.
    pub(super) fn resolve_target_file(&self) -> PathBuf {
        // 1. Explicit config
        if let Some(ref target) = self.config.target_file {
            if target.is_absolute() {
                return target.clone();
            }
            return self.config.working_dir.join(target);
        }

        // 2. Try to extract path from task description
        for word in self.task.split_whitespace() {
            if word.ends_with(".rs") || word.ends_with(".py") || word.ends_with(".nix") {
                let path = PathBuf::from(word);
                if path.is_absolute() {
                    return path;
                }
                return self.config.working_dir.join(path);
            }
        }

        // 3. Default
        self.config.working_dir.join("src").join("lib.rs")
    }

    /// Detect the target language from the target file extension.
    pub(super) fn target_language(&self) -> &'static str {
        let target = self.resolve_target_file();
        match target.extension().and_then(|e| e.to_str()) {
            Some("py") => "python",
            Some("nix") => "nix",
            _ => "rust",
        }
    }

    /// Strip markdown code fences from generated output.
    /// LLM and template outputs sometimes wrap code in ```rust ... ``` blocks.
    pub(super) fn strip_code_fences(code: &str) -> String {
        let trimmed = code.trim();
        // Check for ```rust or ``` at start. Look for the closing fence via
        // `find` (first occurrence after the opener) rather than requiring it
        // to be the literal string suffix — LLMs routinely append trailing
        // prose after the closing fence ("```\n\nThis solution uses...."),
        // which previously made the whole strip silently no-op, leaving the
        // raw fence marker in code that then failed to compile.
        if let Some(rest) = trimmed.strip_prefix("```rust") {
            let rest = rest.strip_prefix('\n').unwrap_or(rest);
            return match rest.find("```") {
                Some(close_idx) => rest[..close_idx].trim().to_string(),
                None => rest.trim().to_string(),
            };
        }
        if let Some(rest) = trimmed.strip_prefix("```") {
            // Could be ```\n...``` or ```rs\n...```
            let rest = rest.strip_prefix("rs").unwrap_or(rest);
            let rest = rest.strip_prefix('\n').unwrap_or(rest);
            return match rest.find("```") {
                Some(close_idx) => rest[..close_idx].trim().to_string(),
                None => rest.trim().to_string(),
            };
        }
        code.to_string()
    }

    /// Post-generation sanitizer: fixes two systematic code generation bugs.
    ///
    /// 1. **`fn main()` stripping**: When the code generator wraps library code in
    ///    `fn main() { ... }`, extract the inner content.
    ///
    /// 2. **Generic parameter declaration**: When code uses type parameter `T` in
    ///    struct fields/function bodies but forgets to declare `<T>` on the item
    ///    signature, the compiler emits E0412.
    pub(super) fn sanitize_generated_code(code: &str) -> String {
        let mut result = code.to_string();
        result = Self::strip_main_wrapper(&result);
        result = Self::fix_undeclared_generics(&result);
        result
    }

    /// Strip `fn main() { ... }` wrapper if the body contains library items.
    pub(super) fn strip_main_wrapper(code: &str) -> String {
        let trimmed = code.trim();

        if !trimmed.starts_with("fn main()") {
            return code.to_string();
        }

        let Some(open_brace) = trimmed.find('{') else {
            return code.to_string();
        };

        let after_brace = &trimmed[open_brace + 1..];
        let mut depth = 1usize;
        let mut close_pos = None;
        for (i, ch) in after_brace.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close_pos = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }

        let Some(pos) = close_pos else {
            return code.to_string();
        };

        let body = after_brace[..pos].trim();

        let has_lib_items = body.contains("pub fn ")
            || body.contains("pub struct ")
            || body.contains("struct ")
            || body.contains("impl ")
            || body.contains("pub enum ")
            || body.contains("enum ")
            || body.contains("pub trait ")
            || body.contains("trait ");

        if has_lib_items {
            let dedented: Vec<String> = body
                .lines()
                .map(|line| {
                    if let Some(rest) = line.strip_prefix("    ") {
                        rest.to_string()
                    } else if let Some(rest) = line.strip_prefix('\t') {
                        rest.to_string()
                    } else {
                        line.to_string()
                    }
                })
                .collect();
            dedented.join("\n")
        } else {
            code.to_string()
        }
    }

    /// Fix undeclared generic type parameters (E0412).
    pub(super) fn fix_undeclared_generics(code: &str) -> String {
        let lines: Vec<&str> = code.lines().collect();
        let mut result: Vec<String> = Vec::with_capacity(lines.len());

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if (trimmed.starts_with("pub struct ")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("pub enum ")
                || trimmed.starts_with("enum "))
                && !trimmed.contains('<')
                && trimmed.ends_with('{')
            {
                if Self::body_uses_generic_t(&lines, i + 1) {
                    let fixed = Self::insert_generic_param_on_item(trimmed);
                    result.push(line.replace(trimmed, &fixed));
                    continue;
                }
            }

            if (trimmed.starts_with("pub fn ")
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("pub(crate) fn "))
                && !trimmed.contains('<')
                && Self::signature_uses_t(trimmed)
            {
                let fixed = Self::insert_generic_param_on_fn(trimmed);
                result.push(line.replace(trimmed, &fixed));
                continue;
            }

            if (trimmed.starts_with("impl ") || trimmed.starts_with("pub impl "))
                && !trimmed.starts_with("impl<")
                && Self::impl_uses_t(trimmed)
            {
                let fixed = trimmed.replacen("impl ", "impl<T> ", 1);
                result.push(line.replace(trimmed, &fixed));
                continue;
            }

            result.push((*line).to_string());
        }

        result.join("\n")
    }

    /// Check if subsequent lines (inside a struct/enum body) reference type `T`.
    pub(super) fn body_uses_generic_t(lines: &[&str], start: usize) -> bool {
        let mut depth = 1usize;
        for line in &lines[start..] {
            let t = line.trim();
            for ch in t.chars() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return false;
                        }
                    }
                    _ => {}
                }
            }
            if Self::line_references_type_t(t) {
                return true;
            }
        }
        false
    }

    /// Check if a line references `T` as a type parameter (not part of a longer word).
    pub(super) fn line_references_type_t(line: &str) -> bool {
        let bytes = line.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'T' {
                let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
                let after_ok = i + 1 >= bytes.len() || !bytes[i + 1].is_ascii_alphanumeric();
                if before_ok && after_ok {
                    return true;
                }
            }
        }
        false
    }

    /// Check if a function signature uses `T` as a type parameter.
    pub(super) fn signature_uses_t(sig: &str) -> bool {
        if let Some(paren_start) = sig.find('(') {
            let rest = &sig[paren_start..];
            if Self::line_references_type_t(rest) {
                return true;
            }
        }
        false
    }

    /// Check if an impl line uses `T`.
    pub(super) fn impl_uses_t(line: &str) -> bool {
        let rest = line.strip_prefix("pub ").unwrap_or(line);
        let rest = rest.strip_prefix("impl ").unwrap_or(rest);
        Self::line_references_type_t(rest)
    }

    /// Insert `<T>` on a struct/enum declaration line.
    pub(super) fn insert_generic_param_on_item(line: &str) -> String {
        if let Some(brace_pos) = line.find('{') {
            let before_brace = line[..brace_pos].trim_end();
            format!("{}<T> {{", before_brace)
        } else {
            line.to_string()
        }
    }

    /// Insert `<T>` on a function declaration line.
    pub(super) fn insert_generic_param_on_fn(line: &str) -> String {
        if let Some(paren_pos) = line.find('(') {
            format!("{}<T>{}", &line[..paren_pos], &line[paren_pos..])
        } else {
            line.to_string()
        }
    }

    /// HDC verification gate for generated code.
    #[cfg(feature = "code_generation")]
    pub(super) fn verify_generated_code_hdc(&self, code: &str) -> (bool, f32) {
        use crate::hdc::code_encoder::CodeHDEncoder;

        let memory = match &self.code_memory {
            Some(m) => m,
            None => return (true, 0.0),
        };

        let encoder = memory.encoder();
        let code_hv = encoder.encode_name(code);
        let surprise = memory.compute_surprise(&code_hv);
        let matches = memory.query(&code_hv, 3);
        let best_similarity = matches.first().map(|m| m.similarity).unwrap_or(0.0);
        let codebase_too_small = matches.len() < 2;
        let passes = surprise < 0.85 || best_similarity > 0.15 || codebase_too_small;

        if !passes {
            tracing::warn!(
                target: "symthaea::coding_agent",
                surprise = surprise,
                best_similarity = best_similarity,
                "HDC verification gate: generated code is highly surprising"
            );
        }

        (passes, surprise)
    }

    #[cfg(not(feature = "code_generation"))]
    pub(super) fn verify_generated_code_hdc(&self, _code: &str) -> (bool, f32) {
        (true, 0.0)
    }

    pub(super) fn write_code_to_disk(&mut self, target: &PathBuf, code: &str) {
        let code = Self::strip_code_fences(code);
        let code = Self::sanitize_generated_code(&code);

        let (verified, surprise) = self.verify_generated_code_hdc(&code);
        if !verified {
            self.observations.push(format!(
                "HDC verification warning: generated code has surprise={surprise:.3} — \
                 significantly different from codebase patterns. Writing anyway but flagging."
            ));
        }

        if let Some(parent) = target.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                self.errors.push(format!(
                    "Failed to create directory {}: {e}",
                    parent.display()
                ));
                return;
            }
        }

        match std::fs::write(target, &code) {
            Ok(()) => {
                if !self.files_modified.contains(target) {
                    self.files_modified.push(target.clone());
                }
                self.observations.push(format!(
                    "Wrote {} bytes to {}",
                    code.len(),
                    target.display()
                ));
                #[cfg(feature = "code_generation")]
                self.reindex_file(target, &code);
            }
            Err(e) => {
                self.errors
                    .push(format!("Failed to write {}: {e}", target.display()));
            }
        }
    }

    /// Check code quality before allowing it into the Testing phase.
    pub(super) fn check_code_quality(code: &str) -> Option<String> {
        let trimmed = code.trim();

        if trimmed.is_empty() || trimmed.len() < 10 {
            return Some("code is empty or trivially short".into());
        }

        if trimmed.contains("// TODO: implement") || trimmed.contains("todo!(") {
            return Some("code contains TODO placeholder".into());
        }
        if trimmed.contains("unimplemented!(") {
            return Some("code contains unimplemented!() placeholder".into());
        }
        if trimmed.contains("raise NotImplementedError") {
            return Some("code contains NotImplementedError placeholder".into());
        }

        for line in trimmed.lines() {
            let l = line.trim();
            if l.starts_with("pub fn ") || l.starts_with("fn ") {
                break;
            }
        }

        let non_comment_lines: Vec<&str> = trimmed
            .lines()
            .filter(|l| {
                let t = l.trim();
                !t.is_empty() && !t.starts_with("//") && !t.starts_with("///")
            })
            .collect();
        if non_comment_lines.len() <= 1 {
            return Some("code contains only comments, no logic".into());
        }

        if trimmed.starts_with("```") {
            return Some("code is wrapped in markdown fences".into());
        }

        let hallucinated_crates = [
            "use crate_name::",
            "use my_crate::",
            "use your_crate::",
            "use example::",
            "use foo::",
            "use bar::",
        ];
        for hc in &hallucinated_crates {
            if trimmed.contains(hc) {
                return Some(format!("hallucinated import: {hc}"));
            }
        }

        if trimmed.contains("...") && (trimmed.contains("fn ") || trimmed.contains("impl ")) {
            return Some("code contains '...' ellipsis (incomplete)".into());
        }
        if trimmed.contains("/* ... */") || trimmed.contains("/* TODO */") {
            return Some("code contains placeholder comment block".into());
        }
        for line in non_comment_lines.iter().copied() {
            let lower = line.to_lowercase();
            if lower.contains("placeholder for ")
                || lower.contains("placeholder implementation")
                || lower.contains("todo placeholder")
            {
                return Some("code contains placeholder text".into());
            }
        }

        let explanation_markers = [
            "Here is the implementation",
            "Here's the code",
            "Below is",
            "I'll implement",
            "Let me",
            "As you can see",
            "Note that",
            "This function",
            "The following",
        ];
        for marker in &explanation_markers {
            for line in trimmed.lines() {
                let l = line.trim();
                if l.starts_with(marker) && !l.starts_with("//") && !l.starts_with("///") {
                    return Some(format!("LLM explanation leak: '{}'", &l[..l.len().min(60)]));
                }
            }
        }

        let fn_names: Vec<&str> = trimmed
            .lines()
            .filter_map(|l| {
                let t = l.trim();
                if (t.starts_with("pub fn ") || t.starts_with("fn ")) && t.contains('(') {
                    t.split('(').next()
                } else {
                    None
                }
            })
            .collect();
        for (i, name) in fn_names.iter().enumerate() {
            if fn_names[i + 1..].contains(name) {
                return Some(format!("duplicate function definition: {name}"));
            }
        }

        None
    }

    /// Normalize an error message for pattern matching (strip paths, line numbers).
    pub(super) fn normalize_error_pattern(error: &str) -> String {
        let mut normalized = String::new();
        for line in error.lines().take(3) {
            if line.contains("error[E") || line.contains("error:") {
                normalized.push_str(line.trim());
                normalized.push(' ');
            }
        }
        if normalized.is_empty() {
            error.lines().next().unwrap_or(error).to_string()
        } else {
            normalized.trim().to_string()
        }
    }
}
