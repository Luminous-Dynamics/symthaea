use super::CodingAgent;

impl CodingAgent {
    /// Check code quality before allowing it into the Testing phase.
    ///
    /// Returns `Some(reason)` if the code is too low quality to test, `None` if OK.
    pub(crate) fn check_code_quality(code: &str) -> Option<String> {
        let trimmed = code.trim();

        // Empty or near-empty code
        if trimmed.is_empty() || trimmed.len() < 10 {
            return Some("code is empty or trivially short".into());
        }

        // Contains TODO/unimplemented markers (indicating the generator punted)
        if trimmed.contains("// TODO: implement") || trimmed.contains("todo!(") {
            return Some("code contains TODO placeholder".into());
        }
        if trimmed.contains("unimplemented!(") {
            return Some("code contains unimplemented!() placeholder".into());
        }

        // Function with empty body: `fn X() { }` or `fn X() {}`
        for line in trimmed.lines() {
            let l = line.trim();
            if l.starts_with("pub fn ") || l.starts_with("fn ") {
                break; // Only flag via the TODO/unimplemented checks above
            }
        }

        // Pure comment code (no actual Rust statements)
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

        // LLM markdown wrapper (code came back with ```rust ... ```)
        if trimmed.starts_with("```") {
            return Some("code is wrapped in markdown fences".into());
        }

        // ── LLM-specific failure patterns ────────────────────────────────

        // Hallucinated imports: `use` of crates that don't exist in this project
        let hallucinated_crates = [
            "use crate_name::", // placeholder crate
            "use my_crate::",   // LLM default naming
            "use your_crate::", // LLM addressing user
            "use example::",    // example placeholder
            "use foo::",        // test placeholder
            "use bar::",        // test placeholder
        ];
        for hc in &hallucinated_crates {
            if trimmed.contains(hc) {
                return Some(format!("hallucinated import: {hc}"));
            }
        }

        // Incomplete function: function signature with `...` or `/* ... */` body
        if trimmed.contains("...") && (trimmed.contains("fn ") || trimmed.contains("impl ")) {
            return Some("code contains '...' ellipsis (incomplete)".into());
        }
        if trimmed.contains("/* ... */") || trimmed.contains("/* TODO */") {
            return Some("code contains placeholder comment block".into());
        }

        // LLM explanation leak: natural language sentences in what should be pure code
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
        // Only flag if these appear outside of doc comments
        for marker in &explanation_markers {
            for line in trimmed.lines() {
                let l = line.trim();
                if l.starts_with(marker) && !l.starts_with("//") && !l.starts_with("///") {
                    return Some(format!("LLM explanation leak: '{}'", &l[..l.len().min(60)]));
                }
            }
        }

        // Duplicate function definitions (LLM sometimes generates the same fn twice)
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

    /// HDC verification gate: checks generated code against codebase patterns.
    ///
    /// Returns `true` if the code passes verification (safe to write).
    /// Returns `false` if the code is flagged as suspicious (high surprise AND
    /// epistemic uncertainty). When `false`, the code is still written but
    /// a warning observation is recorded.
    #[cfg(feature = "code_generation")]
    pub(crate) fn verify_generated_code_hdc(&self, code: &str) -> (bool, f32) {
        use crate::hdc::code_encoder::CodeHDEncoder;

        let memory = match &self.code_memory {
            Some(m) => m,
            None => return (true, 0.0), // no memory → skip verification
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
    pub(crate) fn verify_generated_code_hdc(&self, _code: &str) -> (bool, f32) {
        (true, 0.0)
    }
}
