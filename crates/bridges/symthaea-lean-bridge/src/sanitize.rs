// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared sanitization for strings embedded in generated Lean 4 source.
//!
//! `symthaea-lean-bridge` builds `.lean` files by string interpolation, not
//! through a real Lean parser/printer. Any string that reaches a render
//! function unsanitized is untrusted with respect to Lean's own grammar --
//! in particular a raw newline lets a crafted identifier or statement break
//! out of the enclosing `theorem`/`variable` declaration and start a new
//! top-level Lean command (`#eval`, `run_cmd`, ...), which Lean's elaborator
//! can execute with real IO capability when the file is checked. Every
//! render function in this crate (`LeanTerm::to_lean`, `LeanProofScript::to_lean`,
//! `fol_ext_bridge`'s formula/term renderers) must route caller-supplied
//! text through one of these two functions before interpolating it.

/// Sanitize a string used as a bare Lean identifier (a variable,
/// hypothesis, or theorem name). Returns `s` unchanged if it is already a
/// safe ASCII identifier; otherwise returns a version with every unsafe
/// character replaced by `_`, so the result can never break out of the
/// surrounding declaration. A sanitized (and therefore almost certainly
/// unintended) identifier makes the emitted theorem fail to elaborate --
/// an honest, safe failure, not a silent forgery or code execution.
pub fn sanitize_ident(s: &str) -> String {
    let is_safe_char = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '\'';
    let is_safe = !s.is_empty()
        && !s.starts_with(|c: char| c.is_ascii_digit())
        && s.chars().all(is_safe_char);
    if is_safe {
        return s.to_string();
    }
    let mut out: String = s
        .chars()
        .map(|c| if is_safe_char(c) { c } else { '_' })
        .collect();
    if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
        out = format!("id_{out}");
    }
    out
}

/// Sanitize a string embedded as Lean *statement* syntax (a `Prop`/term
/// expression) rather than a bare identifier -- these legitimately contain
/// parens, arrows, and Unicode math operators, so [`sanitize_ident`] is far
/// too strict. The only thing that must never survive is a raw control
/// character: that is the actual mechanism by which a crafted string could
/// terminate the current declaration and start a new top-level command.
/// Every control character (including newline, carriage return, NUL, and
/// DEL) is replaced with a single space, which can never merge two tokens
/// into a new one.
pub fn sanitize_statement(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_idents_pass_through_unchanged() {
        for s in ["P", "Q", "h0", "x", "n", "t_mp", "h_1'", "mathd_algebra_37"] {
            assert_eq!(sanitize_ident(s), s);
        }
    }

    #[test]
    fn ident_injection_attempt_is_neutralized() {
        let attack =
            "t\nend\n#eval IO.Process.run { cmd := \"sh\", args := #[\"-c\", \"echo pwned\"] }";
        let safe = sanitize_ident(attack);
        assert!(!safe.contains('\n'));
        assert!(!safe.contains('#'));
        assert!(!safe.contains(' '));
        assert!(!safe.contains('"'));
        // Must still be a single safe identifier token.
        assert!(
            safe.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '\'')
        );
    }

    #[test]
    fn empty_ident_gets_safe_fallback() {
        let safe = sanitize_ident("");
        assert!(!safe.is_empty());
        assert!(
            safe.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '\'')
        );
    }

    #[test]
    fn ident_starting_with_digit_gets_prefixed() {
        let safe = sanitize_ident("1theorem");
        assert!(!safe.starts_with(|c: char| c.is_ascii_digit()));
    }

    #[test]
    fn dotted_ident_is_not_safe_here_but_neutralized() {
        // This module only validates single identifiers; dotted theorem
        // names (e.g. axiom_gate's probe target) are that call site's own
        // concern. Confirm dots still get neutralized rather than passed
        // through raw, since `sanitize_ident` doesn't special-case them.
        let safe = sanitize_ident("Foo.bar");
        assert!(!safe.contains('.'));
    }

    #[test]
    fn statement_newline_injection_is_neutralized() {
        let attack = "True\nend\n#eval 1";
        let safe = sanitize_statement(attack);
        assert!(!safe.contains('\n'));
        assert!(safe.contains("#eval 1")); // content survives, just can't break the line
    }

    #[test]
    fn statement_legitimate_unicode_math_survives() {
        let s = "(∀ n, (Nat → True)) ∧ (¬ P) ∨ Q";
        assert_eq!(sanitize_statement(s), s);
    }

    #[test]
    fn statement_control_chars_all_stripped() {
        let attack = "a\rb\tc\0d\x1bd";
        let safe = sanitize_statement(attack);
        assert!(safe.chars().all(|c| !c.is_control()));
    }
}
