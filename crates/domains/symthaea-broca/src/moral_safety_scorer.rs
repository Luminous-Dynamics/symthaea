// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Moral Safety Scorer (M-axis)
//! Detects dangerous code patterns using regex.

use regex::Regex;

pub fn compute_moral_safety(code: &str) -> f32 {
    let severe_patterns = [
        r#"Command::new\s*\(\s*"rm"\s*\)"#,
        r#"std::process::Command::new\s*\(\s*"rm"\s*\)"#,
        r"rm\s+-rf",
        r"/etc/shadow",
        r"/etc/passwd",
    ];
    let patterns = [
        r"unsafe\s*\{",
        r"std::mem::transmute",
        r"std::ptr::",
        r"std::os::unix::io",
        r"std::os::windows::io",
        r"Command::new",
        r"std::process::Command",
        r"system\(",
        r"exec\(",
        r"eval\(",
        r"chmod",
        r"chown",
        r"curl\s+",
        r"wget\s+",
        r"nc\s+-l",
        r"iptables",
    ];

    let mut penalty: f32 = 0.0;
    for p in severe_patterns {
        if let Ok(re) = Regex::new(p)
            && re.is_match(code)
        {
            penalty += 0.4;
        }
    }
    for p in patterns {
        if let Ok(re) = Regex::new(p)
            && re.is_match(code)
        {
            penalty += 0.2;
        }
    }

    (1.0 - penalty).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moral_safety() {
        assert_eq!(compute_moral_safety("fn safe() {}"), 1.0);
        assert!(compute_moral_safety("unsafe { }") < 1.0);
        assert!(compute_moral_safety("Command::new(\"rm\")") < 0.7);
    }
}
