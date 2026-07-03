// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Password Strength Explorer — interactive cybersecurity game.
//!
//! The student types a candidate password (never sent anywhere — this is a
//! pure client-side teaching tool) and sees a live entropy estimate, a
//! strength meter, and concrete feedback on what would make it stronger.
//! Teaches character-set diversity, length-vs-complexity tradeoffs, and why
//! common/dictionary passwords are weak regardless of length.

use leptos::prelude::*;

const COMMON_PASSWORDS: &[&str] = &[
    "password",
    "123456",
    "12345678",
    "qwerty",
    "letmein",
    "admin",
    "welcome",
    "monkey",
    "dragon",
    "football",
    "iloveyou",
    "trustno1",
    "password1",
    "abc123",
    "111111",
    "sunshine",
    "princess",
    "starwars",
];

struct StrengthResult {
    entropy_bits: f64,
    label: &'static str,
    color_var: &'static str,
    tips: Vec<&'static str>,
}

fn estimate_strength(password: &str) -> StrengthResult {
    let len = password.chars().count();
    let mut tips = Vec::new();

    if len == 0 {
        return StrengthResult {
            entropy_bits: 0.0,
            label: "Type a password to begin",
            color_var: "var(--text-tertiary)",
            tips: vec!["Try mixing character types and aiming for 12+ characters."],
        };
    }

    let has_lower = password.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = password.chars().any(|c| c.is_ascii_uppercase());
    let has_digit = password.chars().any(|c| c.is_ascii_digit());
    let has_symbol = password
        .chars()
        .any(|c| !c.is_ascii_alphanumeric() && !c.is_whitespace());

    // Charset size drives per-character entropy (log2 of the pool size).
    let mut pool_size: f64 = 0.0;
    if has_lower {
        pool_size += 26.0;
    }
    if has_upper {
        pool_size += 26.0;
    }
    if has_digit {
        pool_size += 10.0;
    }
    if has_symbol {
        pool_size += 32.0;
    }
    if pool_size == 0.0 {
        pool_size = 1.0;
    }

    let mut entropy_bits = len as f64 * pool_size.log2();

    let lower_pw = password.to_lowercase();
    let is_common = COMMON_PASSWORDS
        .iter()
        .any(|common| lower_pw.contains(common));
    if is_common {
        // A dictionary/common password is guessable in a handful of attempts
        // no matter how long the surrounding padding is.
        entropy_bits = entropy_bits.min(10.0);
        tips.push("This contains a very common password or word — attackers try these first, regardless of length.");
    }

    if len < 8 {
        tips.push("Aim for at least 12 characters — length matters more than complexity.");
    }
    if !has_upper {
        tips.push("Add an uppercase letter to widen the character pool.");
    }
    if !has_digit {
        tips.push("Add a digit to widen the character pool.");
    }
    if !has_symbol {
        tips.push("Add a symbol (like ! or # or -) to widen the character pool.");
    }

    let (label, color_var) = match entropy_bits {
        b if b < 28.0 => ("Very Weak", "var(--error)"),
        b if b < 36.0 => ("Weak", "var(--warning)"),
        b if b < 60.0 => ("Reasonable", "var(--info)"),
        b if b < 80.0 => ("Strong", "var(--mastery-green)"),
        _ => ("Very Strong", "var(--mastery-green)"),
    };

    if tips.is_empty() {
        tips.push("Great mix of length and character variety!");
    }

    StrengthResult {
        entropy_bits,
        label,
        color_var,
        tips,
    }
}

#[component]
pub fn PasswordStrengthGame(node_id: String) -> impl IntoView {
    let _node_id = node_id;
    let (password, set_password) = signal(String::new());
    let (reveal, set_reveal) = signal(false);

    let strength = Memo::new(move |_| {
        let result = estimate_strength(&password.get());
        (
            result.entropy_bits,
            result.label,
            result.color_var,
            result.tips,
        )
    });

    // Bar fill is capped visually at ~90 bits so the meter doesn't look
    // permanently near-empty for realistic passwords.
    let meter_pct = move || {
        let (bits, ..) = strength.get();
        (bits / 90.0 * 100.0).clamp(2.0, 100.0)
    };

    view! {
        <div class="password-strength-game" style="max-width: 480px">
            <p style="font-size: 0.8rem; color: var(--text-tertiary); margin-bottom: 1rem">
                "Nothing you type here leaves your browser — this is a local teaching tool, not a real login form."
            </p>

            <div style="position: relative; margin-bottom: 0.5rem">
                <input
                    type=move || if reveal.get() { "text" } else { "password" }
                    placeholder="Try a candidate password..."
                    style="width: 100%; padding: 0.6rem 2.5rem 0.6rem 0.6rem; font-family: monospace; font-size: 1rem"
                    prop:value=password
                    on:input=move |ev| set_password.set(event_target_value(&ev))
                />
                <button
                    type="button"
                    class="btn-outline"
                    style="position: absolute; right: 0.3rem; top: 50%; transform: translateY(-50%); font-size: 0.7rem; padding: 0.2rem 0.5rem"
                    on:click=move |_| set_reveal.update(|r| *r = !*r)
                >
                    {move || if reveal.get() { "Hide" } else { "Show" }}
                </button>
            </div>

            <div style="height: 10px; border-radius: 5px; background: var(--surface-low); overflow: hidden; margin-bottom: 0.5rem">
                <div style=move || format!(
                    "height: 100%; width: {}%; background: {}; transition: width 0.2s ease, background 0.2s ease",
                    meter_pct(), strength.get().2
                )></div>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 1rem">
                <span style=move || format!("font-weight: 700; color: {}", strength.get().2)>
                    {move || strength.get().1}
                </span>
                <span style="font-size: 0.7rem; color: var(--text-tertiary); font-family: monospace">
                    {move || format!("~{:.0} bits of entropy", strength.get().0)}
                </span>
            </div>

            <ul style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.6; padding-left: 1.2rem; margin: 0">
                {move || strength.get().3.into_iter().map(|tip| view! { <li>{tip}</li> }).collect::<Vec<_>>()}
            </ul>
        </div>
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_password_is_zero_entropy() {
        let r = estimate_strength("");
        assert_eq!(r.entropy_bits, 0.0);
    }

    #[test]
    fn common_password_is_capped_low_regardless_of_length() {
        let r = estimate_strength("password12345678");
        assert!(r.entropy_bits <= 10.0);
    }

    #[test]
    fn longer_mixed_charset_scores_higher_than_short_lowercase_only() {
        let weak = estimate_strength("abcdefgh");
        let strong = estimate_strength("Tr0ub4dor&3xplor3!");
        assert!(strong.entropy_bits > weak.entropy_bits);
    }

    #[test]
    fn adding_a_symbol_increases_entropy_for_same_length() {
        let without = estimate_strength("Abcdefg1");
        let with = estimate_strength("Abcdefg!");
        assert!(with.entropy_bits > without.entropy_bits);
    }
}
