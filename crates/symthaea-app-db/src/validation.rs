// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Input validation for NixOS installer fields.

/// Result of validating one or more installer fields.
pub struct ValidationResult {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// RFC 1123 hostname: lowercase, [a-z0-9-], max 63 chars, no leading/trailing hyphen.
pub fn validate_hostname(s: &str) -> Result<String, String> {
    let s = s.trim().to_lowercase();
    if s.is_empty() {
        return Err("Hostname cannot be empty".into());
    }
    if s.len() > 63 {
        return Err("Hostname must be 63 characters or fewer".into());
    }
    if s.starts_with('-') || s.ends_with('-') {
        return Err("Hostname cannot start or end with a hyphen".into());
    }
    if !s
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err("Hostname can only contain lowercase letters, numbers, and hyphens".into());
    }
    Ok(s)
}

/// Linux username: lowercase, [a-z_][a-z0-9_-]*, max 32 chars.
pub fn validate_username(s: &str) -> Result<String, String> {
    let s = s.trim().to_lowercase().replace(' ', "");
    if s.is_empty() {
        return Err("Username cannot be empty".into());
    }
    if s.len() > 32 {
        return Err("Username must be 32 characters or fewer".into());
    }
    if s.chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        return Err("Username cannot start with a digit".into());
    }
    if !s
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
    {
        return Err(
            "Username can only contain lowercase letters, numbers, underscores, and hyphens".into(),
        );
    }
    if s == "root" {
        return Err("Cannot use 'root' as username".into());
    }
    Ok(s)
}

/// IANA timezone format: Region/City or UTC.
pub fn validate_timezone(s: &str) -> Result<(), String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("Timezone cannot be empty".into());
    }
    if s == "UTC" || s == "GMT" {
        return Ok(());
    }
    if s.contains('/')
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '/' || c == '_' || c == '-' || c == '+')
    {
        return Ok(());
    }
    Err("Timezone must be in IANA format (e.g., America/New_York, Europe/London, UTC)".into())
}

/// Known XKB keyboard layouts.
const KNOWN_KEYBOARDS: &[&str] = &[
    "us", "gb", "de", "fr", "es", "it", "pt", "br-abnt2", "ru", "jp", "kr", "se", "no", "dk", "fi",
    "nl", "be", "ch", "at", "pl", "cz", "hu", "ro", "bg", "hr", "sk", "si", "lt", "lv", "ee", "tr",
    "il", "ar", "in", "th", "cn", "tw", "za", "dvorak", "colemak", "neo",
];

/// Validate a keyboard layout string. Unknown layouts pass (user may know better).
pub fn validate_keyboard(s: &str) -> Result<(), String> {
    let s = s.trim().to_lowercase();
    if s.is_empty() {
        return Err("Keyboard layout cannot be empty".into());
    }
    if KNOWN_KEYBOARDS.contains(&s.as_str()) {
        return Ok(());
    }
    // Allow unknown layouts with a warning-style pass (user may know better)
    Ok(())
}

/// Returns true if the given keyboard layout is a known XKB layout.
pub fn is_known_keyboard(s: &str) -> bool {
    KNOWN_KEYBOARDS.contains(&s.trim().to_lowercase().as_str())
}

/// Validate all step 1 fields (system basics).
pub fn validate_step1(
    hostname: &str,
    username: &str,
    timezone: &str,
    keyboard: &str,
) -> ValidationResult {
    let mut errors = vec![];
    let mut warnings = vec![];
    if let Err(e) = validate_hostname(hostname) {
        errors.push(e);
    }
    if let Err(e) = validate_username(username) {
        errors.push(e);
    }
    if let Err(e) = validate_timezone(timezone) {
        errors.push(e);
    }
    if let Err(e) = validate_keyboard(keyboard) {
        warnings.push(e); // keyboard is a warning, not blocking
    }
    if validate_keyboard(keyboard).is_ok()
        && !is_known_keyboard(keyboard)
        && !keyboard.trim().is_empty()
    {
        warnings.push(format!(
            "\"{}\" is not a recognized XKB layout. It may still work if your system supports it.",
            keyboard.trim()
        ));
    }
    ValidationResult { errors, warnings }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Hostname ──

    #[test]
    fn hostname_valid_simple() {
        assert_eq!(validate_hostname("my-nixos").unwrap(), "my-nixos");
    }

    #[test]
    fn hostname_lowercases() {
        assert_eq!(validate_hostname("MyHost").unwrap(), "myhost");
    }

    #[test]
    fn hostname_trims_whitespace() {
        assert_eq!(validate_hostname("  host  ").unwrap(), "host");
    }

    #[test]
    fn hostname_empty() {
        assert!(validate_hostname("").is_err());
        assert!(validate_hostname("   ").is_err());
    }

    #[test]
    fn hostname_too_long() {
        let long = "a".repeat(64);
        assert!(validate_hostname(&long).is_err());
        // Exactly 63 is fine
        let exact = "a".repeat(63);
        assert!(validate_hostname(&exact).is_ok());
    }

    #[test]
    fn hostname_leading_trailing_hyphen() {
        assert!(validate_hostname("-bad").is_err());
        assert!(validate_hostname("bad-").is_err());
        assert!(validate_hostname("-both-").is_err());
    }

    #[test]
    fn hostname_invalid_chars() {
        assert!(validate_hostname("host_name").is_err());
        assert!(validate_hostname("host.name").is_err());
        assert!(validate_hostname("host name").is_err());
        assert!(validate_hostname("host@name").is_err());
    }

    #[test]
    fn hostname_digits_and_hyphens() {
        assert_eq!(validate_hostname("nixos-42").unwrap(), "nixos-42");
        assert_eq!(validate_hostname("123").unwrap(), "123");
    }

    // ── Username ──

    #[test]
    fn username_valid() {
        assert_eq!(validate_username("tristan").unwrap(), "tristan");
        assert_eq!(validate_username("user_name").unwrap(), "user_name");
        assert_eq!(validate_username("dev-1").unwrap(), "dev-1");
    }

    #[test]
    fn username_strips_spaces_and_lowercases() {
        assert_eq!(validate_username("My User").unwrap(), "myuser");
    }

    #[test]
    fn username_empty() {
        assert!(validate_username("").is_err());
        assert!(validate_username("   ").is_err());
    }

    #[test]
    fn username_too_long() {
        let long = "a".repeat(33);
        assert!(validate_username(&long).is_err());
        let exact = "a".repeat(32);
        assert!(validate_username(&exact).is_ok());
    }

    #[test]
    fn username_starts_with_digit() {
        assert!(validate_username("1user").is_err());
    }

    #[test]
    fn username_root_forbidden() {
        assert!(validate_username("root").is_err());
        assert!(validate_username("  ROOT  ").is_err());
    }

    #[test]
    fn username_invalid_chars() {
        assert!(validate_username("user@name").is_err());
        assert!(validate_username("user.name").is_err());
    }

    // ── Timezone ──

    #[test]
    fn timezone_valid() {
        assert!(validate_timezone("UTC").is_ok());
        assert!(validate_timezone("GMT").is_ok());
        assert!(validate_timezone("America/New_York").is_ok());
        assert!(validate_timezone("Africa/Johannesburg").is_ok());
        assert!(validate_timezone("Asia/Kolkata").is_ok());
    }

    #[test]
    fn timezone_empty() {
        assert!(validate_timezone("").is_err());
    }

    #[test]
    fn timezone_invalid() {
        assert!(validate_timezone("Eastern").is_err());
        assert!(validate_timezone("EST").is_err());
        assert!(validate_timezone("Central Time").is_err());
    }

    // ── Keyboard ──

    #[test]
    fn keyboard_known_layouts() {
        assert!(validate_keyboard("us").is_ok());
        assert!(validate_keyboard("dvorak").is_ok());
        assert!(validate_keyboard("br-abnt2").is_ok());
    }

    #[test]
    fn keyboard_empty() {
        assert!(validate_keyboard("").is_err());
    }

    #[test]
    fn keyboard_unknown_passes() {
        // Unknown layouts are allowed (pass-through)
        assert!(validate_keyboard("latam").is_ok());
    }

    #[test]
    fn keyboard_case_insensitive() {
        assert!(validate_keyboard("US").is_ok());
        assert!(is_known_keyboard("US"));
    }

    // ── Step 1 composite ──

    #[test]
    fn step1_all_valid() {
        let r = validate_step1("my-nixos", "tristan", "Africa/Johannesburg", "us");
        assert!(r.is_ok());
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn step1_collects_multiple_errors() {
        let r = validate_step1("", "", "", "");
        assert!(!r.is_ok());
        assert!(r.errors.len() >= 3); // hostname, username, timezone
    }

    #[test]
    fn step1_unknown_keyboard_warns() {
        let r = validate_step1("host", "user", "UTC", "latam");
        assert!(r.is_ok()); // no errors
        assert!(!r.warnings.is_empty()); // has keyboard warning
    }

    // ── Edge case tests ──

    #[test]
    fn hostname_with_consecutive_hyphens() {
        assert!(validate_hostname("my--host").is_ok()); // Valid per RFC
    }

    #[test]
    fn hostname_single_char() {
        assert!(validate_hostname("a").is_ok());
    }

    #[test]
    fn hostname_max_length() {
        let h = "a".repeat(63);
        assert!(validate_hostname(&h).is_ok());
        let h = "a".repeat(64);
        assert!(validate_hostname(&h).is_err());
    }

    #[test]
    fn username_with_underscore() {
        assert!(validate_username("my_user").is_ok());
    }

    #[test]
    fn username_root_rejected() {
        assert!(validate_username("root").is_err());
    }

    #[test]
    fn timezone_common_values() {
        for tz in &[
            "UTC",
            "America/New_York",
            "Europe/London",
            "Asia/Tokyo",
            "Australia/Sydney",
        ] {
            assert!(
                validate_timezone(tz).is_ok(),
                "Timezone '{}' should be valid",
                tz
            );
        }
    }

    #[test]
    fn timezone_invalid_format() {
        assert!(validate_timezone("Not A Timezone!!!").is_err());
        assert!(validate_timezone("12345").is_err());
    }
}
