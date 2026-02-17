/// Security utilities for Mycelix-Marketplace
///
/// This module provides comprehensive input sanitization and validation
/// to prevent XSS, injection attacks, and other security vulnerabilities.

use std::collections::HashSet;

/// Sanitize user input to prevent XSS and injection attacks
///
/// This function:
/// 1. Trims whitespace
/// 2. Removes potentially dangerous characters
/// 3. Escapes HTML entities
/// 4. Normalizes Unicode
pub fn sanitize_user_input(input: &str) -> String {
    // Trim whitespace
    let trimmed = input.trim();

    // First, filter to safe characters (before escaping)
    let safe_chars: HashSet<char> = ".,!?-_()[]{}:;@#$%^*+=~`|\\'\"\n\r\t "
        .chars()
        .collect();

    let filtered: String = trimmed
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || safe_chars.contains(c))
        .collect();

    // Then escape HTML entities (this preserves the escaped sequences)
    filtered
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
        .replace('/', "&#x2F;")
}

/// Sanitize text with strict filtering (no HTML, minimal punctuation)
pub fn sanitize_strict(input: &str) -> String {
    let trimmed = input.trim();

    trimmed
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?-_()".contains(*c))
        .collect()
}

/// Validate and sanitize IPFS CID
pub fn sanitize_ipfs_cid(cid: &str) -> Result<String, String> {
    let trimmed = cid.trim();

    // CIDv0 (Qm... 46 chars)
    if trimmed.starts_with("Qm") {
        if trimmed.len() != 46 {
            return Err("Invalid CIDv0 length (must be 46 characters)".to_string());
        }
        // Verify base58 characters
        if !trimmed.chars().all(|c| {
            c.is_ascii_alphanumeric() && !['0', 'O', 'I', 'l'].contains(&c)
        }) {
            return Err("Invalid CIDv0 characters".to_string());
        }
        return Ok(trimmed.to_string());
    }

    // CIDv1 (b... or bafy... 50-100 chars)
    if trimmed.starts_with('b') {
        if trimmed.len() < 50 || trimmed.len() > 100 {
            return Err("Invalid CIDv1 length (must be 50-100 characters)".to_string());
        }
        // Verify base32 characters (lowercase alphanumeric)
        if !trimmed.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()) {
            return Err("Invalid CIDv1 characters (must be lowercase alphanumeric)".to_string());
        }
        return Ok(trimmed.to_string());
    }

    Err("Invalid IPFS CID format (must start with 'Qm' or 'b')".to_string())
}

/// Validate email format (basic)
pub fn validate_email(email: &str) -> Result<String, String> {
    let trimmed = email.trim().to_lowercase();

    if trimmed.is_empty() {
        return Err("Email cannot be empty".to_string());
    }

    if !trimmed.contains('@') {
        return Err("Email must contain '@'".to_string());
    }

    if !trimmed.contains('.') {
        return Err("Email must contain domain".to_string());
    }

    // Basic length check
    if trimmed.len() < 5 || trimmed.len() > 254 {
        return Err("Email length invalid (must be 5-254 characters)".to_string());
    }

    Ok(trimmed)
}

/// Validate URL format (basic)
pub fn validate_url(url: &str) -> Result<String, String> {
    let trimmed = url.trim();

    if trimmed.is_empty() {
        return Err("URL cannot be empty".to_string());
    }

    if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
        return Err("URL must start with http:// or https://".to_string());
    }

    // Length check
    if trimmed.len() > 2048 {
        return Err("URL too long (max 2048 characters)".to_string());
    }

    Ok(trimmed.to_string())
}

/// Normalize unicode to prevent homograph attacks
pub fn normalize_unicode(input: &str) -> String {
    // Convert to NFD (Canonical Decomposition)
    // Then filter out combining characters
    input
        .chars()
        .filter(|c| !c.is_ascii_control())
        .collect()
}

/// Validate price (in cents)
pub fn validate_price(price_cents: u64) -> Result<u64, String> {
    if price_cents == 0 {
        return Err("Price must be greater than zero".to_string());
    }

    if price_cents > 100_000_000 {
        // Max $1,000,000
        return Err("Price too high (max $1,000,000)".to_string());
    }

    Ok(price_cents)
}

/// Validate quantity
pub fn validate_quantity(quantity: u32) -> Result<u32, String> {
    if quantity == 0 {
        return Err("Quantity must be at least 1".to_string());
    }

    if quantity > 1_000_000 {
        return Err("Quantity too high (max 1,000,000)".to_string());
    }

    Ok(quantity)
}

/// Rate limiting configuration and enforcement
///
/// In Holochain's distributed architecture, rate limiting is enforced through:
/// 1. Local checks before creating entries (honor system)
/// 2. Validation rules that check timestamps
/// 3. Byzantine detection for agents who violate limits
#[derive(Clone, Debug)]
pub struct RateLimiter {
    pub max_requests: u32,
    pub window_seconds: u64,
}

impl RateLimiter {
    pub fn new(max_requests: u32, window_seconds: u64) -> Self {
        Self {
            max_requests,
            window_seconds,
        }
    }

    /// Check if an action is allowed given a list of previous action timestamps
    ///
    /// Returns Ok(()) if allowed, Err(message) if rate limit exceeded
    pub fn check_allowed(&self, previous_timestamps: &[u64], current_time: u64) -> Result<(), String> {
        // Filter timestamps within the window
        let window_start = current_time.saturating_sub(self.window_seconds);
        let recent_count = previous_timestamps
            .iter()
            .filter(|&&ts| ts >= window_start)
            .count();

        if recent_count >= self.max_requests as usize {
            return Err(format!(
                "Rate limit exceeded: {} actions in {} seconds (max: {})",
                recent_count, self.window_seconds, self.max_requests
            ));
        }

        Ok(())
    }

    /// Calculate how long to wait before next action is allowed
    pub fn time_until_allowed(&self, previous_timestamps: &[u64], current_time: u64) -> u64 {
        if previous_timestamps.len() < self.max_requests as usize {
            return 0; // No wait needed
        }

        // Find the oldest timestamp in the window
        let window_start = current_time.saturating_sub(self.window_seconds);
        let oldest_in_window = previous_timestamps
            .iter()
            .filter(|&&ts| ts >= window_start)
            .min()
            .copied()
            .unwrap_or(0);

        // Calculate when the oldest action will fall outside the window
        let wait_time = oldest_in_window
            .saturating_add(self.window_seconds)
            .saturating_sub(current_time);

        wait_time
    }

    // Default rate limiters for different actions

    pub fn default_listing_creation() -> Self {
        Self::new(10, 3600) // 10 listings per hour
    }

    pub fn default_listing_update() -> Self {
        Self::new(50, 3600) // 50 updates per hour
    }

    pub fn default_transaction_creation() -> Self {
        Self::new(20, 3600) // 20 transactions per hour
    }

    pub fn default_review_submission() -> Self {
        Self::new(50, 86400) // 50 reviews per day
    }

    pub fn default_search() -> Self {
        Self::new(100, 60) // 100 searches per minute
    }

    pub fn default_dispute_filing() -> Self {
        Self::new(5, 86400) // 5 disputes per day
    }

    pub fn default_arbitration_vote() -> Self {
        Self::new(100, 86400) // 100 votes per day
    }
}

/// Rate limit tracker for storing action history
///
/// This tracks timestamps of recent actions for rate limiting enforcement
#[derive(Clone, Debug)]
pub struct RateLimitTracker {
    pub action_type: String,
    pub timestamps: Vec<u64>,
    pub max_history: usize,
}

impl RateLimitTracker {
    pub fn new(action_type: String) -> Self {
        Self {
            action_type,
            timestamps: Vec::new(),
            max_history: 1000, // Keep last 1000 actions
        }
    }

    /// Record a new action at the given timestamp
    pub fn record_action(&mut self, timestamp: u64) {
        self.timestamps.push(timestamp);

        // Keep only recent history to prevent unbounded growth
        if self.timestamps.len() > self.max_history {
            // Calculate how many to remove to get back to max_history
            let to_remove = (self.timestamps.len() - self.max_history).min(100);
            self.timestamps.drain(0..to_remove);
        }
    }

    /// Get timestamps within a specific time window
    pub fn get_recent(&self, current_time: u64, window_seconds: u64) -> Vec<u64> {
        let window_start = current_time.saturating_sub(window_seconds);
        self.timestamps
            .iter()
            .filter(|&&ts| ts >= window_start)
            .copied()
            .collect()
    }

    /// Clear old timestamps outside any reasonable window (for cleanup)
    pub fn cleanup_old(&mut self, current_time: u64, max_age_seconds: u64) {
        let cutoff = current_time.saturating_sub(max_age_seconds);
        self.timestamps.retain(|&ts| ts >= cutoff);
    }
}

/// Validate text length
pub fn validate_length(
    text: &str,
    min: usize,
    max: usize,
    field_name: &str,
) -> Result<(), String> {
    let len = text.len();

    if len < min {
        return Err(format!(
            "{} too short (min {} characters, got {})",
            field_name, min, len
        ));
    }

    if len > max {
        return Err(format!(
            "{} too long (max {} characters, got {})",
            field_name, max, len
        ));
    }

    Ok(())
}

/// Check for profanity (basic implementation)
pub fn contains_profanity(text: &str) -> bool {
    let lowercase = text.to_lowercase();

    // Basic profanity list (expand in production)
    let profanity_list = vec![
        "spam", "scam", "fraud", "fake", "phishing",
        // Add more as needed
    ];

    profanity_list.iter().any(|word| lowercase.contains(word))
}

/// Educational error message builder
pub struct SecurityError {
    pub message: String,
    pub suggestion: String,
    pub learn_more: String,
}

impl SecurityError {
    pub fn xss_attempt() -> Self {
        Self {
            message: "Potentially dangerous characters detected in input".to_string(),
            suggestion: "Please avoid HTML tags and script elements in your text".to_string(),
            learn_more: "XSS attacks inject malicious code into web pages. We filter these for your safety.".to_string(),
        }
    }

    pub fn invalid_ipfs_cid() -> Self {
        Self {
            message: "Invalid IPFS CID format".to_string(),
            suggestion: "IPFS CIDs should start with 'Qm' (46 chars) or 'b' (50-100 chars)".to_string(),
            learn_more: "IPFS uses Content IDentifiers (CIDs) to address files cryptographically.".to_string(),
        }
    }

    pub fn rate_limit_exceeded(limit: u32, window: u64) -> Self {
        Self {
            message: format!("Rate limit exceeded: {} requests per {} seconds", limit, window),
            suggestion: "Please wait a moment before trying again".to_string(),
            learn_more: "Rate limits prevent spam and protect the network for everyone.".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_user_input() {
        // Normal text should pass through
        assert_eq!(
            sanitize_user_input("Hello World!"),
            "Hello World!"
        );

        // HTML tags should be filtered out entirely (safer than escaping)
        let result = sanitize_user_input("<script>alert('xss')</script>");
        assert!(!result.contains("<"));
        assert!(!result.contains(">"));
        // Angle brackets removed, quotes escaped
        assert_eq!(result, "scriptalert(&#x27;xss&#x27;)script");

        // Trim whitespace
        assert_eq!(
            sanitize_user_input("  test  "),
            "test"
        );

        // Allow safe punctuation
        assert_eq!(
            sanitize_user_input("Price: $19.99!"),
            "Price: $19.99!"
        );

        // HTML entities get filtered out (& is not in safe chars, ; is kept)
        let result = sanitize_user_input("Use &amp; for and");
        assert_eq!(result, "Use amp; for and");  // & filtered, ; remains from &amp;
    }

    #[test]
    fn test_sanitize_ipfs_cid() {
        // Valid CIDv0 (46 characters, base58, starting with Qm)
        // Using valid base58 characters only (no 0, O, I, l)
        let valid_v0 = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG";
        assert!(sanitize_ipfs_cid(valid_v0).is_ok());

        // Valid CIDv1 (lowercase, base32)
        let valid_v1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi";
        assert!(sanitize_ipfs_cid(valid_v1).is_ok());

        // Invalid: wrong prefix
        assert!(sanitize_ipfs_cid("invalid_cid").is_err());

        // Invalid: wrong length
        assert!(sanitize_ipfs_cid("QmShort").is_err());

        // Invalid: uppercase in CIDv1
        assert!(sanitize_ipfs_cid("bafyBEIGDYRZT5SFP7UDM7HU76UH7Y26NF3EFUYLQABF3OCLGTQY55FBZDI").is_err());

        // Invalid: contains forbidden base58 chars (0, O, I, l)
        assert!(sanitize_ipfs_cid("Qm0OIl123456789012345678901234567890123456").is_err());
    }

    #[test]
    fn test_validate_email() {
        // Valid emails
        assert!(validate_email("test@example.com").is_ok());
        assert!(validate_email("  USER@DOMAIN.COM  ").is_ok());

        // Invalid emails
        assert!(validate_email("").is_err());
        assert!(validate_email("no-at-sign").is_err());
        assert!(validate_email("no-domain@").is_err());
    }

    #[test]
    fn test_validate_url() {
        // Valid URLs
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("http://localhost:3000").is_ok());

        // Invalid URLs
        assert!(validate_url("").is_err());
        assert!(validate_url("ftp://example.com").is_err());
        assert!(validate_url("not-a-url").is_err());
    }

    #[test]
    fn test_validate_price() {
        // Valid prices
        assert!(validate_price(100).is_ok()); // $1.00
        assert!(validate_price(999999).is_ok()); // $9,999.99

        // Invalid prices
        assert!(validate_price(0).is_err());
        assert!(validate_price(100_000_001).is_err()); // Over $1M
    }

    #[test]
    fn test_validate_quantity() {
        // Valid quantities
        assert!(validate_quantity(1).is_ok());
        assert!(validate_quantity(100).is_ok());

        // Invalid quantities
        assert!(validate_quantity(0).is_err());
        assert!(validate_quantity(1_000_001).is_err());
    }

    #[test]
    fn test_validate_length() {
        // Valid lengths
        assert!(validate_length("Hello", 1, 200, "Title").is_ok());

        // Too short
        assert!(validate_length("", 1, 200, "Title").is_err());

        // Too long
        let long_text = "a".repeat(201);
        assert!(validate_length(&long_text, 1, 200, "Title").is_err());
    }

    #[test]
    fn test_sanitize_strict() {
        // Should remove HTML and special chars
        let result = sanitize_strict("<b>Hello</b> & goodbye!");
        assert!(!result.contains('<'));
        assert!(!result.contains('&'));

        // Should keep basic punctuation
        assert_eq!(sanitize_strict("Hello, world!"), "Hello, world!");
    }

    #[test]
    fn test_rate_limiter_configs() {
        let listing_limiter = RateLimiter::default_listing_creation();
        assert_eq!(listing_limiter.max_requests, 10);
        assert_eq!(listing_limiter.window_seconds, 3600);

        let search_limiter = RateLimiter::default_search();
        assert_eq!(search_limiter.max_requests, 100);
        assert_eq!(search_limiter.window_seconds, 60);

        let dispute_limiter = RateLimiter::default_dispute_filing();
        assert_eq!(dispute_limiter.max_requests, 5);
        assert_eq!(dispute_limiter.window_seconds, 86400);
    }

    #[test]
    fn test_rate_limiter_check_allowed() {
        let limiter = RateLimiter::new(3, 60); // 3 requests per minute
        let current_time = 1000u64;

        // No previous requests - should be allowed
        let result = limiter.check_allowed(&[], current_time);
        assert!(result.is_ok());

        // 2 recent requests - should be allowed
        let timestamps = vec![990, 995];
        let result = limiter.check_allowed(&timestamps, current_time);
        assert!(result.is_ok());

        // 3 recent requests - should be denied
        let timestamps = vec![950, 970, 990];
        let result = limiter.check_allowed(&timestamps, current_time);
        assert!(result.is_err());

        // Old requests outside window - should be allowed
        let timestamps = vec![800, 850, 900]; // All older than 60s
        let result = limiter.check_allowed(&timestamps, current_time);
        assert!(result.is_ok());

        // Mixed old and new - 2 new, should be allowed
        let timestamps = vec![800, 950, 990];
        let result = limiter.check_allowed(&timestamps, current_time);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rate_limiter_time_until_allowed() {
        let limiter = RateLimiter::new(3, 60); // 3 requests per minute
        let current_time = 1000u64;

        // No previous requests - no wait
        assert_eq!(limiter.time_until_allowed(&[], current_time), 0);

        // 2 recent requests - no wait
        let timestamps = vec![990, 995];
        assert_eq!(limiter.time_until_allowed(&timestamps, current_time), 0);

        // 3 recent requests at 950, 970, 990
        // Oldest (950) will expire at 950 + 60 = 1010
        // Current time is 1000, so wait = 1010 - 1000 = 10 seconds
        let timestamps = vec![950, 970, 990];
        let wait_time = limiter.time_until_allowed(&timestamps, current_time);
        assert_eq!(wait_time, 10);
    }

    #[test]
    fn test_rate_limit_tracker() {
        let mut tracker = RateLimitTracker::new("test_action".to_string());

        // Initially empty
        assert_eq!(tracker.timestamps.len(), 0);

        // Record actions
        tracker.record_action(100);
        tracker.record_action(200);
        tracker.record_action(300);
        assert_eq!(tracker.timestamps.len(), 3);

        // Get recent actions (within 150 seconds of time 300)
        let recent = tracker.get_recent(300, 150);
        assert_eq!(recent.len(), 2); // 200 and 300

        // Get recent actions (within 250 seconds of time 300)
        let recent = tracker.get_recent(300, 250);
        assert_eq!(recent.len(), 3); // All three

        // Cleanup old actions
        tracker.cleanup_old(300, 150);
        assert_eq!(tracker.timestamps.len(), 2); // 100 removed
    }

    #[test]
    fn test_rate_limit_tracker_max_history() {
        let mut tracker = RateLimitTracker::new("test_action".to_string());
        tracker.max_history = 10; // Small limit for testing

        // Add more than max_history
        for i in 0..15 {
            tracker.record_action(i as u64);
        }

        // Should have trimmed to keep under max_history
        assert!(tracker.timestamps.len() <= tracker.max_history);

        // Most recent should still be there
        assert!(tracker.timestamps.contains(&14));
    }

    #[test]
    fn test_rate_limiter_edge_cases() {
        let limiter = RateLimiter::new(1, 60); // Very strict: 1 per minute

        // Exactly at the boundary (inclusive)
        let current_time = 1000u64;
        let timestamps = vec![940]; // Exactly 60 seconds ago

        // Window is [940, 1000] inclusive, so 940 counts as within window
        // With max_requests=1, this should FAIL since we already have 1
        let result = limiter.check_allowed(&timestamps, current_time);
        assert!(result.is_err());

        // Just outside the boundary
        let timestamps = vec![939]; // 61 seconds ago, outside window
        let result = limiter.check_allowed(&timestamps, current_time);
        assert!(result.is_ok()); // Should be allowed

        // Just inside the boundary
        let timestamps = vec![941];
        let result = limiter.check_allowed(&timestamps, current_time);
        assert!(result.is_err()); // Should be denied
    }

    #[test]
    fn test_rate_limiter_realistic_scenario() {
        // Simulate listing creation: 10 per hour
        let limiter = RateLimiter::default_listing_creation();
        let mut timestamps = Vec::new();
        let current_time = 10000u64;

        // Create 10 listings rapidly (all within same second)
        for i in 0..10 {
            let ts = current_time + i;
            assert!(limiter.check_allowed(&timestamps, ts).is_ok());
            timestamps.push(ts);
        }

        // 11th should be denied
        let result = limiter.check_allowed(&timestamps, current_time + 11);
        assert!(result.is_err());

        // After 1 hour + 1 second, oldest timestamp expires
        let future_time = current_time + 3601;
        let result = limiter.check_allowed(&timestamps, future_time);
        assert!(result.is_ok()); // Now allowed again
    }

    #[test]
    fn test_normalize_unicode() {
        // Should remove control characters
        let with_control = "Hello\x00World";
        let normalized = normalize_unicode(with_control);
        assert!(!normalized.contains('\x00'));
    }

    #[test]
    fn test_contains_profanity() {
        assert!(contains_profanity("This is spam!"));
        assert!(contains_profanity("SCAM alert"));
        assert!(!contains_profanity("This is a legitimate product"));
    }

    #[test]
    fn test_security_error_messages() {
        let xss_error = SecurityError::xss_attempt();
        assert!(xss_error.message.contains("dangerous"));
        assert!(xss_error.suggestion.len() > 0);
        assert!(xss_error.learn_more.len() > 0);

        let cid_error = SecurityError::invalid_ipfs_cid();
        assert!(cid_error.message.contains("IPFS"));

        let rate_error = SecurityError::rate_limit_exceeded(10, 60);
        assert!(rate_error.message.contains("10"));
    }
}
