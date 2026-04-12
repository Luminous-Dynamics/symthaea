// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Clipboard synchronization for Sovereign RDP.
//!
//! Bidirectional clipboard sync via `ControlMessage::ClipboardUpdate`.
//! Consciousness-gated: disabled in ViewOnly mode.
//! Privacy-scrubbed before transmission.

use super::rdp_protocol_ext::ClipboardFormat;
use super::rdp_session::SessionPermission;

/// Maximum clipboard payload size (10 MB).
pub const CLIPBOARD_MAX_BYTES: usize = 10 * 1024 * 1024;

/// Clipboard sync state tracker.
pub struct ClipboardSync {
    /// Hash of last sent clipboard content (avoid redundant sends).
    last_sent_hash: [u8; 32],
    /// Hash of last received clipboard content.
    last_received_hash: [u8; 32],
    /// Whether sync is enabled.
    enabled: bool,
}

impl ClipboardSync {
    pub fn new() -> Self {
        Self {
            last_sent_hash: [0u8; 32],
            last_received_hash: [0u8; 32],
            enabled: true,
        }
    }

    /// Check if clipboard content has changed and should be sent.
    ///
    /// Returns a `ControlMessage::ClipboardUpdate` if content is new,
    /// or `None` if unchanged or blocked.
    /// Returns `(format, data)` tuple for the caller to wrap in a transport message.
    pub fn prepare_update(
        &mut self,
        content: &[u8],
        format: ClipboardFormat,
        permission: SessionPermission,
    ) -> Option<(ClipboardFormat, Vec<u8>)> {
        // Consciousness gate: require Interactive or Full.
        if permission < SessionPermission::Interactive {
            return None;
        }

        if !self.enabled {
            return None;
        }

        // Size limit.
        if content.len() > CLIPBOARD_MAX_BYTES {
            return None;
        }

        // Dedup: don't send if unchanged.
        let hash = *blake3::hash(content).as_bytes();
        if hash == self.last_sent_hash {
            return None;
        }

        self.last_sent_hash = hash;

        Some((format, content.to_vec()))
    }

    /// Process a received clipboard update.
    ///
    /// Returns the clipboard content if it's new (not a reflection of what we sent).
    pub fn receive_update(&mut self, data: &[u8], _format: &ClipboardFormat) -> Option<Vec<u8>> {
        if data.len() > CLIPBOARD_MAX_BYTES {
            return None;
        }

        let hash = *blake3::hash(data).as_bytes();

        // Skip if this is a reflection of our own last send.
        if hash == self.last_sent_hash {
            return None;
        }

        // Skip if we already received this exact content.
        if hash == self.last_received_hash {
            return None;
        }

        self.last_received_hash = hash;
        Some(data.to_vec())
    }

    /// Enable or disable clipboard sync.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether sync is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for ClipboardSync {
    fn default() -> Self {
        Self::new()
    }
}

/// Scrub sensitive patterns from clipboard text before transmission.
///
/// Removes common secret patterns: API keys, passwords, private keys.
pub fn scrub_clipboard_text(text: &str) -> String {
    let mut scrubbed = text.to_string();

    // Common secret patterns.
    let patterns = [
        // AWS keys
        (r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_KEY]"),
        // Generic API keys (long hex strings)
        (r"[a-fA-F0-9]{40,}", "[REDACTED_LONG_HEX]"),
        // Private key blocks
        (r"-----BEGIN[^-]*PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]"),
        // Bearer tokens
        (r"Bearer [a-zA-Z0-9._-]{20,}", "Bearer [REDACTED]"),
        // Password patterns
        (r"(?i)password\s*[:=]\s*\S+", "password: [REDACTED]"),
    ];

    for (pattern, replacement) in &patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            scrubbed = re.replace_all(&scrubbed, *replacement).to_string();
        }
    }

    scrubbed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_prevents_redundant_sends() {
        let mut sync = ClipboardSync::new();
        let content = b"Hello, world!";

        // First send should work.
        let msg = sync.prepare_update(
            content,
            ClipboardFormat::PlainText,
            SessionPermission::Interactive,
        );
        assert!(msg.is_some());

        // Same content → no send.
        let msg = sync.prepare_update(
            content,
            ClipboardFormat::PlainText,
            SessionPermission::Interactive,
        );
        assert!(msg.is_none());

        // Different content → send.
        let msg = sync.prepare_update(
            b"Changed!",
            ClipboardFormat::PlainText,
            SessionPermission::Interactive,
        );
        assert!(msg.is_some());
    }

    #[test]
    fn test_view_only_blocks_clipboard() {
        let mut sync = ClipboardSync::new();
        let msg = sync.prepare_update(
            b"test",
            ClipboardFormat::PlainText,
            SessionPermission::ViewOnly,
        );
        assert!(msg.is_none());
    }

    #[test]
    fn test_size_limit() {
        let mut sync = ClipboardSync::new();
        let huge = vec![0u8; CLIPBOARD_MAX_BYTES + 1];
        let msg = sync.prepare_update(&huge, ClipboardFormat::PlainText, SessionPermission::Full);
        assert!(msg.is_none());
    }

    #[test]
    fn test_receive_dedup() {
        let mut sync = ClipboardSync::new();
        let data = b"received content";

        // First receive.
        let result = sync.receive_update(data, &ClipboardFormat::PlainText);
        assert!(result.is_some());

        // Same content again → skip.
        let result = sync.receive_update(data, &ClipboardFormat::PlainText);
        assert!(result.is_none());
    }

    #[test]
    fn test_scrub_clipboard_sensitive() {
        let text = "My password: secretpass123 and key AKIAIOSFODNN7EXAMPLE";
        let scrubbed = scrub_clipboard_text(text);
        assert!(!scrubbed.contains("secretpass123"));
        assert!(!scrubbed.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(scrubbed.contains("[REDACTED"));
    }

    #[test]
    fn test_scrub_preserves_normal_text() {
        let text = "Hello, this is normal clipboard content.";
        let scrubbed = scrub_clipboard_text(text);
        assert_eq!(scrubbed, text);
    }
}
