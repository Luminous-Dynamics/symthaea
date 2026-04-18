// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Replay-protection sliding window for AEAD-sealed streams.
//!
//! ## Why
//!
//! ChaCha20-Poly1305 + monotonic nonce prevents *encryption* reuse (the
//! sender will never produce two ciphertexts with the same nonce under the
//! same key), but it does NOT prevent a network attacker from capturing a
//! sealed envelope and replaying it later — the receiver will accept it
//! because AEAD verification still succeeds against the original key.
//!
//! For RDP frames (idempotent screen updates) replay is mostly cosmetic. For
//! `InputFrame` reverse-path messages, replay is a real security hole: a
//! captured `tap (504, 1122)` could be re-fired to re-execute the action.
//!
//! ## Design
//!
//! Sliding window over received sequence numbers, keyed by `(source_id,
//! payload_type)`. Window size is 64 bits — the receiver tracks the highest
//! sequence number seen so far plus a 64-bit bitmap of the most recent 64
//! sequences. A sequence is accepted iff:
//!
//! 1. It is strictly higher than the highest-seen-so-far (advance the
//!    window), OR
//! 2. It falls within the bitmap range AND the corresponding bit is unset
//!    (mark the bit, accept the message).
//!
//! Sequences that are too old (more than 64 below the highest-seen) are
//! rejected outright. Duplicates within the bitmap range are rejected.
//!
//! ## Wraparound
//!
//! u64 sequence space is effectively unbounded — at 30 fps RDP this wraps
//! in ~19 billion years. The implementation does not handle wraparound
//! specifically because real session lifetime (governed by key rotation)
//! is many orders of magnitude shorter.
//!
//! ## Multi-stream isolation
//!
//! Different `(source_id, payload_type)` tuples have independent windows.
//! This is required because `RdpFrame` (payload_type=0x10) and `InputFrame`
//! (payload_type=0x11) share a session key but maintain independent
//! sequence counters via `RdpSession::next_nonce()`. Replay protection per
//! tuple, not per session.
//!
//! ## Future use
//!
//! Phase I.A.5 ships this primitive specifically for `RdpSession::open` (RDP
//! reverse path). Mesh layer (`swarm::mesh::peer_registry`) is **equally
//! vulnerable** to replay (verified by exploration during Phase I.A.5
//! planning) and should adopt this primitive in a follow-up.

#![cfg(feature = "mesh-encryption")]

use std::collections::HashMap;

/// Width of the bitmap tracking recent sequence numbers (in bits).
///
/// 64 bits is the standard IPsec/DTLS replay window width and is large
/// enough to absorb realistic out-of-order delivery on a WebSocket while
/// being cheap to update (single u64 shift + OR).
pub const WINDOW_BITS: u64 = 64;

/// Per-stream replay state: highest sequence seen + bitmap of the most
/// recent `WINDOW_BITS` sequences.
#[derive(Debug, Clone)]
struct StreamWindow {
    /// Highest sequence number seen so far. The bitmap tracks
    /// `[highest - WINDOW_BITS + 1, highest]`. Bit position 0 = highest,
    /// bit position WINDOW_BITS-1 = oldest.
    highest: u64,
    /// 64-bit bitmap of received sequences. Bit i corresponds to sequence
    /// `highest - i`. Bit 0 (LSB) is always set if `highest` was seen
    /// (which it always is once the window is initialized).
    bitmap: u64,
    /// Whether this window has seen any sequence yet.
    initialized: bool,
}

impl Default for StreamWindow {
    fn default() -> Self {
        Self {
            highest: 0,
            bitmap: 0,
            initialized: false,
        }
    }
}

/// Sliding-window replay protection for multiple independent streams.
///
/// Streams are keyed by `(source_id, payload_type)`. Use `accept(key, seq)`
/// to atomically check-and-mark a sequence as received.
#[derive(Debug, Default, Clone)]
pub struct ReplayWindow {
    streams: HashMap<(u64, u8), StreamWindow>,
}

impl ReplayWindow {
    /// Create an empty replay window with no tracked streams.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all tracked streams. Useful after rekey when sequence counters
    /// also reset.
    pub fn clear(&mut self) {
        self.streams.clear();
    }

    /// Atomically check whether `seq` is acceptable for the given stream
    /// key, and mark it as received if so.
    ///
    /// Returns `true` if the message should be processed (sequence is new
    /// and within the window), `false` if it should be dropped (duplicate
    /// or too old).
    ///
    /// `key` is typically `(source_id_high_64, payload_type)` where
    /// `source_id_high_64` is the high 64 bits of the AEAD source identifier
    /// — for RDP this is the per-session random `source_id` interpreted as
    /// little-endian u64.
    pub fn accept(&mut self, key: (u64, u8), seq: u64) -> bool {
        let win = self.streams.entry(key).or_default();

        if !win.initialized {
            // First sequence for this stream: accept and initialize.
            win.highest = seq;
            win.bitmap = 1; // bit 0 = "seq seen"
            win.initialized = true;
            return true;
        }

        if seq > win.highest {
            // New high sequence: shift the bitmap.
            let shift = seq - win.highest;
            if shift >= WINDOW_BITS {
                // Window jumped entirely past the old bitmap.
                win.bitmap = 1;
            } else {
                // Shift left and set bit 0 for the new highest.
                win.bitmap = (win.bitmap << shift) | 1;
            }
            win.highest = seq;
            true
        } else {
            // seq <= highest: check if it falls within the window and is unseen.
            let offset = win.highest - seq;
            if offset >= WINDOW_BITS {
                // Too old.
                false
            } else {
                let mask = 1u64 << offset;
                if win.bitmap & mask != 0 {
                    // Already seen.
                    false
                } else {
                    // In window, unseen → accept and mark.
                    win.bitmap |= mask;
                    true
                }
            }
        }
    }

    /// Number of distinct streams currently tracked. Mostly for tests and
    /// observability; not part of the protection guarantee.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(payload_type: u8) -> (u64, u8) {
        (0xDEADBEEF_CAFEBABE, payload_type)
    }

    #[test]
    fn first_sequence_accepted() {
        let mut w = ReplayWindow::new();
        assert!(w.accept(key(0x10), 0));
    }

    #[test]
    fn sequential_sequences_accepted() {
        let mut w = ReplayWindow::new();
        for seq in 0..100 {
            assert!(w.accept(key(0x10), seq), "seq {seq} should accept");
        }
    }

    #[test]
    fn duplicate_at_highest_rejected() {
        let mut w = ReplayWindow::new();
        assert!(w.accept(key(0x10), 5));
        assert!(!w.accept(key(0x10), 5), "duplicate at highest should reject");
    }

    #[test]
    fn duplicate_within_window_rejected() {
        let mut w = ReplayWindow::new();
        // Establish window: 0, 1, 2, 3, 4, 5
        for seq in 0..=5 {
            assert!(w.accept(key(0x10), seq));
        }
        // Replay an old one within the window.
        assert!(!w.accept(key(0x10), 2));
        // But future sequences still work.
        assert!(w.accept(key(0x10), 6));
    }

    #[test]
    fn out_of_order_within_window_accepted() {
        let mut w = ReplayWindow::new();
        // Out-of-order arrival within window: highest=10, then 7 arrives
        // (offset 3, in window).
        assert!(w.accept(key(0x10), 10));
        assert!(w.accept(key(0x10), 7));
        // 7 again is now a duplicate.
        assert!(!w.accept(key(0x10), 7));
        // 8 still works.
        assert!(w.accept(key(0x10), 8));
    }

    #[test]
    fn too_old_sequence_rejected() {
        let mut w = ReplayWindow::new();
        // Establish window at 100.
        assert!(w.accept(key(0x10), 100));
        // 35 is 65 below highest, outside the 64-bit window.
        assert!(!w.accept(key(0x10), 35));
        // 36 is 64 below highest, also outside (offset >= WINDOW_BITS).
        assert!(!w.accept(key(0x10), 36));
        // 37 is 63 below highest, just inside.
        assert!(w.accept(key(0x10), 37));
    }

    #[test]
    fn future_arrival_shifts_window_correctly() {
        let mut w = ReplayWindow::new();
        // Establish: 0 .. 5
        for seq in 0..=5 {
            assert!(w.accept(key(0x10), seq));
        }
        // Big jump forward.
        assert!(w.accept(key(0x10), 1000));
        // Replay of 0..5 should now all be too old.
        for seq in 0..=5 {
            assert!(!w.accept(key(0x10), seq), "old seq {seq} after jump should reject");
        }
        // Sequences just below 1000 in the new window are still acceptable
        // (they were never seen).
        assert!(w.accept(key(0x10), 999));
        assert!(w.accept(key(0x10), 950));
        // 936 = 1000 - 64 → outside window.
        assert!(!w.accept(key(0x10), 936));
    }

    #[test]
    fn independent_streams_dont_interfere() {
        let mut w = ReplayWindow::new();
        // Two streams with the same source_id but different payload_type
        // should have independent windows.
        assert!(w.accept(key(0x10), 5));
        assert!(w.accept(key(0x11), 5)); // same seq, different stream → ok
        assert!(!w.accept(key(0x10), 5)); // duplicate on 0x10
        assert!(!w.accept(key(0x11), 5)); // duplicate on 0x11
        assert_eq!(w.stream_count(), 2);
    }

    #[test]
    fn different_source_ids_dont_interfere() {
        let mut w = ReplayWindow::new();
        let k1 = (0xAAAAAAAA_AAAAAAAA, 0x10);
        let k2 = (0xBBBBBBBB_BBBBBBBB, 0x10);
        assert!(w.accept(k1, 100));
        assert!(w.accept(k2, 100)); // same seq, different source → ok
        assert!(!w.accept(k1, 100)); // duplicate on k1
        assert_eq!(w.stream_count(), 2);
    }

    #[test]
    fn window_edge_exactly_64_below_rejected() {
        let mut w = ReplayWindow::new();
        assert!(w.accept(key(0x10), 100));
        // Offset 64 = exactly at WINDOW_BITS → rejected (offset >= WINDOW_BITS).
        assert!(!w.accept(key(0x10), 36));
        // Offset 63 = inside.
        assert!(w.accept(key(0x10), 37));
    }

    #[test]
    fn clear_resets_all_streams() {
        let mut w = ReplayWindow::new();
        assert!(w.accept(key(0x10), 5));
        assert!(w.accept(key(0x11), 7));
        assert_eq!(w.stream_count(), 2);
        w.clear();
        assert_eq!(w.stream_count(), 0);
        // After clear, same sequences are accepted again as fresh.
        assert!(w.accept(key(0x10), 5));
        assert!(w.accept(key(0x11), 7));
    }
}
