//! HDC encoder for `LogEvent` → 16,384D hypervector.
//!
//! ## Phase 1 spike strategy
//!
//! We deliberately do NOT link to `symthaea-core` during the spike — that
//! pulls the full consciousness pipeline and makes iteration slow. Instead we
//! use a local reference implementation of MAP-I style binary/bipolar HDC with
//! the same dimensionality (16,384) and the same binding operators (XOR-style)
//! that `symthaea-core/src/hdc/binary_hv.rs` uses, so results transfer.
//!
//! When the `hdc-encoder` feature is enabled, this module exposes a re-export
//! point where the production `symthaea-core` encoder can be swapped in. This
//! is intentional: the spike measures whether the *approach* works. Once
//! we prove it, we swap the reference impl for the real one and re-run.
//!
//! ## What gets encoded
//!
//! Five role-filler bindings, then bundled (majority):
//!   SOURCE     ⊗ source_hv
//!   SEVERITY   ⊗ severity_hv
//!   PROVIDER   ⊗ hash_hv(provider)
//!   EVENT_ID   ⊗ hash_hv(event_id)
//!   COMPONENT  ⊗ hash_hv(component)
//!   FIELDS     ⊗ bundle(k_hv ⊗ v_hv for each field)
//!
//! This is the standard VSA "record" encoding. Token vocabulary is generated
//! deterministically from a seed so the same string → same hypervector across
//! runs, which is essential for reproducibility on the benchmark corpus.

use crate::event::{LogEvent, Severity, Source};
use std::sync::OnceLock;

/// Hypervector dimensionality. Matches `symthaea-core` for direct comparison.
pub const HDC_DIM: usize = 16_384;

/// Bipolar hypervector: each element is -1 or +1, stored as i8 for cache.
pub type Hdv = Vec<i8>;

/// Deterministic xorshift64* PRNG for reproducible codebook generation.
/// We do not use rand::thread_rng — every run must produce identical vectors.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

/// Generate a random bipolar hypervector from a 64-bit seed.
pub fn random_hv(seed: u64) -> Hdv {
    let mut state = seed.max(1);
    let mut hv = vec![0i8; HDC_DIM];
    for chunk in hv.chunks_mut(64) {
        let bits = xorshift(&mut state);
        for (i, slot) in chunk.iter_mut().enumerate() {
            *slot = if (bits >> i) & 1 == 1 { 1 } else { -1 };
        }
    }
    hv
}

/// FNV-1a string hash → u64 seed. Deterministic.
fn seed_for(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Per-string hypervector with interning. Thread-local is fine for the spike.
fn hash_hv(s: &str) -> Hdv {
    random_hv(seed_for(s))
}

/// Fixed role hypervectors. Seeds are constants so they never collide with
/// content-derived vectors.
fn role_source() -> &'static Hdv {
    static R: OnceLock<Hdv> = OnceLock::new();
    R.get_or_init(|| random_hv(0xA1A1_A1A1_A1A1_A1A1))
}
fn role_severity() -> &'static Hdv {
    static R: OnceLock<Hdv> = OnceLock::new();
    R.get_or_init(|| random_hv(0xB2B2_B2B2_B2B2_B2B2))
}
fn role_provider() -> &'static Hdv {
    static R: OnceLock<Hdv> = OnceLock::new();
    R.get_or_init(|| random_hv(0xC3C3_C3C3_C3C3_C3C3))
}
fn role_event_id() -> &'static Hdv {
    static R: OnceLock<Hdv> = OnceLock::new();
    R.get_or_init(|| random_hv(0xD4D4_D4D4_D4D4_D4D4))
}
fn role_component() -> &'static Hdv {
    static R: OnceLock<Hdv> = OnceLock::new();
    R.get_or_init(|| random_hv(0xE5E5_E5E5_E5E5_E5E5))
}
fn role_fields() -> &'static Hdv {
    static R: OnceLock<Hdv> = OnceLock::new();
    R.get_or_init(|| random_hv(0xF6F6_F6F6_F6F6_F6F6))
}

fn source_hv(s: Source) -> Hdv {
    hash_hv(&format!("SOURCE::{s:?}"))
}
fn severity_hv(s: Severity) -> Hdv {
    hash_hv(&format!("SEVERITY::{s:?}"))
}

/// Bind = elementwise multiply (XOR for bipolar {-1,+1}). Non-commutative
/// variant uses cyclic shift on `a` first; we don't need that here because
/// role-filler binding in VSA is intentionally commutative.
pub fn bind(a: &Hdv, b: &Hdv) -> Hdv {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Bundle = elementwise majority. Takes a slice of HVs.
pub fn bundle(hvs: &[Hdv]) -> Hdv {
    if hvs.is_empty() {
        return vec![0; HDC_DIM];
    }
    let mut acc = vec![0i32; HDC_DIM];
    for hv in hvs {
        for (i, &v) in hv.iter().enumerate() {
            acc[i] += v as i32;
        }
    }
    acc.into_iter()
        .map(|s| {
            if s > 0 {
                1
            } else if s < 0 {
                -1
            } else {
                1
            }
        })
        .collect()
}

/// Encode a `LogEvent` to a 16,384D hypervector.
pub fn encode(event: &LogEvent) -> Hdv {
    let mut parts: Vec<Hdv> = Vec::with_capacity(6);

    parts.push(bind(role_source(), &source_hv(event.source)));
    parts.push(bind(role_severity(), &severity_hv(event.severity)));
    parts.push(bind(role_provider(), &hash_hv(&event.provider)));
    parts.push(bind(
        role_event_id(),
        &hash_hv(&format!("EVT::{}", event.event_id)),
    ));
    parts.push(bind(role_component(), &hash_hv(&event.component)));

    // Fields: bundle of k⊗v bindings.
    if !event.fields.is_empty() {
        let kv: Vec<Hdv> = event
            .fields
            .iter()
            .map(|(k, v)| bind(&hash_hv(k), &hash_hv(v)))
            .collect();
        let fields_bundle = bundle(&kv);
        parts.push(bind(role_fields(), &fields_bundle));
    }

    bundle(&parts)
}

/// Cosine similarity on bipolar HVs. Range [-1, 1].
pub fn cosine(a: &Hdv, b: &Hdv) -> f32 {
    let dot: i32 = a.iter().zip(b.iter()).map(|(x, y)| (x * y) as i32).sum();
    dot as f32 / HDC_DIM as f32
}

/// Diagnostic: how many distinct strings we've seen this session. Used by the
/// benchmark example to sanity-check that the codebook is stable.
#[allow(dead_code)]
fn _codebook_size_stub() -> usize {
    // Intentional: codebook is deterministic via hash, not interned. Stub
    // kept so the spike script can call it without a cfg gate.
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{LogEvent, Severity, Source};
    use chrono::Utc;
    use std::collections::BTreeMap;

    fn ev(provider: &str, event_id: u32, msg: &str) -> LogEvent {
        LogEvent {
            timestamp: Utc::now(),
            source: Source::WindowsEvent,
            severity: Severity::Error,
            component: "Test".into(),
            provider: provider.into(),
            event_id,
            message: msg.into(),
            fields: BTreeMap::new(),
            host: None,
            label: None,
        }
    }

    #[test]
    fn deterministic_encoding() {
        let a = encode(&ev("Foo", 42, "hello"));
        let b = encode(&ev("Foo", 42, "hello"));
        assert_eq!(cosine(&a, &b), 1.0);
    }

    #[test]
    fn different_events_diverge() {
        let a = encode(&ev("Foo", 42, "hello"));
        let b = encode(&ev("Bar", 99, "world"));
        let sim = cosine(&a, &b);
        assert!(
            sim < 0.5,
            "unrelated events should not be near-duplicates, got {sim}"
        );
    }

    #[test]
    fn same_provider_more_similar_than_different() {
        let base = encode(&ev("Foo", 42, "a"));
        let same_prov = encode(&ev("Foo", 43, "b"));
        let diff_prov = encode(&ev("Bar", 42, "a"));
        assert!(cosine(&base, &same_prov) > cosine(&base, &diff_prov));
    }
}
