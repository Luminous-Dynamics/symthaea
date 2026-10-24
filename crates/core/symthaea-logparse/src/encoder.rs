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

/// Stable identifier for the experimental cross-corpus encoding contract.
/// Change this value when the namespace map, field canonicalization, or family semantics change.
pub const DOMAIN_INVARIANT_ENCODER_VERSION: &str = "cyber-logparse-domain-invariant-v1";

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

/// Coarse Windows event family used by the domain-invariant ablation.
///
/// This is intentionally an ablation, not a replacement for `encode`: the numeric EventID
/// is discarded after mapping known IDs into a small semantic family. Unknown IDs
/// share an `unknown` bucket. Numeric IDs can collide across providers, so this
/// heuristic must be evaluated cross-corpus before use in production.
fn normalized_source_token(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

/// Recognize a small number of provider namespaces without retaining their raw
/// names in the ablation vector. Unrecognized namespaces fail closed to unknown.
fn provider_family(provider: &str, component: &str) -> &'static str {
    let provider = normalized_source_token(provider);
    let component = normalized_source_token(component);

    let provider_hint = if provider.contains("sysmon") {
        Some("sysmon")
    } else if provider.contains("securityauditing") {
        Some("security_audit")
    } else if provider.contains("servicecontrolmanager") {
        Some("service_control_manager")
    } else if provider.contains("powershell") {
        Some("powershell")
    } else {
        None
    };

    let component_hint = if component.contains("sysmon") {
        Some("sysmon")
    } else if component.contains("powershell") {
        Some("powershell")
    } else if provider.is_empty() && component == "security" {
        Some("security_audit")
    } else {
        None
    };

    match (provider_hint, component_hint) {
        (Some(provider), Some(component)) if provider != component => "unknown",
        (Some(provider), _) => provider,
        (None, Some(component)) => component,
        (None, None) => "unknown",
    }
}

/// Coarse, provider-aware Windows event family used by the domain-invariant
/// ablation. The same numeric EventID in an unrelated namespace is not assumed
/// to have the same meaning. Provider/channel tokens select the namespace only;
/// raw provider/channel names are never encoded into the resulting vector.
fn event_family(provider: &str, component: &str, event_id: u32) -> &'static str {
    match provider_family(provider, component) {
        "sysmon" => match event_id {
            1 => "process",
            3 => "network",
            11 => "file",
            12 | 13 | 14 => "registry",
            _ => "unknown",
        },
        "security_audit" => match event_id {
            4624 | 4625 | 4634 | 4647 | 4648 | 4768 | 4769 | 4771 | 4776 => "authentication",
            4688 | 4689 => "process",
            4657 => "registry",
            4697 | 4698 | 4699 | 4700 | 4702 => "service_task",
            4656 | 4660 | 4663 => "object_access",
            5156 => "network",
            _ => "unknown",
        },
        "service_control_manager" => match event_id {
            7036 | 7040 | 7045 => "service_task",
            _ => "unknown",
        },
        "powershell" => match event_id {
            4103 | 4104 => "scripting",
            _ => "unknown",
        },
        _ => "unknown",
    }
}

/// Canonicalize common field-name aliases so casing/producer spelling differences
/// do not produce different key hypervectors. Unrecognized keys are normalized
/// to lowercase ASCII alphanumerics with underscores; values remain unchanged in
/// this first ablation to keep the experiment interpretable.
fn canonical_field_key(key: &str) -> String {
    let compact: String = key
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let canonical = match compact.as_str() {
        "targetusername" | "username" | "user" | "accountname" | "subjectusername"
        | "subjectuser" | "useraccount" => "identity_user",
        "ipaddress" | "sourceip" | "sourceaddress" | "clientaddress" | "remoteaddress" => {
            "network_address"
        }
        "destinationip" | "destinationaddress" | "destaddress" | "remoteip" => {
            "network_destination"
        }
        "logontype" | "logintype" => "authentication_logon_type",
        "authenticationpackagename" | "authpackage" => "authentication_package",
        "commandline" | "processcommandline" | "processcmdline" => "process_command_line",
        "image" | "newprocessname" | "processname" | "executable" | "processpath" => {
            "process_executable"
        }
        "parentimage" | "parentprocessname" | "parentprocess" => "process_parent",
        "targetfilename" | "filename" | "filepath" | "filepathname" => "file_path",
        "servicename" | "service" => "service_name",
        "status" | "substatus" | "statuscode" | "returncode" => "result_status",
        "processid" | "newprocessid" => "process_id",
        "parentprocessid" => "parent_process_id",
        "hash" | "hashes" | "filehash" => "file_hash",
        "objectname" | "registrykey" | "keyname" => "object_name",
        _ => {
            let normalized: String = key
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() {
                        c.to_ascii_lowercase()
                    } else {
                        '_'
                    }
                })
                .collect();
            let mut compacted = String::with_capacity(normalized.len());
            let mut previous_underscore = false;
            for ch in normalized.chars() {
                if ch == '_' {
                    if !previous_underscore {
                        compacted.push(ch);
                    }
                    previous_underscore = true;
                } else {
                    compacted.push(ch);
                    previous_underscore = false;
                }
            }
            return compacted.trim_matches('_').to_string();
        }
    };
    canonical.to_string()
}

/// Experimental domain-invariant encoder for cross-corpus ablation.
///
/// Compared with [`encode`], this version:
/// - drops raw provider and component strings;
/// - maps EventIDs into coarse families within recognized provider namespaces;
/// - canonicalizes common structured-field name aliases;
/// - retains source, severity, and field values as separate channels.
///
/// It is deliberately additive: existing `encode` behavior and historical
/// benchmark results remain reproducible. This is not yet an OCSF mapper and
/// does not normalize high-cardinality field values; cross-corpus gains must be
/// measured rather than presumed.
pub fn encode_domain_invariant(event: &LogEvent) -> Hdv {
    let mut parts: Vec<Hdv> = Vec::with_capacity(4);
    parts.push(bind(role_source(), &source_hv(event.source)));
    parts.push(bind(role_severity(), &severity_hv(event.severity)));
    parts.push(bind(
        role_event_id(),
        &hash_hv(&format!(
            "{}::EVENT_FAMILY::{}",
            DOMAIN_INVARIANT_ENCODER_VERSION,
            event_family(&event.provider, &event.component, event.event_id)
        )),
    ));

    if !event.fields.is_empty() {
        let kv: Vec<Hdv> = event
            .fields
            .iter()
            .map(|(key, value)| {
                bind(
                    &hash_hv(&format!(
                        "{}::FIELD::{}",
                        DOMAIN_INVARIANT_ENCODER_VERSION,
                        canonical_field_key(key)
                    )),
                    &hash_hv(value),
                )
            })
            .collect();
        parts.push(bind(role_fields(), &bundle(&kv)));
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
    fn domain_invariant_encoding_ignores_raw_provider_and_component() {
        let mut a = ev("Vendor-A-Provider", 4624, "ignored message");
        a.component = "Security".into();
        a.fields.insert("TargetUserName".into(), "alice".into());

        let mut b = ev("Vendor-B-Provider", 4624, "different ignored message");
        b.component = "OtherComponent".into();
        b.fields.insert("TargetUserName".into(), "alice".into());

        assert_ne!(encode(&a), encode(&b), "baseline must retain producer metadata");
        assert_eq!(encode_domain_invariant(&a), encode_domain_invariant(&b));
    }

    #[test]
    fn domain_invariant_encoding_canonicalizes_common_field_aliases() {
        let mut a = ev("provider-a", 4624, "");
        a.fields.insert("TargetUserName".into(), "alice".into());

        let mut b = ev("provider-b", 4624, "");
        b.fields.insert("user".into(), "alice".into());

        assert_eq!(encode_domain_invariant(&a), encode_domain_invariant(&b));
    }

    #[test]
    fn domain_invariant_encoding_buckets_known_event_id_families() {
        let a = encode_domain_invariant(&ev(
            "Microsoft-Windows-Security-Auditing",
            4624,
            "",
        ));
        let b = encode_domain_invariant(&ev(
            "Microsoft-Windows-Security-Auditing",
            4625,
            "",
        ));
        let c = encode_domain_invariant(&ev(
            "Microsoft-Windows-Security-Auditing",
            4688,
            "",
        ));

        assert_eq!(a, b, "security authentication IDs share a family");
        assert_ne!(a, c, "authentication and process families remain distinct");
    }

    #[test]
    fn domain_invariant_encoding_maps_equivalent_families_across_providers() {
        let sysmon_process = encode_domain_invariant(&ev(
            "Microsoft-Windows-Sysmon",
            1,
            "",
        ));
        let security_process = encode_domain_invariant(&ev(
            "Microsoft-Windows-Security-Auditing",
            4688,
            "",
        ));
        assert_eq!(sysmon_process, security_process);

        let sysmon_network = encode_domain_invariant(&ev(
            "Microsoft-Windows-Sysmon",
            3,
            "",
        ));
        let security_network = encode_domain_invariant(&ev(
            "Microsoft-Windows-Security-Auditing",
            5156,
            "",
        ));
        assert_eq!(sysmon_network, security_network);
    }

    #[test]
    fn domain_invariant_encoding_uses_security_channel_when_provider_is_missing() {
        let mut channel_only = ev("", 4624, "");
        channel_only.component = "Security".into();
        let provider_labeled = ev("Microsoft-Windows-Security-Auditing", 4625, "");
        assert_eq!(
            encode_domain_invariant(&channel_only),
            encode_domain_invariant(&provider_labeled)
        );
    }

    #[test]
    fn domain_invariant_encoding_rejects_conflicting_provider_and_channel_hints() {
        let conflicting = encode_domain_invariant(&ev(
            "Microsoft-Windows-Security-Auditing",
            4688,
            "",
        ));
        let unknown_namespace = encode_domain_invariant(&ev(
            "Vendor-Unknown-Provider",
            90001,
            "",
        ));
        let mut conflicting_event = ev(
            "Microsoft-Windows-Security-Auditing",
            4688,
            "",
        );
        conflicting_event.component = "Microsoft-Windows-Sysmon/Operational".into();
        assert_eq!(
            encode_domain_invariant(&conflicting_event),
            unknown_namespace,
            "contradictory provider and channel hints must fail closed"
        );
        assert_ne!(conflicting, unknown_namespace);
    }

    #[test]
    fn domain_invariant_encoding_does_not_conflate_event_ids_across_namespaces() {
        let sysmon_file = encode_domain_invariant(&ev(
            "Microsoft-Windows-Sysmon",
            11,
            "",
        ));
        let unknown_id_eleven = encode_domain_invariant(&ev(
            "Vendor-Unknown-Provider",
            11,
            "",
        ));
        assert_ne!(sysmon_file, unknown_id_eleven);
    }

    #[test]
    fn unknown_event_ids_do_not_leak_raw_numeric_id() {
        let a = encode_domain_invariant(&ev("Microsoft-Windows-Sysmon", 90001, ""));
        let b = encode_domain_invariant(&ev("Microsoft-Windows-Sysmon", 90002, ""));
        assert_eq!(a, b, "unmapped IDs intentionally share the unknown family");
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
