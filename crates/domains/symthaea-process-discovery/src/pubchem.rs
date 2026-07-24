// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PubChem PUG-REST cross-reference lookups.
//!
//! **This is the first network I/O anywhere in this crate.** Chosen
//! deliberately, not by default: Tristan was asked whether the first
//! Reaction Corpus Auditor pass should be fixture-only (no network) or
//! include live PubChem cross-referencing, and chose the latter. See
//! `PROCESS_DISCOVERY_THREAT_MODEL_2026-07-12.md` for the updated
//! assumptions this implies.
//!
//! **Bounded scope, by design**:
//! - Read-only `GET` to a single fixed public API (`pubchem.ncbi.nlm.nih.gov`),
//!   never a write, never arbitrary attacker-influenced URLs (the only
//!   caller-controlled input is a SMILES string, percent-encoded into a URL
//!   *path segment* via the `url` crate's own encoder, never string-formatted
//!   directly into a URL).
//! - Every outcome -- found, not-found, or any network/parse failure -- is
//!   advisory-only. Nothing in this crate's acceptance logic (`validity.rs`,
//!   `policy.rs`, `oracle.rs`) reads a `PubChemQueryOutcome`. A network
//!   failure is `Unavailable`, never treated as a rejection or as an
//!   automatic pass -- matches the same "advisory, not authoritative"
//!   pattern already established for the composition-stability estimate
//!   (`oracle.rs`).
//! - No caching/retry/backoff loop that could turn one audit run into
//!   unbounded request volume -- the audit harness (`audit.rs`) queries each
//!   distinct compound at most once per run and throttles between calls.
//! - A response over [`MAX_RESPONSE_BYTES`] is rejected as `Unavailable`
//!   rather than fully buffered -- checked against `Content-Length` when the
//!   server sends one, and against the actual body size after reading
//!   otherwise (real PubChem property responses are a few hundred bytes; a
//!   response this large is either a bug or something to not trust).
//!
//! **Source injection (Phase A.1 reproducibility hardening)**: real network
//! access always goes through [`PubChemSource`], never called directly by
//! `audit.rs`. This is what makes three separate guarantees possible without
//! duplicating lookup logic: [`LivePubChemSource`] (the real thing),
//! [`AlwaysUnavailableSource`] (fault injection -- proves a network failure
//! never changes a local verdict, see `audit.rs`'s parity test), and
//! `cache::ReplaySource` (deterministic offline replay from a frozen
//! fixture -- see `cache.rs`).

use reqwest::Url;
use serde::{Deserialize, Serialize};

const PUBCHEM_BASE: &str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug";
const PUBCHEM_PROPERTIES: &str = "MolecularFormula,ConnectivitySMILES,IUPACName";
const USER_AGENT: &str = "symthaea-process-discovery-reaction-corpus-auditor/0.1 (research use, see github.com/Luminous-Dynamics/symthaea)";
/// Real PubChem property responses for the compounds this crate looks up
/// are a few hundred bytes. 64KiB is generous headroom, not a tuned limit --
/// the point is *a* real cap exists, not that this exact number is load-bearing.
const MAX_RESPONSE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PubChemRecord {
    pub cid: u64,
    pub molecular_formula: String,
    pub connectivity_smiles: Option<String>,
    pub iupac_name: Option<String>,
}

/// Every case is a legitimate outcome for an advisory cross-reference, never
/// an error the caller must panic or bail on. `Unavailable` deliberately
/// carries a reason string but not a stack trace / raw error object -- it's
/// meant to end up in a human-readable audit report, not be pattern-matched
/// on for control flow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PubChemQueryOutcome {
    Found(PubChemRecord),
    NotFound,
    Unavailable(String),
}

/// Injection point for where a `PubChemQueryOutcome` comes from. Every
/// consumer (`audit.rs`) codes against this trait, never `lookup_by_smiles`
/// directly -- see the module doc's "Source injection" note.
pub trait PubChemSource {
    fn lookup(&self, smiles: &str) -> PubChemQueryOutcome;
}

/// The real thing: a live network call per `lookup`, via [`lookup_by_smiles`].
pub struct LivePubChemSource;

impl PubChemSource for LivePubChemSource {
    fn lookup(&self, smiles: &str) -> PubChemQueryOutcome {
        lookup_by_smiles(smiles)
    }
}

/// Fault injection: every lookup is `Unavailable`, unconditionally, as if
/// PubChem were completely unreachable for the whole run. Exists so
/// `audit.rs` can prove -- not just document -- that a total network
/// failure never changes which candidates get certified, rejected, or left
/// unclassified (see `audit.rs`'s `network_fault_never_changes_local_verdict`
/// test).
pub struct AlwaysUnavailableSource;

impl PubChemSource for AlwaysUnavailableSource {
    fn lookup(&self, _smiles: &str) -> PubChemQueryOutcome {
        PubChemQueryOutcome::Unavailable("fault-injected for testing: network unreachable".into())
    }
}

#[derive(Deserialize)]
struct PropertyTableResponse {
    #[serde(rename = "PropertyTable")]
    property_table: PropertyTable,
}

#[derive(Deserialize)]
struct PropertyTable {
    #[serde(rename = "Properties")]
    properties: Vec<PropertyEntry>,
}

#[derive(Deserialize)]
struct PropertyEntry {
    #[serde(rename = "CID")]
    cid: u64,
    #[serde(rename = "MolecularFormula")]
    molecular_formula: Option<String>,
    #[serde(rename = "ConnectivitySMILES")]
    connectivity_smiles: Option<String>,
    #[serde(rename = "IUPACName")]
    iupac_name: Option<String>,
}

fn parse_response(body: &str) -> PubChemQueryOutcome {
    match serde_json::from_str::<PropertyTableResponse>(body) {
        Ok(parsed) => match parsed.property_table.properties.into_iter().next() {
            Some(entry) => PubChemQueryOutcome::Found(PubChemRecord {
                cid: entry.cid,
                molecular_formula: entry.molecular_formula.unwrap_or_default(),
                connectivity_smiles: entry.connectivity_smiles,
                iupac_name: entry.iupac_name,
            }),
            None => PubChemQueryOutcome::NotFound,
        },
        Err(e) => PubChemQueryOutcome::Unavailable(format!("unexpected response shape: {e}")),
    }
}

/// Builds the PUG-REST URL for a SMILES property lookup. Uses `url`'s own
/// `path_segments_mut().push(...)`, which percent-encodes each segment
/// correctly for its position in the path -- the SMILES string is never
/// interpolated into a URL via `format!`.
fn build_smiles_property_url(smiles: &str) -> Result<Url, String> {
    let mut url = Url::parse(PUBCHEM_BASE).map_err(|e| e.to_string())?;
    url.path_segments_mut()
        .map_err(|_| "PUBCHEM_BASE is not a valid base URL".to_string())?
        .push("compound")
        .push("smiles")
        .push(smiles)
        .push("property")
        .push(PUBCHEM_PROPERTIES)
        .push("JSON");
    Ok(url)
}

/// Injectable HTTP layer: `(status_code, body)` on any response received at
/// all, `Err` only for a transport-level failure (couldn't connect, timed
/// out, etc.). Kept separate from `lookup_by_smiles` so unit tests can
/// supply a fixed response and never touch the network -- see the `tests`
/// module for the live, network-touching case, which is `#[ignore]`d.
///
/// Returns the raw response body alongside the parsed outcome (`None` only
/// when `fetch` itself returned `Err`, i.e. no body was ever received) --
/// `cache.rs` hashes this raw body for reproducibility bookkeeping, so it
/// has to be the literal bytes that came back, not a re-serialization of
/// the parsed `PubChemQueryOutcome`.
fn lookup_via<F: Fn(&str) -> Result<(u16, String), String>>(
    smiles: &str,
    fetch: F,
) -> (PubChemQueryOutcome, Option<String>) {
    let url = match build_smiles_property_url(smiles) {
        Ok(u) => u,
        Err(e) => return (PubChemQueryOutcome::Unavailable(e), None),
    };
    match fetch(url.as_str()) {
        Ok((200, body)) => (parse_response(&body), Some(body)),
        Ok((400, body)) | Ok((404, body)) => (PubChemQueryOutcome::NotFound, Some(body)),
        Ok((status, body)) => (
            PubChemQueryOutcome::Unavailable(format!(
                "HTTP {status}: {}",
                body.chars().take(200).collect::<String>()
            )),
            Some(body),
        ),
        Err(e) => (PubChemQueryOutcome::Unavailable(e), None),
    }
}

/// Pure, unit-testable half of the size-cap logic: given an optional
/// advertised `Content-Length` and the actual byte count read (0 if
/// rejected before reading), decide whether to proceed. Split out from
/// `real_http_get` specifically so this can be tested without a live
/// network call.
fn check_response_size(content_length: Option<usize>, actual_len: usize) -> Result<(), String> {
    if let Some(len) = content_length {
        if len > MAX_RESPONSE_BYTES {
            return Err(format!(
                "response Content-Length {len} exceeds the {MAX_RESPONSE_BYTES}-byte cap, refusing to read it"
            ));
        }
    }
    if actual_len > MAX_RESPONSE_BYTES {
        return Err(format!(
            "response body ({actual_len} bytes) exceeds the {MAX_RESPONSE_BYTES}-byte cap (no Content-Length header was present to reject it earlier)"
        ));
    }
    Ok(())
}

/// Single real-network entry point for this whole module -- both
/// `lookup_by_smiles` and `lookup_by_smiles_with_raw` (and therefore every
/// `PubChemSource` that actually touches the network) route through this
/// one function, so the throttle below applies uniformly rather than being
/// duplicated per-caller and risking one of them forgetting it.
fn real_http_get(url: &str) -> Result<(u16, String), String> {
    // Throttle: a brief delay before every real network call, good-citizen
    // behavior for a public API (PubChem recommends ~5 req/sec max).
    // Injected-fetch-closure unit tests (`lookup_via` with a fake `fetch`)
    // never reach this function, so they're unaffected.
    std::thread::sleep(std::time::Duration::from_millis(250));
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| e.to_string())?;
    let response = client.get(url).send().map_err(|e| e.to_string())?;
    let status = response.status().as_u16();
    check_response_size(response.content_length().map(|l| l as usize), 0)?;
    let bytes = response.bytes().map_err(|e| e.to_string())?;
    check_response_size(None, bytes.len())?;
    let body = String::from_utf8(bytes.to_vec()).map_err(|e| e.to_string())?;
    Ok((status, body))
}

/// Cross-reference a molecule's SMILES against PubChem. Advisory only -- see
/// module doc. Real network call; the audit harness is responsible for
/// throttling across multiple lookups in one run.
pub fn lookup_by_smiles(smiles: &str) -> PubChemQueryOutcome {
    lookup_via(smiles, real_http_get).0
}

/// Same as [`lookup_by_smiles`], but also returns the raw response body
/// when one was received -- used by `cache.rs`'s recording source, which
/// hashes the literal bytes for reproducibility bookkeeping.
pub fn lookup_by_smiles_with_raw(smiles: &str) -> (PubChemQueryOutcome, Option<String>) {
    lookup_via(smiles, real_http_get)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ETHANOL_RESPONSE: &str = r#"{
        "PropertyTable": {
            "Properties": [
                {"CID": 702, "MolecularFormula": "C2H6O", "ConnectivitySMILES": "CCO", "IUPACName": "ethanol"}
            ]
        }
    }"#;

    #[test]
    fn found_response_parses_correctly() {
        let outcome = lookup_via("CCO", |_url| Ok((200, ETHANOL_RESPONSE.to_string()))).0;
        assert_eq!(
            outcome,
            PubChemQueryOutcome::Found(PubChemRecord {
                cid: 702,
                molecular_formula: "C2H6O".to_string(),
                connectivity_smiles: Some("CCO".to_string()),
                iupac_name: Some("ethanol".to_string()),
            })
        );
    }

    #[test]
    fn http_400_is_not_found_not_an_error() {
        // Confirmed live (2026-07-12): PubChem returns HTTP 400 for a
        // malformed/unrecognized SMILES -- treated as "not found," not a
        // failure, since it's a legitimate (if uninformative) outcome.
        let outcome = lookup_via("XYZINVALID", |_url| Ok((400, String::new()))).0;
        assert_eq!(outcome, PubChemQueryOutcome::NotFound);
    }

    #[test]
    fn transport_failure_is_unavailable_not_a_panic() {
        let outcome = lookup_via("CCO", |_url| Err("connection refused".to_string())).0;
        assert!(matches!(outcome, PubChemQueryOutcome::Unavailable(_)));
    }

    #[test]
    fn malformed_json_is_unavailable_not_a_panic() {
        let outcome = lookup_via("CCO", |_url| Ok((200, "not json at all".to_string()))).0;
        assert!(matches!(outcome, PubChemQueryOutcome::Unavailable(_)));
    }

    #[test]
    fn empty_property_table_is_not_found() {
        let empty = r#"{"PropertyTable": {"Properties": []}}"#;
        let outcome = lookup_via("CCO", |_url| Ok((200, empty.to_string()))).0;
        assert_eq!(outcome, PubChemQueryOutcome::NotFound);
    }

    #[test]
    fn parens_and_equals_are_left_as_is_being_rfc3986_legal_in_a_path_segment() {
        // Acetic acid: CC(=O)O. First draft of this test wrongly assumed
        // '(', ')', '=' needed percent-encoding -- they're all in RFC 3986's
        // `sub-delims`, which is part of `pchar`, so they're syntactically
        // legal unencoded in a path segment and `url` correctly leaves them
        // alone. Caught by actually running the test, not by assumption.
        let url = build_smiles_property_url("CC(=O)O").unwrap();
        assert_eq!(
            url.path(),
            "/rest/pug/compound/smiles/CC(=O)O/property/MolecularFormula,ConnectivitySMILES,IUPACName/JSON"
        );
    }

    #[test]
    fn hash_is_percent_encoded_since_it_would_otherwise_start_a_url_fragment() {
        // '#' genuinely needs encoding: unescaped, it would truncate the URL
        // at the fragment boundary and silently request the wrong thing.
        // Real fixture SMILES contain '#' (e.g. "CCC#N", "C=CC#N") so this
        // is a real correctness property, not a hypothetical one.
        let url = build_smiles_property_url("CCC#N").unwrap();
        assert!(
            !url.path().contains('#'),
            "raw '#' must not appear in the URL path (it would start a fragment): {}",
            url.path()
        );
        assert!(
            url.path().contains("%23"),
            "expected '#' percent-encoded as %23: {}",
            url.path()
        );
    }

    #[test]
    #[ignore = "touches the real network -- run explicitly with `cargo test -- --ignored`"]
    fn live_pubchem_lookup_for_ethanol() {
        // Proves real connectivity; not run by default `cargo test` so the
        // suite stays deterministic and network-independent.
        let outcome = lookup_by_smiles("CCO");
        match outcome {
            PubChemQueryOutcome::Found(record) => {
                assert_eq!(record.molecular_formula, "C2H6O");
            }
            other => panic!("expected a live Found result for ethanol, got {other:?}"),
        }
    }

    #[test]
    fn always_unavailable_source_never_touches_network_and_is_advisory() {
        let source = AlwaysUnavailableSource;
        let outcome = source.lookup("CCO");
        assert!(matches!(outcome, PubChemQueryOutcome::Unavailable(_)));
    }

    #[test]
    fn response_within_cap_is_accepted() {
        assert!(check_response_size(Some(200), 200).is_ok());
        assert!(check_response_size(None, 200).is_ok());
    }

    #[test]
    fn oversized_content_length_is_rejected_before_reading() {
        let err = check_response_size(Some(MAX_RESPONSE_BYTES + 1), 0).unwrap_err();
        assert!(err.contains("Content-Length"));
    }

    #[test]
    fn oversized_actual_body_is_rejected_even_without_content_length_header() {
        let err = check_response_size(None, MAX_RESPONSE_BYTES + 1).unwrap_err();
        assert!(err.contains("body"));
    }

    #[test]
    fn record_and_outcome_round_trip_through_json() {
        // Reproducibility (Phase A.1) depends on PubChemQueryOutcome being
        // faithfully serializable -- this is what CachedLookup (cache.rs)
        // persists to a frozen fixture and reloads for offline replay.
        let outcome = PubChemQueryOutcome::Found(PubChemRecord {
            cid: 702,
            molecular_formula: "C2H6O".to_string(),
            connectivity_smiles: Some("CCO".to_string()),
            iupac_name: Some("ethanol".to_string()),
        });
        let json = serde_json::to_string(&outcome).unwrap();
        let round_tripped: PubChemQueryOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(outcome, round_tripped);
    }
}
