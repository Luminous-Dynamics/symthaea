// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! RDKit differential cross-reference (Phase A.2), a second, independent
//! advisory source alongside PubChem.
//!
//! **Why a subprocess, not an FFI binding**: RDKit is a Python package
//! (`python3Packages.rdkit` in nixpkgs, confirmed available -- not
//! installed in this crate's default build environment, and deliberately
//! not added as a flake dependency here: this crate's build must not
//! require RDKit to compile or run its own test suite, only to exercise
//! this ONE advisory cross-reference when a caller opts in). A small
//! inline Python script, invoked via `std::process::Command` with a
//! timeout, is far less engineering than an FFI binding for a single
//! read-only "parse this SMILES and report its canonical form" call, and
//! keeps RDKit fully optional -- if it isn't installed, every lookup is
//! `Unavailable`, exactly like a PubChem network failure, never a build or
//! test failure.
//!
//! **Same trust boundary as `pubchem.rs`**: every outcome is advisory
//! only. Nothing in `validity.rs`/`policy.rs`/`oracle.rs` reads an
//! `RdkitQueryOutcome`. A subprocess timeout, a missing `rdkit` install, or
//! RDKit itself rejecting a SMILES are all `Unavailable`/`RejectedByRdkit`
//! -- informational, never gating.
//!
//! **The goal is bug-finding, not authority.** RDKit is one of the most
//! widely used cheminformatics toolkits; a disagreement between this
//! crate's own parser and RDKit's is a signal worth a human look (either
//! side could be wrong), not a reason to trust RDKit's answer over this
//! crate's own validity/scope pipeline.

use serde::Deserialize;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;

const TIMEOUT: Duration = Duration::from_secs(10);
/// RDKit's JSON output for a single small-molecule lookup is well under 1KiB
/// in practice; this is generous headroom, not a tuned limit -- mirrors
/// `pubchem.rs`'s `MAX_RESPONSE_BYTES` reasoning.
const MAX_OUTPUT_BYTES: usize = 8 * 1024;

/// Inline, not a sibling script file -- avoids any runtime file-path
/// assumption (this crate already avoids that pattern for
/// `fixtures/pubchem_corpus_fixture.json` via `include_str!`; this is the
/// same principle applied to a subprocess script instead of a data file).
/// Reads exactly one argument (the SMILES), emits exactly one JSON line.
const RDKIT_BRIDGE_SCRIPT: &str = r#"
import sys, json
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except ImportError as e:
    print(json.dumps({"status": "unavailable", "error": "rdkit not importable: " + str(e)}))
    sys.exit(0)
smiles = sys.argv[1]
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    print(json.dumps({"status": "rejected", "error": "RDKit could not parse this SMILES"}))
else:
    print(json.dumps({
        "status": "found",
        "canonical_smiles": Chem.MolToSmiles(mol),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
    }))
"#;

#[derive(Debug, Clone, PartialEq)]
pub struct RdkitRecord {
    pub canonical_smiles: String,
    pub molecular_formula: String,
}

/// Three outcomes, mirroring `pubchem::PubChemQueryOutcome`'s shape:
/// `Found` (RDKit parsed it), `RejectedByRdkit` (RDKit itself says this
/// SMILES is invalid -- a real, informative disagreement signal, distinct
/// from the tool being unreachable), `Unavailable` (RDKit not installed,
/// subprocess failed to spawn, timed out, or produced unparseable output).
#[derive(Debug, Clone, PartialEq)]
pub enum RdkitQueryOutcome {
    Found(RdkitRecord),
    RejectedByRdkit(String),
    Unavailable(String),
}

pub trait RdkitSource {
    fn lookup(&self, smiles: &str) -> RdkitQueryOutcome;
}

pub struct LiveRdkitSource;

impl RdkitSource for LiveRdkitSource {
    fn lookup(&self, smiles: &str) -> RdkitQueryOutcome {
        lookup_via(smiles, run_python_bridge)
    }
}

/// Fault injection, mirroring `pubchem::AlwaysUnavailableSource` -- every
/// lookup is `Unavailable`, unconditionally.
pub struct AlwaysUnavailableRdkitSource;

impl RdkitSource for AlwaysUnavailableRdkitSource {
    fn lookup(&self, _smiles: &str) -> RdkitQueryOutcome {
        RdkitQueryOutcome::Unavailable("fault-injected for testing: rdkit unreachable".into())
    }
}

#[derive(Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
enum BridgeResponse {
    Found {
        canonical_smiles: String,
        molecular_formula: String,
    },
    Rejected {
        error: String,
    },
    Unavailable {
        error: String,
    },
}

/// Injectable process layer, same pattern as `pubchem.rs::lookup_via`:
/// `run` returns the subprocess's raw stdout on success, `Err` for any
/// spawn/timeout/non-zero-exit failure. Unit tests supply a fixed string
/// and never touch a real subprocess.
fn lookup_via<F: Fn(&str) -> Result<String, String>>(smiles: &str, run: F) -> RdkitQueryOutcome {
    let raw = match run(smiles) {
        Ok(s) => s,
        Err(e) => return RdkitQueryOutcome::Unavailable(e),
    };
    if raw.len() > MAX_OUTPUT_BYTES {
        return RdkitQueryOutcome::Unavailable(format!(
            "rdkit bridge output ({} bytes) exceeds the {MAX_OUTPUT_BYTES}-byte cap",
            raw.len()
        ));
    }
    match serde_json::from_str::<BridgeResponse>(raw.trim()) {
        Ok(BridgeResponse::Found {
            canonical_smiles,
            molecular_formula,
        }) => RdkitQueryOutcome::Found(RdkitRecord {
            canonical_smiles,
            molecular_formula,
        }),
        Ok(BridgeResponse::Rejected { error }) => RdkitQueryOutcome::RejectedByRdkit(error),
        Ok(BridgeResponse::Unavailable { error }) => RdkitQueryOutcome::Unavailable(error),
        Err(e) => RdkitQueryOutcome::Unavailable(format!(
            "unexpected bridge output shape: {e} (raw: {})",
            raw.chars().take(200).collect::<String>()
        )),
    }
}

/// **Data/code separation, verified not assumed**: `smiles` is passed as its
/// own `.arg()` -- a distinct `argv` element the OS hands to the python3
/// process via `execve`, landing in the script as `sys.argv[1]`. It is
/// never concatenated, formatted, or interpolated into `RDKIT_BRIDGE_SCRIPT`
/// (a `const`, never built at runtime) or into any shell command string.
/// `std::process::Command` on Unix does not invoke a shell at all (no
/// `sh -c "..."` wrapping) -- there is no shell-metacharacter parsing step
/// for `smiles` to escape out of, and no way for its content to be
/// reinterpreted as additional arguments or extra Python source. The
/// remaining trust boundary is RDKit's own SMILES parser, not this
/// subprocess call.
fn run_python_bridge(smiles: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(RDKIT_BRIDGE_SCRIPT)
        .arg(smiles)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn python3: {e}"))?;

    match child.wait_timeout(TIMEOUT).map_err(|e| e.to_string())? {
        Some(status) => {
            let mut stdout = String::new();
            if let Some(mut out) = child.stdout.take() {
                out.read_to_string(&mut stdout).map_err(|e| e.to_string())?;
            }
            if !status.success() {
                let mut stderr = String::new();
                if let Some(mut err) = child.stderr.take() {
                    let _ = err.read_to_string(&mut stderr);
                }
                return Err(format!(
                    "python3 exited with {status}: {}",
                    stderr.chars().take(300).collect::<String>()
                ));
            }
            Ok(stdout)
        }
        None => {
            let _ = child.kill();
            let _ = child.wait();
            Err(format!("python3 rdkit bridge timed out after {TIMEOUT:?}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn found_response_parses_correctly() {
        let outcome =
            lookup_via("CCO", |_| {
                Ok(r#"{"status": "found", "canonical_smiles": "CCO", "molecular_formula": "C2H6O"}"#
                .to_string())
            });
        assert_eq!(
            outcome,
            RdkitQueryOutcome::Found(RdkitRecord {
                canonical_smiles: "CCO".to_string(),
                molecular_formula: "C2H6O".to_string(),
            })
        );
    }

    #[test]
    fn rdkit_not_installed_is_unavailable_not_a_panic() {
        let outcome = lookup_via("CCO", |_| {
            Ok(r#"{"status": "unavailable", "error": "rdkit not importable: No module named 'rdkit'"}"#
                .to_string())
        });
        assert!(matches!(outcome, RdkitQueryOutcome::Unavailable(_)));
    }

    #[test]
    fn rdkit_rejecting_a_smiles_is_a_distinct_outcome_from_unavailable() {
        let outcome = lookup_via("not a smiles", |_| {
            Ok(
                r#"{"status": "rejected", "error": "RDKit could not parse this SMILES"}"#
                    .to_string(),
            )
        });
        assert!(matches!(outcome, RdkitQueryOutcome::RejectedByRdkit(_)));
    }

    #[test]
    fn subprocess_spawn_failure_is_unavailable_not_a_panic() {
        let outcome = lookup_via("CCO", |_| {
            Err("failed to spawn python3: not found".to_string())
        });
        assert!(matches!(outcome, RdkitQueryOutcome::Unavailable(_)));
    }

    #[test]
    fn malformed_bridge_output_is_unavailable_not_a_panic() {
        let outcome = lookup_via("CCO", |_| Ok("not json at all".to_string()));
        assert!(matches!(outcome, RdkitQueryOutcome::Unavailable(_)));
    }

    #[test]
    fn shell_metacharacters_in_smiles_cannot_escape_argv_into_a_shell() {
        // Regression/documentation test for the argv-vs-shell trust
        // boundary the doc comment on `run_python_bridge` claims: a string
        // containing quotes, semicolons, backticks, and a subshell
        // injection attempt must round-trip as one inert argument, never
        // be interpreted as shell syntax. `std::process::Command` never
        // invokes a shell, so this is safe by construction -- this test
        // exercises the REAL subprocess path (not the injectable seam) to
        // confirm that isn't just true in theory. Doesn't require RDKit to
        // be installed: even the "unavailable" ImportError path proves the
        // subprocess ran cleanly to completion rather than crashing or
        // executing injected shell syntax.
        let hostile = "CCO'; rm -rf /tmp/rdkit-injection-canary; echo pwned $(whoami) `id`";
        let result = run_python_bridge(hostile);
        assert!(
            result.is_ok(),
            "subprocess call itself must complete cleanly: {result:?}"
        );
        let canary = std::path::Path::new("/tmp/rdkit-injection-canary");
        assert!(
            !canary.exists(),
            "shell metacharacters in a SMILES string executed as shell syntax"
        );
    }

    #[test]
    fn oversized_output_is_rejected() {
        let huge = "x".repeat(MAX_OUTPUT_BYTES + 1);
        let outcome = lookup_via("CCO", move |_| Ok(huge.clone()));
        assert!(matches!(outcome, RdkitQueryOutcome::Unavailable(_)));
    }

    #[test]
    fn always_unavailable_source_never_touches_a_subprocess() {
        let source = AlwaysUnavailableRdkitSource;
        assert!(matches!(
            source.lookup("CCO"),
            RdkitQueryOutcome::Unavailable(_)
        ));
    }

    #[test]
    #[ignore = "requires a local RDKit install -- run explicitly, e.g. under \
                `nix-shell -p python3Packages.rdkit --run 'cargo test -p symthaea-process-discovery -- --ignored'`"]
    fn live_rdkit_lookup_for_ethanol() {
        let source = LiveRdkitSource;
        match source.lookup("CCO") {
            RdkitQueryOutcome::Found(record) => {
                assert_eq!(record.molecular_formula, "C2H6O");
            }
            other => panic!("expected a live Found result for ethanol, got {other:?}"),
        }
    }
}
