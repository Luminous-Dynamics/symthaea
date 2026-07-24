// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reaction Corpus Auditor runner for `symthaea/CHEMICAL_PROCESS_DISCOVERY_PLAN_2026-07-12.md`
//! Phase A / A.1 / A.2.
//!
//! Runs the hand-curated fixture corpus (`corpus.rs`) through this crate's
//! existing validity/template/policy/certificate pipeline (`audit.rs`),
//! optionally cross-references each distinct compound against PubChem and
//! RDKit, and prints a full per-record report plus a per-category metrics
//! table (`metrics.rs`) -- this is the tool this project built, actually run.
//!
//! PubChem modes (mutually exclusive; first flag found wins):
//! - (no flag) -- live PubChem cross-referencing, no fixture written. This
//!   is the "just run it" default.
//! - `--offline` -- skip PubChem entirely (`pubchem_source: None`).
//! - `--record <path>` -- live PubChem, AND writes every response received
//!   to `<path>` as a frozen fixture for later `--replay`.
//! - `--replay <path>` -- zero network access; answers every lookup from a
//!   previously recorded fixture at `<path>`.
//!
//! RDKit is a separate, independent flag: on by default (attempts a live
//! subprocess lookup, gracefully reports `Unavailable` if RDKit isn't
//! installed locally -- see `rdkit.rs`'s module doc). Pass `--no-rdkit` to
//! skip it entirely.

use std::path::PathBuf;
use symthaea_process_discovery::audit::{
    PubChemAgreement, RdkitAgreement, RecordOutcome, run_audit,
};
use symthaea_process_discovery::cache::{PubChemFixtureCache, RecordingSource, ReplaySource};
use symthaea_process_discovery::corpus::phase_a_fixture_corpus;
use symthaea_process_discovery::hazard_heuristics::ExternalScopeConfig;
use symthaea_process_discovery::metrics::compute_metrics;
use symthaea_process_discovery::policy::OpenWithHeuristicScreenPolicy;
use symthaea_process_discovery::pubchem::{LivePubChemSource, PubChemQueryOutcome};
use symthaea_process_discovery::rdkit::{LiveRdkitSource, RdkitQueryOutcome, RdkitSource};

enum PubchemMode {
    Live,
    Offline,
    Record(PathBuf),
    Replay(PathBuf),
}

fn parse_pubchem_mode(args: &[String]) -> PubchemMode {
    for (i, a) in args.iter().enumerate() {
        match a.as_str() {
            "--offline" => return PubchemMode::Offline,
            "--record" => {
                let path = args
                    .get(i + 1)
                    .unwrap_or_else(|| panic!("--record requires a path argument"));
                return PubchemMode::Record(PathBuf::from(path));
            }
            "--replay" => {
                let path = args
                    .get(i + 1)
                    .unwrap_or_else(|| panic!("--replay requires a path argument"));
                return PubchemMode::Replay(PathBuf::from(path));
            }
            _ => {}
        }
    }
    PubchemMode::Live
}

fn print_report(report: &symthaea_process_discovery::audit::AuditReport) {
    for record in &report.records {
        println!("--- {} [{}] ---", record.name, record.category.label());
        println!("  source: {}", record.source);
        match &record.outcome {
            RecordOutcome::ParseFailed(e) => println!("  PARSE FAILED: {e}"),
            RecordOutcome::DeclaredProductMismatch {
                template,
                computed_formulas,
                declared_formulas,
            } => println!(
                "  DECLARED PRODUCT MISMATCH [{template}]: computed={computed_formulas:?} declared={declared_formulas:?}"
            ),
            RecordOutcome::MatchedButFailedValidity { template, reason } => {
                println!("  MATCHED [{template}] BUT FAILED VALIDITY: {reason}")
            }
            RecordOutcome::MatchedButScopeRejected { template, reason } => {
                println!("  MATCHED [{template}] BUT SCOPE REJECTED: {reason}")
            }
            RecordOutcome::Certified { template } => {
                println!("  CERTIFIED [{template}]");
                if let Some(cert) = &record.certificate {
                    println!("    {}", cert.summary());
                }
            }
            RecordOutcome::Unclassified => println!("  UNCLASSIFIED (no template matched)"),
        }
        match &record.raw_molecule_validity {
            Ok(()) => println!(
                "  raw structural validity: OK{}",
                if record.normalization_applied {
                    " (after normalization)"
                } else {
                    ""
                }
            ),
            Err(e) => println!("  raw structural validity: FAILED -- {e}"),
        }
        println!(
            "  expectation match: outcome={} raw_validity={}",
            record.matched_expectation, record.raw_validity_matched_expectation
        );
        for xref in &record.pubchem {
            let agreement = match xref.agreement {
                PubChemAgreement::Agrees => "AGREES",
                PubChemAgreement::RepresentationOnlyDifference => {
                    "agrees (representation-only formula ordering difference)"
                }
                PubChemAgreement::Disagrees => "DISAGREES",
                PubChemAgreement::NotFoundInPubChem => "not found in PubChem",
                PubChemAgreement::Unavailable => "unavailable",
            };
            match &xref.outcome {
                PubChemQueryOutcome::Found(pc) => println!(
                    "  pubchem[{}]: CID={} formula={} ours={} -> {agreement} (advisory only)",
                    xref.smiles, pc.cid, pc.molecular_formula, xref.our_formula
                ),
                PubChemQueryOutcome::NotFound => {
                    println!("  pubchem[{}]: not found (advisory only)", xref.smiles)
                }
                PubChemQueryOutcome::Unavailable(reason) => println!(
                    "  pubchem[{}]: unavailable -- {reason} (advisory only)",
                    xref.smiles
                ),
            }
        }
        for xref in &record.rdkit {
            let agreement = match xref.agreement {
                RdkitAgreement::Agrees => "AGREES",
                RdkitAgreement::RepresentationOnlyDifference => {
                    "agrees (representation-only formula ordering difference)"
                }
                RdkitAgreement::Disagrees => "DISAGREES",
                RdkitAgreement::RejectedByRdkit => "rejected by RDKit",
                RdkitAgreement::Unavailable => "unavailable",
            };
            match &xref.outcome {
                RdkitQueryOutcome::Found(rd) => println!(
                    "  rdkit[{}]: canonical={} formula={} ours={} -> {agreement} (advisory only)",
                    xref.smiles, rd.canonical_smiles, rd.molecular_formula, xref.our_formula
                ),
                RdkitQueryOutcome::RejectedByRdkit(reason) => println!(
                    "  rdkit[{}]: rejected -- {reason} (advisory only)",
                    xref.smiles
                ),
                RdkitQueryOutcome::Unavailable(reason) => println!(
                    "  rdkit[{}]: unavailable -- {reason} (advisory only)",
                    xref.smiles
                ),
            }
        }
        println!();
    }

    let s = &report.summary;
    println!("=== Summary ===");
    println!("total_records={}", s.total_records);
    println!("parse_failed={}", s.parse_failed);
    println!("declared_product_mismatch={}", s.declared_product_mismatch);
    println!("matched_failed_validity={}", s.matched_failed_validity);
    println!("matched_scope_rejected={}", s.matched_scope_rejected);
    println!("certified={}", s.certified);
    println!("unclassified={}", s.unclassified);
    println!(
        "pubchem: agrees={} representation_only={} disagrees={} not_found={} unavailable={}",
        s.pubchem_agreements,
        s.pubchem_representation_only,
        s.pubchem_disagreements,
        s.pubchem_not_found,
        s.pubchem_unavailable
    );
    println!(
        "rdkit: agrees={} representation_only={} disagrees={} rejected={} unavailable={}",
        s.rdkit_agreements,
        s.rdkit_representation_only,
        s.rdkit_disagreements,
        s.rdkit_rejected,
        s.rdkit_unavailable
    );
    println!();
    println!("Every 'CERTIFIED' line above is backed by a real ProcessCertificate (full atom/bond");
    println!("structure, gate evidence, normalization evidence) -- this printout shows only its");
    println!("one-line summary. PubChem/RDKit cross-references are advisory only and never affect");
    println!("classification or certification.");
}

fn print_metrics(
    corpus: &[symthaea_process_discovery::corpus::CorpusRecord],
    policy: &OpenWithHeuristicScreenPolicy,
    report: &symthaea_process_discovery::audit::AuditReport,
) {
    println!("\n=== Per-category metrics (Phase A.2) ===");
    for m in compute_metrics(corpus, policy, report) {
        println!(
            "{:<28} total={:<3} expectation_pass_rate={:.0}% raw_validity_matches={}/{} certified={} deterministic={}/{} normalized={} pubchem_disagree={} rdkit_disagree={}",
            m.category.label(),
            m.total,
            m.expectation_pass_rate() * 100.0,
            m.raw_validity_matches,
            m.total,
            m.certified,
            m.certificate_deterministic,
            m.certified,
            m.normalization_applied,
            m.pubchem_disagreements,
            m.rdkit_disagreements,
        );
    }
    println!();
    println!("expectation_pass_rate is the core signal: did each category get the CORRECT kind");
    println!("of outcome (certify / abstain / catch-a-mismatch / scope-reject), not just some");
    println!("outcome. A category below 100% here means either the corpus's own hand-authored");
    println!("expectation is wrong, or the pipeline regressed -- either way, worth investigating.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rdkit_enabled = !args.iter().any(|a| a == "--no-rdkit");

    let policy = OpenWithHeuristicScreenPolicy {
        external: ExternalScopeConfig::default(),
    };
    let corpus = phase_a_fixture_corpus();
    println!("=== Reaction Corpus Auditor ===");
    println!(
        "records={} rdkit_cross_reference={}",
        corpus.len(),
        if rdkit_enabled {
            "ON (live subprocess, gracefully degrades if not installed)"
        } else {
            "OFF (--no-rdkit)"
        }
    );

    let rdkit_source = LiveRdkitSource;
    let rdkit_source: Option<&dyn RdkitSource> = if rdkit_enabled {
        Some(&rdkit_source)
    } else {
        None
    };

    match parse_pubchem_mode(&args) {
        PubchemMode::Offline => {
            println!("pubchem_cross_reference=OFF (--offline)\n");
            let report = run_audit(&corpus, &policy, None, rdkit_source);
            print_report(&report);
            print_metrics(&corpus, &policy, &report);
        }
        PubchemMode::Live => {
            println!("pubchem_cross_reference=ON (live network, not recorded)\n");
            let source = LivePubChemSource;
            let report = run_audit(&corpus, &policy, Some(&source), rdkit_source);
            print_report(&report);
            print_metrics(&corpus, &policy, &report);
        }
        PubchemMode::Record(path) => {
            println!("pubchem_cross_reference=ON (live network, recording to {path:?})\n");
            let source = RecordingSource::new();
            let report = run_audit(&corpus, &policy, Some(&source), rdkit_source);
            print_report(&report);
            print_metrics(&corpus, &policy, &report);
            let cache = source.into_cache();
            cache
                .save_to_file(&path)
                .unwrap_or_else(|e| panic!("failed to save fixture cache to {path:?}: {e}"));
            println!(
                "Recorded {} PubChem lookups to {path:?}",
                cache.entries.len()
            );
        }
        PubchemMode::Replay(path) => {
            println!("pubchem_cross_reference=REPLAY (zero network, fixture={path:?})\n");
            let cache = PubChemFixtureCache::load_from_file(&path)
                .unwrap_or_else(|e| panic!("failed to load fixture cache from {path:?}: {e}"));
            let source = ReplaySource::from_cache(cache);
            let report = run_audit(&corpus, &policy, Some(&source), rdkit_source);
            print_report(&report);
            print_metrics(&corpus, &policy, &report);
        }
    }
}
