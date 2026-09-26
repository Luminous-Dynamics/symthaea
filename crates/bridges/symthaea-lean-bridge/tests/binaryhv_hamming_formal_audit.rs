// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end SYM-FV-004A theorem/axiom audit.
//!
//! The child Hamming file deliberately reuses the exact SYM-FV-002 vocabulary
//! instead of copying it. For real-Lean qualification this test creates one
//! temporary combined source from the exact parent and child bytes, then routes
//! every child theorem through the reusable theorem-set audit introduced by
//! SYM-FV-002A.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use symthaea_lean_bridge::axiom_gate::AxiomPolicy;
use symthaea_lean_bridge::theorem_set_audit::{
    TheoremAuditCase, audit_lean_theorem_set,
};

static COMBINED_NONCE: AtomicU64 = AtomicU64::new(0);

fn formal_audit_enabled() -> bool {
    std::env::var("SYMTHAEA_LEAN_FORMAL_AUDIT")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../").join(relative)
}

fn parent_path() -> PathBuf {
    repo_path("formal/lean/hdc/BinaryHVBind.lean")
}

fn child_path() -> PathBuf {
    repo_path("formal/lean/hdc/BinaryHVHamming.lean")
}

fn create_combined_source(parent: &Path, child: &Path) -> PathBuf {
    let parent_bytes = fs::read(parent).expect("read exact SYM-FV-002 parent theorem source");
    let child_bytes = fs::read(child).expect("read exact SYM-FV-004A child theorem source");
    let nonce = COMBINED_NONCE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "sym_fv_004a_combined_{}_{}.lean",
        std::process::id(),
        nonce
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("create-new combined Lean subject");
    file.write_all(&parent_bytes).expect("write parent theorem bytes");
    file.write_all(b"\n").expect("write theorem separator");
    file.write_all(&child_bytes).expect("write child theorem bytes");
    file.flush().expect("flush combined Lean subject");
    path
}

fn cases() -> [TheoremAuditCase<'static>; 7] {
    [
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.mismatch_bit",
            expected_statement: "(a b : BinaryHV) (i : BitIndex) : bit (mismatchVector a b) i = Bool.xor (bit a i) (bit b i)",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.mismatchVector_eq_bind",
            expected_statement: "(a b : BinaryHV) : mismatchVector a b = bind a b",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.hamming_eq_bitCount_bind",
            expected_statement: "(a b : BinaryHV) : hammingDistance a b = bitCount (bind a b)",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.hamming_symm",
            expected_statement: "(a b : BinaryHV) : hammingDistance a b = hammingDistance b a",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.mismatch_bind_right_invariant",
            expected_statement: "(a b mask : BinaryHV) : mismatchVector (bind a mask) (bind b mask) = mismatchVector a b",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.hamming_bind_right_invariant",
            expected_statement: "(a b mask : BinaryHV) : hammingDistance (bind a mask) (bind b mask) = hammingDistance a b",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.hamming_bind_left_invariant",
            expected_statement: "(mask a b : BinaryHV) : hammingDistance (bind mask a) (bind mask b) = hammingDistance a b",
        },
    ]
}

#[test]
fn binaryhv_hamming_theorem_set_is_kernel_checked_and_constitutionally_audited() {
    if !formal_audit_enabled() {
        eprintln!(
            "[sym-fv-004a] SYMTHAEA_LEAN_FORMAL_AUDIT not set; skipping real-Lean theorem audit"
        );
        return;
    }

    let combined = create_combined_source(&parent_path(), &child_path());
    let cases = cases();
    let report = audit_lean_theorem_set(&combined, &cases, &AxiomPolicy::constitutional());
    let cleanup = fs::remove_file(&combined);

    assert_eq!(
        report.results.len(),
        cases.len(),
        "every requested Hamming theorem must produce one explicit audit result: {report:#?}"
    );
    assert!(
        report.accepted(),
        "BinaryHV Hamming theorem set failed end-to-end Lean/proof audit: {report:#?}"
    );
    cleanup.expect("remove combined SYM-FV-004A Lean subject");
}

#[test]
fn hostile_hamming_isometry_statement_mutation_is_rejected() {
    if !formal_audit_enabled() {
        return;
    }

    let combined = create_combined_source(&parent_path(), &child_path());
    let mutant = [TheoremAuditCase {
        theorem: "Symthaea.Formal.HDC.hamming_bind_right_invariant",
        expected_statement: "(a b mask : BinaryHV) : hammingDistance (bind a mask) (bind b mask) = hammingDistance a mask",
    }];
    let report = audit_lean_theorem_set(&combined, &mutant, &AxiomPolicy::constitutional());
    let cleanup = fs::remove_file(&combined);

    assert_eq!(report.results.len(), 1, "mutant must reach exactly one explicit audit result");
    assert!(
        !report.accepted(),
        "hostile wrong Hamming-isometry statement produced a false green: {report:#?}"
    );
    cleanup.expect("remove hostile-control combined Lean subject");
}
