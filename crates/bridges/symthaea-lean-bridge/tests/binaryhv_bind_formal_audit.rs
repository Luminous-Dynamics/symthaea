// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end SYM-FV-002 theorem/axiom audit on the repaired v2 subject.

use std::path::PathBuf;

use symthaea_lean_bridge::axiom_gate::AxiomPolicy;
use symthaea_lean_bridge::theorem_set_audit::{TheoremAuditCase, audit_lean_theorem_set};

fn formal_audit_enabled() -> bool {
    std::env::var("SYMTHAEA_LEAN_FORMAL_AUDIT")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn proof_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../formal/lean/hdc/BinaryHVBind.lean")
}

fn cases() -> [TheoremAuditCase<'static>; 8] {
    [
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.bit_bind",
            expected_statement: "(a b : BinaryHV) (i : BitIndex) : bit (bind a b) i = Bool.xor (bit a i) (bit b i)",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.bind_zero_right",
            expected_statement: "(a : BinaryHV) : bind a zero = a",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.bind_zero_left",
            expected_statement: "(a : BinaryHV) : bind zero a = a",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.bind_self_inverse",
            expected_statement: "(a : BinaryHV) : bind a a = zero",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.bind_comm",
            expected_statement: "(a b : BinaryHV) : bind a b = bind b a",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.bind_assoc",
            expected_statement: "(a b c : BinaryHV) : bind (bind a b) c = bind a (bind b c)",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.unbind_right",
            expected_statement: "(a b : BinaryHV) : bind (bind a b) b = a",
        },
        TheoremAuditCase {
            theorem: "Symthaea.Formal.HDC.unbind_left",
            expected_statement: "(a b : BinaryHV) : bind (bind a b) a = b",
        },
    ]
}

#[test]
fn binaryhv_bind_theorem_set_is_kernel_checked_and_constitutionally_audited() {
    if !formal_audit_enabled() {
        eprintln!("[sym-fv-002a-v2] real-Lean audit disabled outside dedicated formal lane");
        return;
    }

    let cases = cases();
    let report = audit_lean_theorem_set(proof_path(), &cases, &AxiomPolicy::constitutional());
    assert_eq!(
        report.results.len(),
        cases.len(),
        "every requested theorem must produce one explicit result: {report:#?}"
    );
    assert!(
        report.accepted(),
        "BinaryHV theorem set failed real-Lean/proof audit: {report:#?}"
    );
}

#[test]
fn hostile_expected_statement_mutation_is_rejected() {
    if !formal_audit_enabled() {
        return;
    }

    let mutant = [TheoremAuditCase {
        theorem: "Symthaea.Formal.HDC.bit_bind",
        expected_statement: "(a b : BinaryHV) (i : BitIndex) : bit (bind a b) i = Bool.or (bit a i) (bit b i)",
    }];
    let report = audit_lean_theorem_set(proof_path(), &mutant, &AxiomPolicy::constitutional());
    assert_eq!(report.results.len(), 1, "mutant must produce one explicit audit result");
    assert!(
        !report.accepted(),
        "hostile wrong-spec mutant produced a false green: {report:#?}"
    );
}
