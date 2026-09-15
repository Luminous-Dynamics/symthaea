#!/usr/bin/env python3
"""Run the strict SPINE-000B-P1R Rust equivalence harness against production OutputCollector.

This script is qualification-only and measurement-only. It temporarily injects a
`#[cfg(test)]` unit test into the existing crate-private `subsystem_trait.rs` test
module, executes that test, and restores the original source bytes in a finally
block. The production OutputCollector implementation is therefore exercised
without widening its visibility or permanently changing cognitive code.

The injected test compares, bit-for-bit, both I_all and every I_withoutS value
against the independently generated Python golden fixture, including flags and
contributor counts. The source file must be byte-identical after the harness.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

SUBJECT = Path("src/cognitive_loop/subsystem_trait.rs")
FIXTURE = Path("tests/fixtures/spine_000b_golden_fixtures.json")
SENTINEL = "fn test_spine_000b_p1r_strict_generated_harness()"

RUST_TEST = r'''

    /// SPINE-000B-P1R strict generated harness.
    /// Injected only during qualification; production collector code is unchanged.
    #[test]
    fn test_spine_000b_p1r_strict_generated_harness() {
        fn proposal_from_json(p: &serde_json::Value) -> (&'static str, SubsystemOutput) {
            let name_raw = p["subsystem_name"]
                .as_str()
                .expect("subsystem_name must be string");
            let name: &'static str = Box::leak(name_raw.to_string().into_boxed_str());
            let confidence_delta = p
                .get("confidence_delta")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let lr_modulation = p
                .get("lr_modulation")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            let exploration_delta = p
                .get("exploration_delta")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let arousal_delta = p
                .get("arousal_delta")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let valence_delta = p
                .get("valence_delta")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let flags = p.get("flags").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            (
                name,
                SubsystemOutput {
                    confidence_delta,
                    lr_modulation,
                    exploration_delta,
                    arousal_delta,
                    valence_delta,
                    flags,
                    _reserved: 0,
                },
            )
        }

        fn build_collector(
            proposals: &[serde_json::Value],
            omit: Option<&str>,
        ) -> OutputCollector {
            let mut collector = OutputCollector::new();
            for proposal in proposals {
                let raw_name = proposal["subsystem_name"]
                    .as_str()
                    .expect("subsystem_name must be string");
                if omit.is_some_and(|name| name == raw_name) {
                    continue;
                }
                let (name, output) = proposal_from_json(proposal);
                collector.record(name, output);
            }
            collector
        }

        fn assert_integrated_exact(
            actual: &IntegratedOutput,
            expected: &serde_json::Value,
            label: &str,
        ) {
            assert_eq!(
                actual.confidence_delta.to_bits(),
                expected["confidence_delta_bits"].as_u64().unwrap(),
                "{label}: confidence_delta_bits mismatch"
            );
            assert_eq!(
                actual.lr_modulation.to_bits(),
                expected["lr_modulation_bits"].as_u64().unwrap(),
                "{label}: lr_modulation_bits mismatch"
            );
            assert_eq!(
                actual.exploration_delta.to_bits(),
                expected["exploration_delta_bits"].as_u64().unwrap(),
                "{label}: exploration_delta_bits mismatch"
            );
            assert_eq!(
                actual.arousal_delta.to_bits() as u64,
                expected["arousal_delta_bits"].as_u64().unwrap(),
                "{label}: arousal_delta_bits mismatch"
            );
            assert_eq!(
                actual.valence_delta.to_bits() as u64,
                expected["valence_delta_bits"].as_u64().unwrap(),
                "{label}: valence_delta_bits mismatch"
            );
            assert_eq!(
                actual.flags,
                expected["flags"].as_u64().unwrap() as u32,
                "{label}: flags mismatch"
            );
            assert_eq!(
                actual.n_contributors,
                expected["n_contributors"].as_u64().unwrap() as usize,
                "{label}: n_contributors mismatch"
            );
        }

        let fixture_path = std::path::Path::new(
            "tests/fixtures/spine_000b_golden_fixtures.json",
        );
        let fixture_str = std::fs::read_to_string(fixture_path)
            .expect("failed reading P1R golden fixture");
        let fixtures: serde_json::Value = serde_json::from_str(&fixture_str)
            .expect("failed parsing P1R golden fixture");
        let cases = fixtures.as_array().expect("fixtures must be an array");
        assert_eq!(cases.len(), 16, "P1R requires the complete 16-case fixture matrix");

        for case in cases {
            let case_name = case["name"].as_str().unwrap();
            let proposals = case["input_proposals"].as_array().unwrap();

            let collector = build_collector(proposals, None);
            let integrated_all = collector.integrate();
            assert_integrated_exact(
                &integrated_all,
                &case["expected_integrated"]["exact_bits"],
                &format!("{case_name}/I_all"),
            );

            let expected_receipts = case["expected_report"]["receipts"]
                .as_array()
                .expect("expected receipts must be array");
            assert_eq!(expected_receipts.len(), proposals.len());

            let mut names: Vec<&str> = proposals
                .iter()
                .map(|p| p["subsystem_name"].as_str().unwrap())
                .collect();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), proposals.len(), "{case_name}: duplicate subsystem names");

            for (receipt_index, target_name) in names.iter().enumerate() {
                let expected_receipt = &expected_receipts[receipt_index];
                assert_eq!(
                    *target_name,
                    expected_receipt["subsystem_name"].as_str().unwrap(),
                    "{case_name}: oracle receipt order mismatch"
                );

                let collector_without = build_collector(proposals, Some(target_name));
                let integrated_without = collector_without.integrate();
                assert_integrated_exact(
                    &integrated_without,
                    &expected_receipt["integrated_without_subject"],
                    &format!("{case_name}/{target_name}/I_withoutS"),
                );

                let mut changed_channels = Vec::new();
                if integrated_all.confidence_delta.to_bits()
                    != integrated_without.confidence_delta.to_bits()
                {
                    changed_channels.push("confidence_delta_bits");
                }
                if integrated_all.lr_modulation.to_bits()
                    != integrated_without.lr_modulation.to_bits()
                {
                    changed_channels.push("lr_modulation_bits");
                }
                if integrated_all.exploration_delta.to_bits()
                    != integrated_without.exploration_delta.to_bits()
                {
                    changed_channels.push("exploration_delta_bits");
                }
                if integrated_all.arousal_delta.to_bits()
                    != integrated_without.arousal_delta.to_bits()
                {
                    changed_channels.push("arousal_delta_bits");
                }
                if integrated_all.valence_delta.to_bits()
                    != integrated_without.valence_delta.to_bits()
                {
                    changed_channels.push("valence_delta_bits");
                }

                let unique_flags = integrated_all.flags & !integrated_without.flags;
                let integration_changed = !changed_channels.is_empty() || unique_flags != 0;
                let expected_channels: Vec<&str> = expected_receipt["changed_channels"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_str().unwrap())
                    .collect();
                assert_eq!(
                    changed_channels, expected_channels,
                    "{case_name}/{target_name}: changed_channels mismatch"
                );
                assert_eq!(
                    unique_flags,
                    expected_receipt["uniquely_contributed_flags"].as_u64().unwrap() as u32,
                    "{case_name}/{target_name}: unique flags mismatch"
                );
                assert_eq!(
                    integration_changed,
                    expected_receipt["integration_changed"].as_bool().unwrap(),
                    "{case_name}/{target_name}: integration_changed mismatch"
                );
            }
        }
    }
'''


def main() -> int:
    if not SUBJECT.is_file() or not FIXTURE.is_file():
        raise SystemExit("P1R harness subjects are missing")

    original = SUBJECT.read_bytes()
    original_sha = hashlib.sha256(original).hexdigest()
    text = original.decode("utf-8")
    if SENTINEL in text:
        raise SystemExit("strict P1R harness already present in source")
    if not text.rstrip().endswith("}"):
        raise SystemExit("subsystem_trait.rs no longer ends with tests-module brace")

    insertion_at = text.rfind("\n}")
    if insertion_at < 0:
        raise SystemExit("could not locate final tests-module brace")
    patched = text[:insertion_at] + RUST_TEST + text[insertion_at:]

    try:
        SUBJECT.write_text(patched, encoding="utf-8")
        subprocess.run(
            [
                "cargo",
                "test",
                "-p",
                "symthaea",
                "--lib",
                "cognitive_loop::subsystem_trait::tests::test_spine_000b_p1r_strict_generated_harness",
            ],
            check=True,
        )
    finally:
        SUBJECT.write_bytes(original)

    restored = SUBJECT.read_bytes()
    restored_sha = hashlib.sha256(restored).hexdigest()
    if restored != original or restored_sha != original_sha:
        raise SystemExit("P1R harness failed to restore production source byte-for-byte")

    print("SPINE-000B-P1R strict Rust production collector harness: PASS")
    print(f"production_source_sha256={original_sha}")
    print("authority=measurement-only")
    print("runtime_evidence_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
