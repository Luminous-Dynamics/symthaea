#!/usr/bin/env python3
"""Guard capability claims against evidence inflation in the language matrix."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

MATRIX_PATH = Path(__file__).with_name("language-capability-matrix.json")
STAGES = {
    "dispatch_recognition": {"supported", "unsupported", "not_audited"},
    "parser": {"supported", "unsupported", "not_audited"},
    "generation": {"supported", "unsupported", "not_audited"},
    "compile_or_runtime": {
        "backend_declared", "evaluator_backend_declared",
        "unsupported_in_central_dispatch", "not_audited",
    },
    "test_execution": {"supported", "partially_evidenced", "not_run", "not_audited"},
    "independent_semantic_verification": {"supported", "not_run", "not_audited"},
    "held_out_benchmark": {"measured", "not_run"},
}


def validate_matrix(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != "vleb-language-capability-matrix-v1":
        errors.append("schema_version")
    source = data.get("source_snapshot", {})
    if source.get("commit_sha") != "ffac9a0ea3172f0622d9b1a0ab031284722fc70f":
        errors.append("source_revision_drift")
    if not source.get("source_file") or not source.get("source_sha256_git_blob"):
        errors.append("source_evidence_missing")

    rows = data.get("languages", [])
    names = [row.get("language") for row in rows]
    if not names or len(names) != len(set(names)):
        errors.append("language_names_missing_or_duplicate")

    for row in rows:
        name = row.get("language", "<unnamed>")
        for stage, accepted in STAGES.items():
            value = row.get(stage)
            if value not in accepted:
                errors.append(f"invalid_stage:{name}:{stage}")
        if row.get("held_out_benchmark") == "measured":
            errors.append(f"unsubstantiated_benchmark_claim:{name}")
        if row.get("dispatch_recognition") == "supported" and not row.get("evidence"):
            errors.append(f"supported_dispatch_without_evidence:{name}")
        if row.get("generation") == "supported" and row.get("held_out_benchmark") != "measured":
            # Generator support is a source-level capability claim, not benchmark success;
            # nevertheless it must not imply benchmark results. Keep all current generation
            # statuses conservative until independently audited.
            if row.get("generation") != "not_audited":
                errors.append(f"generation_claim_requires_explicit_review:{name}")

    lookup = {row["language"]: row for row in rows}
    for name in ("COBOL", "Assembly / HLASM"):
        row = lookup.get(name)
        if row is None:
            errors.append(f"required_legacy_language_missing:{name}")
            continue
        if row.get("dispatch_recognition") != "unsupported":
            errors.append(f"legacy_dispatch_must_fail_closed:{name}")
        if row.get("compile_or_runtime") != "unsupported_in_central_dispatch":
            errors.append(f"legacy_backend_claim_mismatch:{name}")
        if row.get("generation") != "not_audited":
            errors.append(f"legacy_generation_overclaim:{name}")
        if row.get("held_out_benchmark") != "not_run":
            errors.append(f"legacy_benchmark_overclaim:{name}")

    for name in ("Rust", "Python", "Nix"):
        row = lookup.get(name)
        if row is None:
            errors.append(f"known_dispatch_language_missing:{name}")
            continue
        if row.get("dispatch_recognition") != "supported":
            errors.append(f"known_dispatch_language_mismatch:{name}")
        if row.get("held_out_benchmark") != "not_run":
            errors.append(f"known_language_benchmark_overclaim:{name}")

    if data.get("reporting_rule") != "Never collapse recognition, parsing, generation, execution, test execution, independent verification and held-out performance into a single 'language supported' boolean.":
        errors.append("capability_stage_separation_missing")
    return errors


class LanguageCapabilityMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))

    def test_matrix_claim_boundaries_are_valid(self) -> None:
        self.assertEqual(validate_matrix(self.matrix), [])

    def test_cobol_and_hlasm_are_not_misreported_as_executable(self) -> None:
        rows = {r["language"]: r for r in self.matrix["languages"]}
        for name in ("COBOL", "Assembly / HLASM"):
            self.assertEqual(rows[name]["dispatch_recognition"], "unsupported")
            self.assertEqual(rows[name]["generation"], "not_audited")
            self.assertEqual(rows[name]["held_out_benchmark"], "not_run")

    def test_dispatch_support_does_not_imply_parser_or_generation(self) -> None:
        rows = {r["language"]: r for r in self.matrix["languages"]}
        for name in ("Rust", "Python", "Nix"):
            self.assertEqual(rows[name]["dispatch_recognition"], "supported")
            self.assertEqual(rows[name]["parser"], "not_audited")
            self.assertEqual(rows[name]["generation"], "not_audited")

    def test_no_language_has_a_held_out_score_without_a_run(self) -> None:
        self.assertTrue(all(r["held_out_benchmark"] == "not_run" for r in self.matrix["languages"]))

    def test_tampered_supported_claim_requires_source_evidence(self) -> None:
        data = json.loads(json.dumps(self.matrix))
        data["languages"][0]["evidence"] = []
        self.assertIn("supported_dispatch_without_evidence:Rust", validate_matrix(data))

    def test_matrix_rejects_an_invented_cobol_backend(self) -> None:
        data = json.loads(json.dumps(self.matrix))
        row = next(r for r in data["languages"] if r["language"] == "COBOL")
        row["dispatch_recognition"] = "supported"
        self.assertIn("legacy_dispatch_must_fail_closed:COBOL", validate_matrix(data))


if __name__ == "__main__":
    unittest.main(verbosity=2)
