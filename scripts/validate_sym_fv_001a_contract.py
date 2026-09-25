#!/usr/bin/env python3
"""Static contract checks for SYM-FV-001A.

This validator is intentionally independent of Aeneas execution. It freezes the
subject, translator pins, result taxonomy, workflow anti-cancellation rule, and
claim ceiling before the expensive extraction job runs.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/formal/sym-fv-001a-aeneas-bind-scalar-v1.json"
SOURCE = ROOT / "crates/core/symthaea-core/src/hdc/binary_hv.rs"
RUNNER = ROOT / "scripts/qualify_sym_fv_001a.py"
WORKFLOW = ROOT / ".github/workflows/sym-fv-001a-aeneas-bind.yml"

EXPECTED = {
    "source_commit": "2dfddf6027d8eaf62221f8a71be2bb6d1d7bd9a9",
    "source_blob": "22a56cfaedf5b0cb7d8ff8b73c41ed4caf8d7056",
    "aeneas_commit": "b86120db3183b0107eb5f2637b11c424cd06ef1c",
    "charon_commit": "4bd5a29f6e97ce2201ed35251afe5716e53b0a3a",
}
EXPECTED_CLASSES = {
    "ExtractedAndLeanTypechecked",
    "UnsupportedRustConstruct",
    "UnsupportedDependencyBoundary",
    "TranslatorFailure",
    "LeanGenerationFailure",
    "LeanTypecheckFailure",
    "EnvironmentFailure",
}


class ContractError(RuntimeError):
    pass


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise ContractError(msg)


def validate_manifest(data: dict) -> None:
    require(data.get("schema") == "symthaea-sym-fv-001a-v1", "wrong schema")
    require(data.get("issue") == 5713, "wrong issue")
    src = data.get("source", {})
    tr = data.get("translator", {})
    require(src.get("commit") == EXPECTED["source_commit"], "source commit drift")
    require(src.get("blob") == EXPECTED["source_blob"], "source blob drift")
    require(src.get("path") == "crates/core/symthaea-core/src/hdc/binary_hv.rs", "source path drift")
    require(src.get("function") == "BinaryHV::bind_scalar", "function drift")
    require(src.get("byte_count") == 2048, "dimension/byte-count drift")
    require(src.get("operation") == "bytewise-xor", "operation drift")
    require(tr.get("aeneas_commit") == EXPECTED["aeneas_commit"], "Aeneas pin drift")
    require(tr.get("charon_commit") == EXPECTED["charon_commit"], "Charon pin drift")
    require(set(data.get("result_classes", [])) == EXPECTED_CLASSES, "result taxonomy drift")
    require(data.get("claim_ceiling") == "ExtractedAndLeanTypechecked", "claim ceiling drift")
    require(data.get("successor", {}).get("issue") == 5716, "wrong refinement successor")
    require(data.get("successor", {}).get("required_class") == "ExtractedAndLeanTypechecked", "successor admission weakened")


def validate_source(text: str) -> None:
    pattern = re.compile(
        r"pub fn bind_scalar\(&self, other: &Self\) -> Self\s*\{"
        r".*?let mut result = \[0u8; 2048\];"
        r".*?for i in 0\.\.2048\s*\{"
        r".*?result\[i\] = self\.0\[i\] \^ other\.0\[i\];"
        r".*?Self\(result\)",
        re.S,
    )
    require(bool(pattern.search(text)), "exact scalar XOR kernel not found")


def validate_runner(text: str) -> None:
    for value in EXPECTED.values():
        require(value in text, f"runner missing frozen value {value}")
    require("CARGO_TARGET_DIR" in text, "fresh Charon target-directory guard missing")
    require("external_model_templates" in text, "external-model census missing")
    require("UnsupportedDependencyBoundary" in text, "dependency-boundary classification missing")
    require("semantic_success" in text, "semantic success bit missing")
    require("generated Lean requires external-model templates" in text, "template fail-closed rule missing")


def validate_workflow(text: str) -> None:
    require("cancel-in-progress: false" in text, "formal evidence workflow may be cancelled by concurrency")
    require("11d5960a326750d5838078e36cf38b85af677262" in text, "checkout action not immutable-pinned")
    require("13d8dd58da0234aa297dedd986986ccb8e7f3e24" in text, "install-nix action not immutable-pinned")
    require("ea165f8d65b6e75b540449e92b4886f43607fa02" in text, "artifact action not immutable-pinned")
    require("validate_sym_fv_001a_contract.py" in text, "static validator not enforced")
    require("qualify_sym_fv_001a.py" in text, "qualifier not enforced")
    require("ExtractedAndLeanTypechecked" in text, "semantic gate missing")
    require("if: always()" in text, "receipt upload is not failure-resilient")


def negative_controls(manifest: dict) -> None:
    mutants: list[tuple[str, dict]] = []

    m = copy.deepcopy(manifest)
    m["source"]["blob"] = "0" * 40
    mutants.append(("source-blob-drift", m))

    m = copy.deepcopy(manifest)
    m["translator"]["aeneas_commit"] = "f" * 40
    mutants.append(("aeneas-pin-drift", m))

    m = copy.deepcopy(manifest)
    m["result_classes"].remove("UnsupportedDependencyBoundary")
    mutants.append(("result-class-deletion", m))

    m = copy.deepcopy(manifest)
    m["claim_ceiling"] = "RustRefinementProved"
    mutants.append(("claim-inflation", m))

    for name, mutant in mutants:
        try:
            validate_manifest(mutant)
        except ContractError:
            print(f"negative control {name}: rejected")
        else:
            raise ContractError(f"negative control {name} was accepted")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    validate_source(SOURCE.read_text(encoding="utf-8"))
    validate_runner(RUNNER.read_text(encoding="utf-8"))
    validate_workflow(WORKFLOW.read_text(encoding="utf-8"))
    negative_controls(manifest)
    print("SYM-FV-001A static contract: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ContractError as exc:
        print(f"SYM-FV-001A static contract: FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
