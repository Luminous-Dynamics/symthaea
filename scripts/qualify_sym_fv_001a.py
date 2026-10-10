#!/usr/bin/env python3
"""Execute SYM-FV-001A and emit a classification receipt.

The script deliberately separates evidence capture from semantic success. It exits
zero after writing a structurally valid receipt even when extraction/typechecking
fails; the workflow's semantic-gate step decides whether the exact subject reached
ExtractedAndLeanTypechecked.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/formal/sym-fv-001a-aeneas-bind-scalar-v1.json"
OUT = ROOT / ".artifacts/sym-fv-001a"
SOURCE_PATH = "crates/core/symthaea-core/src/hdc/binary_hv.rs"
SOURCE_COMMIT = "2dfddf6027d8eaf62221f8a71be2bb6d1d7bd9a9"
SOURCE_BLOB = "22a56cfaedf5b0cb7d8ff8b73c41ed4caf8d7056"
AENEAS_COMMIT = "b86120db3183b0107eb5f2637b11c424cd06ef1c"
CHARON_COMMIT = "4bd5a29f6e97ce2201ed35251afe5716e53b0a3a"
AENEAS_FLAKE = f"github:AeneasVerif/aeneas/{AENEAS_COMMIT}"
RESULT_CLASSES = {
    "ExtractedAndLeanTypechecked",
    "UnsupportedRustConstruct",
    "UnsupportedDependencyBoundary",
    "TranslatorFailure",
    "LeanGenerationFailure",
    "LeanTypecheckFailure",
    "EnvironmentFailure",
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], *, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "argv": cmd,
        "cwd": str(cwd or ROOT),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_seconds": round(time.time() - started, 3),
    }


def classify_text(text: str, default: str) -> str:
    low = text.lower()
    if any(token in low for token in ("unsupported", "not supported", "unimplemented rust", "unsupported statement")):
        return "UnsupportedRustConstruct"
    if any(token in low for token in ("external definition", "external function", "unmodeled", "opaque dependency")):
        return "UnsupportedDependencyBoundary"
    return default


def write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_function_present(text: str) -> bool:
    pattern = re.compile(
        r"pub fn bind_scalar\(&self, other: &Self\) -> Self\s*\{"
        r".*?let mut result = \[0u8; 2048\];"
        r".*?for i in 0\.\.2048\s*\{"
        r".*?result\[i\] = self\.0\[i\] \^ other\.0\[i\];"
        r".*?Self\(result\)",
        re.S,
    )
    return bool(pattern.search(text))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": "symthaea-sym-fv-001a-receipt-v1",
        "source_commit": SOURCE_COMMIT,
        "source_blob": SOURCE_BLOB,
        "source_path": SOURCE_PATH,
        "aeneas_commit": AENEAS_COMMIT,
        "charon_commit": CHARON_COMMIT,
        "result_class": "EnvironmentFailure",
        "semantic_success": False,
        "commands": {},
        "generated_lean": [],
        "external_model_templates": [],
        "llbc_sha256": None,
        "lean_toolchain": None,
        "aeneas_reported_version": None,
        "charon_reported_version": None,
    }

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        receipt["manifest_sha256"] = sha256_file(MANIFEST_PATH)
        receipt["head_commit"] = run(["git", "rev-parse", "HEAD"])["stdout"].strip()

        source_blob_at_subject = run(["git", "rev-parse", f"{SOURCE_COMMIT}:{SOURCE_PATH}"])
        receipt["commands"]["source_blob_at_subject"] = source_blob_at_subject
        if source_blob_at_subject["returncode"] != 0 or source_blob_at_subject["stdout"].strip() != SOURCE_BLOB:
            receipt["failure_detail"] = "frozen source commit/path no longer resolves to the required blob"
            write_json(OUT / "receipt.json", receipt)
            return 0

        head_blob = run(["git", "rev-parse", f"HEAD:{SOURCE_PATH}"])
        receipt["commands"]["head_source_blob"] = head_blob
        receipt["head_source_blob"] = head_blob["stdout"].strip() if head_blob["returncode"] == 0 else None
        if receipt["head_source_blob"] != SOURCE_BLOB:
            receipt["failure_detail"] = "PR head changed the exact production source blob"
            write_json(OUT / "receipt.json", receipt)
            return 0

        source_text = (ROOT / SOURCE_PATH).read_text(encoding="utf-8")
        if not source_function_present(source_text):
            receipt["failure_detail"] = "exact bind_scalar XOR kernel not found at PR head"
            write_json(OUT / "receipt.json", receipt)
            return 0

        if shutil.which("nix") is None:
            receipt["failure_detail"] = "nix executable unavailable"
            write_json(OUT / "receipt.json", receipt)
            return 0

        meta = run(["nix", "flake", "metadata", "--json", AENEAS_FLAKE])
        receipt["commands"]["aeneas_flake_metadata"] = meta
        if meta["returncode"] != 0:
            receipt["failure_detail"] = "unable to resolve pinned Aeneas flake"
            write_json(OUT / "receipt.json", receipt)
            return 0
        meta_json = json.loads(meta["stdout"])
        receipt["aeneas_flake_metadata"] = {
            "resolvedUrl": meta_json.get("resolvedUrl"),
            "revision": meta_json.get("revision"),
            "path": meta_json.get("path"),
        }
        aeneas_src = pathlib.Path(meta_json["path"])
        charon_pin = (aeneas_src / "charon-pin").read_text(encoding="utf-8").splitlines()[-1].strip()
        receipt["resolved_charon_pin"] = charon_pin
        if charon_pin != CHARON_COMMIT:
            receipt["failure_detail"] = f"Aeneas charon-pin drifted: {charon_pin}"
            write_json(OUT / "receipt.json", receipt)
            return 0

        charon_version = run(["nix", "run", f"{AENEAS_FLAKE}#charon", "-L", "--", "--version"])
        aeneas_version = run(["nix", "run", AENEAS_FLAKE, "-L", "--", "-version"])
        receipt["commands"]["charon_version"] = charon_version
        receipt["commands"]["aeneas_version"] = aeneas_version
        receipt["charon_reported_version"] = (charon_version["stdout"] or charon_version["stderr"]).strip()
        receipt["aeneas_reported_version"] = (aeneas_version["stdout"] or aeneas_version["stderr"]).strip()

        work = OUT / "work"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        llbc = (work / "symthaea_core.llbc").resolve()
        generated = (work / "lean").resolve()
        generated.mkdir()

        # A fresh target directory forces rustc to execute, avoiding Charon's warm-cache no-op trap.
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str((work / "cargo-target").resolve())
        charon = run(
            [
                "nix", "run", f"{AENEAS_FLAKE}#charon", "-L", "--",
                "cargo", "--preset=aeneas", f"--dest-file={llbc}",
            ],
            cwd=ROOT / "crates/core/symthaea-core",
            env=env,
        )
        receipt["commands"]["charon_extract"] = charon
        if charon["returncode"] != 0 or not llbc.exists():
            receipt["result_class"] = classify_text(charon["stdout"] + "\n" + charon["stderr"], "TranslatorFailure")
            receipt["failure_detail"] = "Charon did not produce the frozen LLBC subject"
            write_json(OUT / "receipt.json", receipt)
            return 0
        receipt["llbc_sha256"] = sha256_file(llbc)
        receipt["llbc_bytes"] = llbc.stat().st_size

        aeneas = run(
            [
                "nix", "run", AENEAS_FLAKE, "-L", "--",
                "-backend", "lean", str(llbc),
                "-dest", str(generated),
                "-subdir", "/SymthaeaFV001A/Generated",
                "-namespace", "SymthaeaFV001A.Generated",
                "-split-files",
            ]
        )
        receipt["commands"]["aeneas_generate"] = aeneas
        if aeneas["returncode"] != 0:
            receipt["result_class"] = classify_text(aeneas["stdout"] + "\n" + aeneas["stderr"], "LeanGenerationFailure")
            receipt["failure_detail"] = "Aeneas Lean generation failed"
            write_json(OUT / "receipt.json", receipt)
            return 0

        lean_files = sorted(generated.rglob("*.lean"))
        receipt["generated_lean"] = [
            {"path": str(p.relative_to(generated)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in lean_files
        ]
        templates = sorted(generated.rglob("*_Template.lean"))
        receipt["external_model_templates"] = [
            {"path": str(p.relative_to(generated)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in templates
        ]
        if templates:
            receipt["result_class"] = "UnsupportedDependencyBoundary"
            receipt["failure_detail"] = "generated Lean requires external-model templates; no proof-hole substitution is permitted"
            write_json(OUT / "receipt.json", receipt)
            return 0

        lean_toolchain_src = aeneas_src / "backends/lean/lean-toolchain"
        receipt["lean_toolchain"] = lean_toolchain_src.read_text(encoding="utf-8").strip()
        shutil.copy2(lean_toolchain_src, generated / "lean-toolchain")
        lakefile = generated / "lakefile.lean"
        backend = aeneas_src / "backends/lean"
        lakefile.write_text(
            "import Lake\n"
            "open Lake DSL\n"
            f"require aeneas from \"{backend}\"\n"
            "package «sym-fv-001a-generated» {}\n"
            "@[default_target]\n"
            "lean_lib «SymthaeaFV001A»\n",
            encoding="utf-8",
        )
        typecheck = run(
            ["nix", "develop", AENEAS_FLAKE, "-L", "--command", "bash", "-lc", "lake build"],
            cwd=generated,
        )
        receipt["commands"]["lean_typecheck"] = typecheck
        if typecheck["returncode"] != 0:
            receipt["result_class"] = "LeanTypecheckFailure"
            receipt["failure_detail"] = "generated Lean did not typecheck under Aeneas's pinned Lean backend toolchain"
            write_json(OUT / "receipt.json", receipt)
            return 0

        receipt["result_class"] = "ExtractedAndLeanTypechecked"
        receipt["semantic_success"] = True
        receipt["failure_detail"] = None
        write_json(OUT / "receipt.json", receipt)
        return 0
    except Exception as exc:  # receipt-first: unexpected environment/tool failures remain evidence.
        receipt["result_class"] = "EnvironmentFailure"
        receipt["semantic_success"] = False
        receipt["failure_detail"] = f"{type(exc).__name__}: {exc}"
        write_json(OUT / "receipt.json", receipt)
        return 0


if __name__ == "__main__":
    sys.exit(main())
