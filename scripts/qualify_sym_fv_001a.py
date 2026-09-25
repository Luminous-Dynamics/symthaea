#!/usr/bin/env python3
"""Exact SYM-FV-001A Rust -> Charon -> Aeneas -> Lean qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
from typing import Any


SUCCESS = "ExtractedAndLeanTypechecked"
ALLOWED_RESULTS = {
    SUCCESS,
    "UnsupportedRustConstruct",
    "UnsupportedDependencyBoundary",
    "TranslatorFailure",
    "LeanGenerationFailure",
    "LeanTypecheckFailure",
    "EnvironmentFailure",
}


class QualificationFailure(RuntimeError):
    def __init__(self, classification: str, message: str):
        super().__init__(message)
        if classification not in ALLOWED_RESULTS:
            raise ValueError(f"invalid result classification: {classification}")
        self.classification = classification


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, cwd: Path | None = None) -> str:
    p = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        raise QualificationFailure(
            "EnvironmentFailure",
            f"git {' '.join(args)} failed: {p.stderr.strip()}",
        )
    return p.stdout.strip()


def run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(p.stdout or "", encoding="utf-8")
    return p


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    generated = out / "generated"
    generated.mkdir(exist_ok=True)
    tool_src = out / "aeneas-src"
    lean_project = out / "lean-project"
    llbc = out / "symthaea-core-bind-scalar.llbc"
    receipt_path = out / "receipt.json"

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subject = manifest["subject"]
    tools = manifest["toolchain"]

    receipt: dict[str, Any] = {
        "schema": "symthaea-fv-001a-receipt-v1",
        "manifest": str(args.manifest),
        "result": None,
        "reason": None,
        "qualification_head": None,
        "qualification_tree": None,
        "subject": subject,
        "toolchain": tools,
        "stages": {},
        "artifacts": {},
        "external_model_census": [],
        "generated_lean_census": [],
    }

    try:
        if set(manifest["allowed_results"]) != ALLOWED_RESULTS:
            raise QualificationFailure("EnvironmentFailure", "manifest result-class set drifted")

        head = git("rev-parse", "HEAD", cwd=repo)
        tree = git("rev-parse", "HEAD^{tree}", cwd=repo)
        receipt["qualification_head"] = head
        receipt["qualification_tree"] = tree

        source_path = repo / subject["path"]
        if not source_path.is_file():
            raise QualificationFailure("EnvironmentFailure", "bound Rust source path is missing")

        historical_blob = git(
            "rev-parse",
            f'{subject["source_commit"]}:{subject["path"]}',
            cwd=repo,
        )
        head_blob = git("rev-parse", f'HEAD:{subject["path"]}', cwd=repo)
        if historical_blob != subject["blob_sha"] or head_blob != subject["blob_sha"]:
            raise QualificationFailure(
                "EnvironmentFailure",
                "source blob identity drifted from the frozen subject",
            )

        source = source_path.read_text(encoding="utf-8")
        required_source_tokens = [
            "pub fn bind_scalar(&self, other: &Self) -> Self",
            "let mut result = [0u8; 2048];",
            "for i in 0..2048",
            "result[i] = self.0[i] ^ other.0[i];",
            "Self(result)",
        ]
        missing = [token for token in required_source_tokens if token not in source]
        if missing:
            raise QualificationFailure(
                "EnvironmentFailure",
                f"bound source no longer contains expected scalar kernel tokens: {missing}",
            )
        bind_scalar_defs = len(re.findall(r"\bfn\s+bind_scalar\s*\(", source))
        if bind_scalar_defs != 1:
            raise QualificationFailure(
                "EnvironmentFailure",
                f"expected exactly one source bind_scalar definition, found {bind_scalar_defs}",
            )
        if subject.get("charon_start_from") != "crate::hdc::binary_hv::_::bind_scalar":
            raise QualificationFailure(
                "EnvironmentFailure",
                "Charon start-from root drifted from the reviewed inherent-method matcher",
            )
        receipt["stages"]["subject_binding"] = "PASS"
        receipt["stages"]["source_bind_scalar_definition_count"] = bind_scalar_defs

        nix = shutil.which("nix")
        if nix is None:
            raise QualificationFailure("EnvironmentFailure", "nix executable unavailable")
        receipt["stages"]["nix_available"] = "PASS"

        if tool_src.exists():
            shutil.rmtree(tool_src)
        p = run_logged(
            ["git", "init", str(tool_src)],
            cwd=repo,
            log_path=logs / "aeneas-source-init.log",
        )
        if p.returncode != 0:
            raise QualificationFailure("EnvironmentFailure", "failed to initialize Aeneas source checkout")

        for cmd, log_name in [
            (["git", "-C", str(tool_src), "remote", "add", "origin", tools["aeneas_repo"]],
             "aeneas-source-remote.log"),
            (["git", "-C", str(tool_src), "fetch", "--depth=1", "origin", tools["aeneas_revision"]],
             "aeneas-source-fetch.log"),
            (["git", "-C", str(tool_src), "checkout", "--detach", "FETCH_HEAD"],
             "aeneas-source-checkout.log"),
        ]:
            p = run_logged(cmd, cwd=repo, log_path=logs / log_name)
            if p.returncode != 0:
                raise QualificationFailure("EnvironmentFailure", f"Aeneas source pin setup failed: {log_name}")

        actual_aeneas = git("-C", str(tool_src), "rev-parse", "HEAD", cwd=repo)
        if actual_aeneas != tools["aeneas_revision"]:
            raise QualificationFailure("EnvironmentFailure", "Aeneas source revision mismatch")

        upstream_lock = json.loads((tool_src / "flake.lock").read_text(encoding="utf-8"))
        actual_charon = upstream_lock["nodes"]["charon"]["locked"]["rev"]
        if actual_charon != tools["charon_revision"]:
            raise QualificationFailure("EnvironmentFailure", "Aeneas->Charon lock drifted")

        actual_lean = (tool_src / "backends/lean/lean-toolchain").read_text(encoding="utf-8").strip()
        if actual_lean != tools["lean_toolchain"]:
            raise QualificationFailure("EnvironmentFailure", "Aeneas Lean toolchain drifted")
        receipt["stages"]["tool_pin_verification"] = "PASS"

        flake_uri = f'github:AeneasVerif/aeneas/{tools["aeneas_revision"]}'
        charon_cmd = [
            nix,
            "--extra-experimental-features",
            "nix-command flakes",
            "run",
            f"{flake_uri}#charon",
            "--",
            "cargo",
            "--preset=aeneas",
            "--lib",
            "--start-from",
            subject["charon_start_from"],
            "--dest-file",
            str(llbc),
        ]
        p = run_logged(
            charon_cmd,
            cwd=repo / subject["crate_dir"],
            log_path=logs / "charon.log",
        )
        receipt["stages"]["charon_exit_code"] = p.returncode
        if p.returncode != 0:
            raise QualificationFailure("TranslatorFailure", "Charon did not produce the frozen LLBC subject")
        if not llbc.is_file() or llbc.stat().st_size == 0:
            raise QualificationFailure("TranslatorFailure", "Charon exited zero but LLBC artifact is missing/empty")

        llbc_text = llbc.read_text(encoding="utf-8", errors="replace")
        bind_scalar_mentions = llbc_text.count("bind_scalar")
        if bind_scalar_mentions < 1:
            raise QualificationFailure(
                "TranslatorFailure",
                "retained LLBC does not contain the selected bind_scalar item name",
            )
        receipt["stages"]["llbc_subject_census"] = {
            "bind_scalar_mentions": bind_scalar_mentions,
            "start_from": subject["charon_start_from"],
        }
        receipt["artifacts"]["llbc"] = {
            "sha256": sha256_file(llbc),
            "bytes": llbc.stat().st_size,
        }

        aeneas_cmd = [
            nix,
            "--extra-experimental-features",
            "nix-command flakes",
            "run",
            flake_uri,
            "--",
            "-backend",
            "lean",
            "-dest",
            str(generated),
            str(llbc),
        ]
        p = run_logged(
            aeneas_cmd,
            cwd=repo,
            log_path=logs / "aeneas.log",
        )
        receipt["stages"]["aeneas_exit_code"] = p.returncode
        if p.returncode != 0:
            raise QualificationFailure("LeanGenerationFailure", "Aeneas Lean generation failed")

        lean_files = sorted(generated.rglob("*.lean"))
        if not lean_files:
            raise QualificationFailure("LeanGenerationFailure", "Aeneas emitted no Lean files")

        receipt["generated_lean_census"] = [
            {
                "path": str(path.relative_to(generated)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in lean_files
        ]
        templates = [path for path in lean_files if "_Template" in path.name]
        receipt["external_model_census"] = [str(path.relative_to(generated)) for path in templates]
        if templates:
            raise QualificationFailure(
                "UnsupportedDependencyBoundary",
                "Aeneas emitted external-model templates; exact dependency models are required before Lean admission",
            )

        if lean_project.exists():
            shutil.rmtree(lean_project)
        lean_project.mkdir()
        shutil.copy2(tool_src / "backends/lean/lean-toolchain", lean_project / "lean-toolchain")

        top_level = [path for path in lean_files if path.parent == generated]
        if not top_level:
            raise QualificationFailure(
                "LeanGenerationFailure",
                "generated Lean has no top-level module suitable for an isolated Lake target",
            )
        for path in lean_files:
            rel = path.relative_to(generated)
            dest = lean_project / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)

        lakefile = [
            "import Lake",
            "open Lake DSL",
            f'require aeneas from "{tool_src / "backends/lean"}"',
            'package «sym_fv_001a» {}',
            "",
        ]
        roots = []
        for path in top_level:
            stem = path.stem
            if not stem.isidentifier():
                raise QualificationFailure(
                    "LeanGenerationFailure",
                    f"generated top-level Lean module is not a simple Lake identifier: {stem}",
                )
            roots.append(stem)
            lakefile.append(f"@[default_target] lean_lib {stem}")
        (lean_project / "lakefile.lean").write_text("\n".join(lakefile) + "\n", encoding="utf-8")
        receipt["artifacts"]["lake_roots"] = roots

        build_script = "set -euo pipefail; cd " + shlex.quote(str(lean_project)) + "; lake update; lake build"
        lean_cmd = [
            nix,
            "--extra-experimental-features",
            "nix-command flakes",
            "develop",
            flake_uri,
            "--command",
            "bash",
            "-lc",
            build_script,
        ]
        p = run_logged(
            lean_cmd,
            cwd=repo,
            log_path=logs / "lean-build.log",
            env={**os.environ, "CI": "1"},
        )
        receipt["stages"]["lean_build_exit_code"] = p.returncode
        if p.returncode != 0:
            raise QualificationFailure("LeanTypecheckFailure", "generated Lean failed exact-toolchain Lake build")

        receipt["result"] = SUCCESS
        receipt["reason"] = "exact Rust subject extracted, Lean generated, and generated project typechecked"
        receipt["stages"]["qualification"] = "PASS"
        write_receipt(receipt_path, receipt)
        print(SUCCESS)
        return 0

    except QualificationFailure as failure:
        receipt["result"] = failure.classification
        receipt["reason"] = str(failure)
        receipt["stages"]["qualification"] = "FAIL"
        write_receipt(receipt_path, receipt)
        print(f"{failure.classification}: {failure}", file=sys.stderr)
        return 1
    except Exception as failure:
        receipt["result"] = "EnvironmentFailure"
        receipt["reason"] = f"unhandled qualifier exception: {type(failure).__name__}: {failure}"
        receipt["stages"]["qualification"] = "FAIL"
        write_receipt(receipt_path, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
