#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

AUDITOR = pathlib.Path(__file__).with_name("verify-continuity-fenced-resource-evidence.py")
CHECKSUMMED = (
    "campaign-manifest.json",
    "campaign-observations.json",
    "campaign-run-context.json",
    "campaign-summary.json",
)


def canonical_write(path: pathlib.Path, obj) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def load(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_sums(root: pathlib.Path) -> None:
    lines = []
    for name in sorted(CHECKSUMMED):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_auditor(root: pathlib.Path, expected_subject: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDITOR), str(root), "--expected-subject", expected_subject],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require_deny(source: pathlib.Path, expected_subject: str, label: str, mutate, needle: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td) / "bundle"
        shutil.copytree(source, root)
        mutate(root)
        rewrite_sums(root)
        result = run_auditor(root, expected_subject)
        if result.returncode != 2 or needle not in result.stderr:
            raise AssertionError(
                f"{label}: expected DENY containing {needle!r}; "
                f"rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=pathlib.Path)
    parser.add_argument("--expected-subject", required=True)
    args = parser.parse_args()

    positive = run_auditor(args.evidence_dir, args.expected_subject)
    if positive.returncode != 0:
        raise AssertionError(f"valid bundle rejected: stdout={positive.stdout!r} stderr={positive.stderr!r}")
    report = json.loads(positive.stdout)
    if report.get("status") != "PASS" or report.get("obligation_count") != 9:
        raise AssertionError(f"unexpected positive report: {report!r}")

    def shadow_manifest(root: pathlib.Path) -> None:
        path = root / "campaign-manifest.json"
        obj = load(path); obj["shadow_authority"] = "permit"; canonical_write(path, obj)

    def mutate_raw_observation(root: pathlib.Path) -> None:
        path = root / "campaign-observations.json"
        obj = load(path); obj["observations"][0]["evidence"]["shadow"] = "mutated"; canonical_write(path, obj)

    def substitute_basis(root: pathlib.Path) -> None:
        path = root / "campaign-observations.json"
        obj = load(path); obj["observations"][0]["required_basis"] = "replay_scenario"; canonical_write(path, obj)

    def substitute_subject(root: pathlib.Path) -> None:
        fake = "f" * 40
        context_path = root / "campaign-run-context.json"
        context = load(context_path); context["subject_sha"] = fake; canonical_write(context_path, context)
        summary_path = root / "campaign-summary.json"
        summary = load(summary_path); summary["subject_sha"] = fake; canonical_write(summary_path, summary)

    def remove_obligation(root: pathlib.Path) -> None:
        observations_path = root / "campaign-observations.json"
        observations = load(observations_path)
        observations["observations"].pop()
        observations["obligation_count"] = 8
        canonical_write(observations_path, observations)

    def forge_oracle_summary(root: pathlib.Path) -> None:
        path = root / "campaign-summary.json"
        obj = load(path)
        obj["campaign_oracle"]["canonical_preimage_sha256"] = "a" * 64
        canonical_write(path, obj)

    def backend_generation_drift(root: pathlib.Path) -> None:
        manifest_path = root / "campaign-manifest.json"
        manifest = load(manifest_path); manifest["backend_generation"] += 1; canonical_write(manifest_path, manifest)
        summary_path = root / "campaign-summary.json"
        summary = load(summary_path)
        summary["campaign_manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        summary["campaign_oracle"]["backend_generation"] = manifest["backend_generation"]
        canonical_write(summary_path, summary)

    require_deny(args.evidence_dir, args.expected_subject, "shadow manifest field", shadow_manifest, "exact fields required")
    require_deny(args.evidence_dir, args.expected_subject, "raw observation mutation", mutate_raw_observation, "observation ID mismatch")
    require_deny(args.evidence_dir, args.expected_subject, "basis substitution", substitute_basis, "wrong evidence basis")
    require_deny(args.evidence_dir, args.expected_subject, "subject substitution", substitute_subject, "unexpected subject")
    require_deny(args.evidence_dir, args.expected_subject, "obligation removal", remove_obligation, "wrong obligation_count")
    require_deny(args.evidence_dir, args.expected_subject, "oracle summary forgery", forge_oracle_summary, "campaign oracle reconstruction mismatch")
    require_deny(args.evidence_dir, args.expected_subject, "backend generation drift", backend_generation_drift, "campaign oracle reconstruction mismatch")

    print(json.dumps({
        "status": "PASS",
        "positive_subject": report["subject_sha"],
        "adversarial_cases": 7,
        "auditor_report": report,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
