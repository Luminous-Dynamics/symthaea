#!/usr/bin/env python3
"""Execute a small, preregistered semantic mutation campaign for the VLEB oracle.

This is CI test tooling, not production accounting code. A mutant only counts as
"killed" after it compiles successfully and the locked contract suite fails.
Compilation failures, harness errors, and timeouts are unresolved—not kills.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
VLEB_ROOT = ROOT / "tools/vleb"
CRATE_REL = Path("ledger-reconcile")
SOURCE_REL = Path("ledger-reconcile/src/lib.rs")
TASK_ID = "vleb-ledger-reconcile-0001"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_exactly_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected exactly one source match, found {count}")
    return source.replace(old, new, 1)


def duplicate_processing(source: str) -> str:
    old = '''        if let Some(previous) = seen.get(&transaction_id) {
            if previous == &transaction {
                totals.exact_duplicate_rows_ignored = totals
                    .exact_duplicate_rows_ignored
                    .checked_add(1)
                    .ok_or_else(|| LedgerError::new("duplicate-row counter overflow"))?;
                continue;
            }
            return Err(LedgerError::new(format!(
                "conflicting duplicate transaction_id {transaction_id:?}; input rejected"
            )));
        }'''
    new = '''        if let Some(previous) = seen.get(&transaction_id) {
            if previous == &transaction {
                // MUTANT: exact duplicates fall through and are applied again.
            } else {
                return Err(LedgerError::new(format!(
                    "conflicting duplicate transaction_id {transaction_id:?}; input rejected"
                )));
            }
        }'''
    return replace_exactly_once(source, old, new, "VLEB-LDG-001 duplicate processing")


def conflicting_duplicate_accepted(source: str) -> str:
    old = '''            return Err(LedgerError::new(format!(
                "conflicting duplicate transaction_id {transaction_id:?}; input rejected"
            )));'''
    new = '''            // MUTANT: conflicting transaction-ID reuse is silently ignored.
            continue;'''
    return replace_exactly_once(source, old, new, "VLEB-LDG-002 conflict acceptance")


def unchecked_overflow(source: str) -> str:
    old = ".checked_add(amount_minor)"
    count = source.count(old)
    if count != 4:
        raise ValueError(f"VLEB-LDG-004 overflow: expected 4 aggregate additions, found {count}")
    return source.replace(old, ".wrapping_add(amount_minor)")


def credit_debit_swap(source: str) -> str:
    old = '''        match side {
            Side::Credit => {
                account.credits_minor = account.credits_minor
                    .checked_add(amount_minor)
                    .ok_or_else(|| LedgerError::new("account credit total overflow"))?;
                totals.credits_minor = totals.credits_minor
                    .checked_add(amount_minor)
                    .ok_or_else(|| LedgerError::new("global credit total overflow"))?;
            }
            Side::Debit => {
                account.debits_minor = account.debits_minor
                    .checked_add(amount_minor)
                    .ok_or_else(|| LedgerError::new("account debit total overflow"))?;
                totals.debits_minor = totals.debits_minor
                    .checked_add(amount_minor)
                    .ok_or_else(|| LedgerError::new("global debit total overflow"))?;
            }
        }'''
    swapped = old.replace("Side::Credit =>", "Side::__TEMP__ =>", 1)
    swapped = swapped.replace("Side::Debit =>", "Side::Credit =>", 1)
    swapped = swapped.replace("Side::__TEMP__ =>", "Side::Debit =>", 1)
    return replace_exactly_once(source, old, swapped, "VLEB-LDG-007 credit/debit swap")


MUTANTS: list[tuple[str, str, Callable[[str], str]]] = [
    ("VLEB-LDG-001", "duplicate_processing", duplicate_processing),
    ("VLEB-LDG-002", "other_semantic", conflicting_duplicate_accepted),
    ("VLEB-LDG-004", "rounding_or_truncation", unchecked_overflow),
    ("VLEB-LDG-007", "other_semantic", credit_debit_swap),
]


def run_command(argv: list[str], cwd: Path, timeout_seconds: int) -> tuple[int | None, bytes, bool]:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        return completed.returncode, completed.stdout, False
    except subprocess.TimeoutExpired as error:
        raw = error.stdout or b""
        if isinstance(raw, str):
            raw = raw.encode("utf-8", errors="replace")
        return None, raw + b"\nHARNESS TIMEOUT\n", True


def write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for machine-readable results and raw compile/test logs",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        print("Refusing to use a source-tree directory as the evidence output directory.", file=sys.stderr)
        return 2
    if output_dir.exists():
        shutil.rmtree(output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    reference_path = VLEB_ROOT / SOURCE_REL
    reference_source = reference_path.read_text(encoding="utf-8")
    base_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()

    versions = {}
    for tool, args_list in (
        ("rustc", ["rustc", "--version"]),
        ("cargo", ["cargo", "--version"]),
    ):
        result = subprocess.run(
            args_list,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        versions[tool] = {"exit_code": result.returncode, "output": result.stdout.strip()}

    report: dict[str, object] = {
        "schema_version": "vleb-mutation-campaign-v1",
        "task_id": TASK_ID,
        "base_revision": base_revision,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "toolchains": versions,
        "reference_source_sha256": sha256(reference_source.encode("utf-8")),
        "rules": {
            "semantic_kill": "The mutant compiles and the locked contract test suite fails.",
            "compile_failure": "unresolved; not counted as a semantic kill.",
            "timeout_or_harness_error": "unresolved; not counted as a semantic kill.",
        },
        "results": [],
        "campaign_pass": False,
    }
    write_report(output_dir / "mutation-results.json", report)
    all_killed = True

    for mutation_id, mutation_class, transform in MUTANTS:
        record: dict[str, object] = {
            "mutation_id": mutation_id,
            "mutation_class": mutation_class,
            "compilable": None,
            "equivalence_status": "non_equivalent",
            "observed_verdict": "unresolved",
            "compile_exit_code": None,
            "test_exit_code": None,
            "compile_log_sha256": None,
            "test_log_sha256": None,
            "evidence_sha256": None,
            "mutant_source_sha256": None,
            "timeout": False,
            "notes": "",
        }
        report["results"].append(record)
        write_report(output_dir / "mutation-results.json", report)

        try:
            mutant_source = transform(reference_source)
        except Exception as error:  # mutation construction must fail closed
            record["notes"] = f"mutation construction error: {error}"
            all_killed = False
            write_report(output_dir / "mutation-results.json", report)
            print(f"{mutation_id}: UNRESOLVED (could not construct mutant): {error}")
            continue

        record["mutant_source_sha256"] = sha256(mutant_source.encode("utf-8"))
        with tempfile.TemporaryDirectory(prefix=f"vleb-{mutation_id.lower()}-") as temporary:
            temporary_root = Path(temporary)
            copied_vleb = temporary_root / "tools" / "vleb"
            shutil.copytree(
                VLEB_ROOT,
                copied_vleb,
                ignore=shutil.ignore_patterns("target", "__pycache__", ".pytest_cache"),
            )
            mutant_path = copied_vleb / SOURCE_REL
            mutant_path.write_text(mutant_source, encoding="utf-8")
            manifest = copied_vleb / CRATE_REL / "Cargo.toml"

            compile_code, compile_output, compile_timed_out = run_command(
                ["cargo", "test", "--locked", "--manifest-path", str(manifest), "--no-run"],
                temporary_root,
                180,
            )
            compile_log = logs_dir / f"{mutation_id}.compile.log"
            compile_log.write_bytes(compile_output)
            record["compile_log_sha256"] = sha256(compile_output)
            record["compile_exit_code"] = compile_code
            record["timeout"] = compile_timed_out

            if compile_code != 0 or compile_timed_out:
                record["compilable"] = False if not compile_timed_out else None
                record["observed_verdict"] = "unresolved"
                record["notes"] = "compile failed or timed out; this is not counted as a semantic kill"
                record["evidence_sha256"] = sha256(compile_output + b"\0")
                all_killed = False
                write_report(output_dir / "mutation-results.json", report)
                print(f"{mutation_id}: UNRESOLVED (compile gate failed)")
                continue

            record["compilable"] = True
            test_code, test_output, test_timed_out = run_command(
                ["cargo", "test", "--locked", "--manifest-path", str(manifest)],
                temporary_root,
                180,
            )
            test_log = logs_dir / f"{mutation_id}.test.log"
            test_log.write_bytes(test_output)
            record["test_log_sha256"] = sha256(test_output)
            record["test_exit_code"] = test_code
            record["timeout"] = test_timed_out
            record["evidence_sha256"] = sha256(compile_output + b"\0" + test_output)

            if not test_timed_out and test_code is not None and test_code != 0 and b"test result: FAILED" in test_output:
                record["observed_verdict"] = "rejected"
                record["notes"] = "compilable mutant rejected by the locked test suite"
            elif test_code == 0:
                record["observed_verdict"] = "accepted"
                record["notes"] = "BUG: semantically incorrect mutant survived the locked test suite"
                all_killed = False
            else:
                record["observed_verdict"] = "unresolved"
                record["notes"] = "test command timed out or failed without an explicit failed-test summary"
                all_killed = False

            write_report(output_dir / "mutation-results.json", report)
            print(f"{mutation_id}: {record['observed_verdict'].upper()} ({record['notes']})")

    report["campaign_pass"] = all_killed and all(
        result["compilable"] is True and result["observed_verdict"] == "rejected"
        for result in report["results"]
    )
    write_report(output_dir / "mutation-results.json", report)
    print(
        f"VLEB mutation campaign: {len(MUTANTS)} registered probes; "
        f"{sum(r['observed_verdict'] == 'rejected' for r in report['results'])} killed; "
        f"{sum(r['observed_verdict'] == 'accepted' for r in report['results'])} survived; "
        f"{sum(r['observed_verdict'] == 'unresolved' for r in report['results'])} unresolved."
    )
    return 0 if report["campaign_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
