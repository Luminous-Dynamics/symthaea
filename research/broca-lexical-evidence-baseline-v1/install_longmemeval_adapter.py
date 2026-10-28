#!/usr/bin/env python3
"""Install the Rust BM25 adapter into an exact, clean LongMemEval-V2 checkout.

This deliberately patches only the pinned upstream revision named below. It
prints the planned changes unless --apply is supplied. New upstream commits
must be reviewed and pinned explicitly before this script is updated.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

PINNED_UPSTREAM_COMMIT = "2cc8c540bdb87fe6761629b585e727e1c4704520"
MAX_JSONL_LINE_BYTES = 32 * 1024 * 1024


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one insertion anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path,
                        help="Path to a clean LongMemEval-V2 Git checkout")
    parser.add_argument("--expected-commit", default=PINNED_UPSTREAM_COMMIT,
                        help="Exact upstream commit to patch (default: pinned 2026-08-09 revision)")
    parser.add_argument("--apply", action="store_true",
                        help="Write the changes; without this flag, validate and print the patch plan only")
    args = parser.parse_args()

    upstream = args.upstream.expanduser().resolve()
    if not upstream.is_dir() or not (upstream / ".git").exists():
        parser.error(f"--upstream is not a Git checkout: {upstream}")

    head = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != args.expected_commit:
        raise RuntimeError(
            f"Refusing to patch upstream HEAD {head}; expected {args.expected_commit}. "
            "Audit the newer upstream and update this installer pin intentionally."
        )

    dirty = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    if dirty.strip():
        raise RuntimeError("Refusing to patch a dirty upstream worktree; commit or stash its changes first.")

    repo_root = Path(__file__).resolve().parents[2]
    adapter_source = repo_root / "research" / "broca-lexical-evidence-baseline-v1" / "broca_rust_memory.py"
    if not adapter_source.is_file():
        raise RuntimeError(f"adapter source not found: {adapter_source}")

    memory_path = upstream / "memory_modules" / "memory.py"
    run_eval_path = upstream / "evaluation" / "run_eval.py"
    memory_text = memory_path.read_text(encoding="utf-8")
    run_eval_text = run_eval_path.read_text(encoding="utf-8")

    memory_import = "from .rag import RagMemory  # noqa: E402,F401\n"
    memory_text_new = replace_once(
        memory_text,
        memory_import,
        memory_import + "from .broca_rust import BrocaRustMemory  # noqa: E402,F401\n",
        "memory registry",
    )

    # The worker is process-isolated per Memory instance and serializes IPC per
    # instance, so it can use the upstream harness's non-shared parallel prompt
    # builder without sharing mutable retrieval state across questions.
    nonshared_anchor = (
        'NONSHARED_PARALLEL_MEMORY_TYPES = {\n'
        '    "codex",\n'
        '    "agentrunbook_c",\n'
        '    "agentrunbook_c_v2",\n'
    )
    memory_text_harness_path = upstream / "evaluation" / "harness.py"
    harness_text = memory_text_harness_path.read_text(encoding="utf-8")
    harness_text_new = replace_once(
        harness_text,
        nonshared_anchor,
        nonshared_anchor + '    "broca_rust",\n',
        "non-shared parallel memory allowlist",
    )

    method_anchor = '    "agentrunbook_c_v2",\n'
    run_eval_text_new = replace_once(
        run_eval_text,
        method_anchor,
        method_anchor + '    "broca_rust",\n',
        "METHODS registry",
    )
    parser_anchor = '    parser.add_argument("--output-dir", required=True)\n'
    run_eval_text_new = replace_once(
        run_eval_text_new,
        parser_anchor,
        parser_anchor
        + '    parser.add_argument("--broca-worker", default=os.getenv("BROCA_LME_WORKER"))\n'
        + '    parser.add_argument("--broca-worker-timeout", type=float, default=float(os.getenv("BROCA_LME_WORKER_TIMEOUT", "60")))\n',
        "CLI worker settings",
    )
    config_anchor = 'def build_memory_config(args: argparse.Namespace, data_root: Path) -> dict[str, object]:\n'
    config_body = (
        config_anchor
        + '    if args.method == "broca_rust":\n'
        + '        if not args.broca_worker:\n'
        + '            raise ValueError("broca_rust requires --broca-worker or BROCA_LME_WORKER")\n'
        + '        worker_path = Path(args.broca_worker).expanduser().resolve()\n'
        + '        if not worker_path.is_file() or not os.access(worker_path, os.X_OK):\n'
        + '            raise ValueError(f"Broca worker binary is missing or not executable: {worker_path}")\n'
        + '        if not (0.1 <= args.broca_worker_timeout <= 3600.0):\n'
        + '            raise ValueError("--broca-worker-timeout must be in [0.1, 3600]")\n'
        + '        return {\n'
        + '            "memory_type": "broca_rust",\n'
        + '            "memory_params": {\n'
        + '                "worker_path": str(worker_path),\n'
        + '                "scope_id": args.domain,\n'
        + '                "timeout_seconds": args.broca_worker_timeout,\n'
        + f'                "max_jsonl_line_bytes": {MAX_JSONL_LINE_BYTES},\n'
        + '                "bm25_params": {"k1": 1.2, "b": 0.75, "max_items": 5},\n'
        + '            },\n'
        + '        }\n'
    )
    run_eval_text_new = replace_once(
        run_eval_text_new,
        config_anchor,
        config_body,
        "memory config builder",
    )

    destination_adapter = upstream / "memory_modules" / "broca_rust.py"
    if destination_adapter.exists():
        if destination_adapter.read_bytes() != adapter_source.read_bytes():
            raise RuntimeError(f"refusing to overwrite a different existing adapter: {destination_adapter}")
    paths = [
        memory_path,
        run_eval_path,
        memory_text_harness_path,
        destination_adapter,
    ]
    print(f"Upstream commit pinned: {head}")
    print("Planned changes:")
    for item in paths:
        print(f"  - {item.relative_to(upstream)}")
    print("  - adds --broca-worker and --broca-worker-timeout to evaluation/run_eval.py")
    print("  - adds broca_rust to METHODS, build_memory_config and the non-shared parallel allowlist")
    if not args.apply:
        print("Dry run only. Re-run with --apply to write the changes.")
        return 0

    memory_path.write_text(memory_text_new, encoding="utf-8")
    run_eval_path.write_text(run_eval_text_new, encoding="utf-8")
    memory_text_harness_path.write_text(harness_text_new, encoding="utf-8")
    if not destination_adapter.exists():
        destination_adapter.write_bytes(adapter_source.read_bytes())

    print("Installed. Review the upstream diff before running:")
    subprocess.run(["git", "-C", str(upstream), "diff", "--check"], check=False)
    subprocess.run(["git", "-C", str(upstream), "diff", "--stat"], check=False)
    print(
        "Next: build symthaea-broca's broca-lme-worker binary, then run the official "
        "small tier on both domains with --method broca_rust and the pinned reader/evaluator."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"install_longmemeval_adapter: {error}", file=sys.stderr)
        raise SystemExit(1)
