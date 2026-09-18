#!/usr/bin/env python3
"""Fail closed if SYM-RSI unit-test regions call sealed measurement APIs.

The protected fresh / verification / OOD partitions are experiment evidence, not
ordinary test fixtures. Unit tests may exercise pure classifiers, digest logic,
replay validation, and synthetic parent qualification, but they must not directly
acquire or execute the sealed measurement partitions.

The scanner examines source text beginning at each file's first `#[cfg(test)]`
marker. This is deliberately conservative: anything after that marker is treated as
part of the test-side authority surface for this theorem.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RSI_DIR = ROOT / "src" / "consciousness" / "recursive_improvement"
TEST_MARKER = "#[cfg(test)]"

FORBIDDEN = (
    "run_fresh_c_vs_a",
    "run_fresh_c_vs_a_after_parent_c",
    "acquire_dream_verification_corpus",
    "acquire_dream_verification_corpus_after_parent_c",
    "run_fresh_d_vs_c",
    "run_fresh_d_vs_c_after_parent_c",
    "run_ood_d_vs_c",
    "run_ood_d_vs_c_after_parent_c",
)
CALL_RE = re.compile(r"\b(" + "|".join(map(re.escape, FORBIDDEN)) + r")\s*\(")


def strip_line_comment(line: str) -> str:
    """Remove ordinary // comments while leaving code before them intact."""
    return line.split("//", 1)[0]


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    text = path.read_text(encoding="utf-8")
    marker = text.find(TEST_MARKER)
    if marker < 0:
        return []

    tail = text[marker:]
    findings: list[tuple[int, str, str]] = []
    start_line = text[:marker].count("\n") + 1
    for offset, raw_line in enumerate(tail.splitlines()):
        code = strip_line_comment(raw_line)
        match = CALL_RE.search(code)
        if match:
            findings.append((start_line + offset, match.group(1), raw_line.rstrip()))
    return findings


def self_test() -> int:
    safe = """fn production() { run_fresh_c_vs_a(); }\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn pure_classifier() { assert!(true); }\n    // run_fresh_d_vs_c() is forbidden, even though this comment names it.\n}\n"""
    unsafe = """#[cfg(test)]\nmod tests {\n    #[test]\n    fn bad() { run_ood_d_vs_c(); }\n}\n"""
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        safe_path = directory_path / "safe.rs"
        unsafe_path = directory_path / "unsafe.rs"
        safe_path.write_text(safe, encoding="utf-8")
        unsafe_path.write_text(unsafe, encoding="utf-8")
        if scan_file(safe_path):
            print("self-test failed: safe test region was rejected", file=sys.stderr)
            return 1
        findings = scan_file(unsafe_path)
        if len(findings) != 1 or findings[0][1] != "run_ood_d_vs_c":
            print("self-test failed: sealed measurement call was not detected", file=sys.stderr)
            return 1
    print("SYM-RSI measurement-boundary scanner self-test: PASS")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        return self_test()
    if len(sys.argv) != 1:
        print(f"usage: {Path(sys.argv[0]).name} [--self-test]", file=sys.stderr)
        return 2

    if not RSI_DIR.is_dir():
        print(f"missing SYM-RSI source directory: {RSI_DIR}", file=sys.stderr)
        return 2

    files = sorted(RSI_DIR.glob("*.rs"))
    if not files:
        print(f"no Rust modules found under {RSI_DIR}", file=sys.stderr)
        return 2

    all_findings: list[tuple[Path, int, str, str]] = []
    test_regions = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        if TEST_MARKER in text:
            test_regions += 1
        for line, api, source in scan_file(path):
            all_findings.append((path, line, api, source))

    if test_regions == 0:
        print("no #[cfg(test)] regions found; refusing to claim the boundary", file=sys.stderr)
        return 2

    if all_findings:
        print("SYM-RSI protected-measurement boundary violation:", file=sys.stderr)
        for path, line, api, source in all_findings:
            rel = path.relative_to(ROOT)
            print(f"  {rel}:{line}: direct test-region call to {api}: {source}", file=sys.stderr)
        return 1

    print(
        "SYM-RSI unit-test measurement boundary: PASS "
        f"({len(files)} modules scanned; {test_regions} test regions)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
