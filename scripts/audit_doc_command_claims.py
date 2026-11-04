#!/usr/bin/env python3
"""Offline audit of Cargo commands documented in tracked Markdown.

This CI helper does not invoke Cargo, build scripts, candidate code, tests,
graders, or the network. Relative manifest paths are resolved from repo root.
"""
from __future__ import annotations
import argparse
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

UNAVAILABLE = re.compile(
    r"\b(?:not currently runnable|not runnable|not executable|"
    r"design[- ](?:only|specification)|intended interfaces? only|"
    r"not yet implemented|not implemented|unavailable|"
    r"manifest(?: and binary source)? (?:is |are )?(?:absent|missing)|"
    r"(?:manifest|binary source|executable) (?:is |are )?(?:absent|missing)|"
    r"not present|does not exist|were not found)\b",
    re.IGNORECASE,
)
DYNAMIC = ("$", chr(96), "*", "?", "{", "}")
FENCE = re.compile(r"^ {0,3}(\x60{3,}|~{3,})(.*)$")


def fenced_blocks(markdown: str):
    lines = markdown.splitlines()
    char = None
    length = 0
    info = ""
    start = 0
    body = []
    for number, line in enumerate(lines, 1):
        if char is None:
            match = FENCE.match(line)
            if match:
                marker, info = match.groups()
                char, length, start, body = marker[0], len(marker), number, []
            continue
        closing = rf"^ {{0,3}}{re.escape(char)}{{{length},}}[ \t]*$"
        if re.match(closing, line):
            yield start, info.strip(), "\n".join(body)
            char, length, info, body = None, 0, "", []
        else:
            body.append(line)


def cargo_references(source: str):
    # Preserve shell line continuations, then separate distinct commands so
    # --bin targets from one Cargo invocation are not applied to another.
    flattened = re.sub(r"\\\r?\n", " ", source)
    commands = re.split(r"&&|\|\||[;|\n]", flattened)
    references = []
    errors = []
    for command in commands:
        try:
            tokens = shlex.split(command, comments=True, posix=True)
        except ValueError as error:
            if "--manifest-path" in command:
                errors.append(f"cannot tokenize Cargo command: {error}")
            continue
        if "cargo" not in tokens:
            continue

        bins = []
        for index, token in enumerate(tokens):
            if token == "--bin":
                if index + 1 >= len(tokens):
                    errors.append("--bin has no target name")
                    break
                bins.append(tokens[index + 1])
            elif token.startswith("--bin="):
                bins.append(token.split("=", 1)[1])

        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "--manifest-path":
                if index + 1 >= len(tokens):
                    errors.append("--manifest-path has no path")
                    break
                references.append((tokens[index + 1], bins))
                index += 2
                continue
            if token.startswith("--manifest-path="):
                references.append((token.split("=", 1)[1], bins))
            index += 1
    return references, errors
def is_dynamic(value: str) -> bool:
    return not value or any(marker in value for marker in DYNAMIC)


def repo_path(root: Path, value: str):
    candidate = Path(value)
    if candidate.is_absolute():
        return None
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


def has_unavailable_note(markdown: str, path: str) -> bool:
    normalized = path.replace("\\", "/").removeprefix("./")
    return normalized in markdown and UNAVAILABLE.search(markdown) is not None


def bin_source(manifest: Path, name: str):
    try:
        parsed = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot parse {manifest}: {error}") from error

    for target in parsed.get("bin", []):
        if target.get("name") == name and target.get("path"):
            return (manifest.parent / target["path"]).resolve()

    package = parsed.get("package", {}).get("name", "").replace("-", "_")
    candidates = [
        manifest.parent / "src" / "bin" / f"{name}.rs",
        manifest.parent / "src" / "bin" / name / "main.rs",
    ]
    if name == package:
        candidates.insert(0, manifest.parent / "src" / "main.rs")
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def audit_markdown(root: Path, file: Path):
    text = file.read_text(encoding="utf-8")
    failures, checked, unavailable, dynamic = [], 0, 0, 0
    for line, _info, source in fenced_blocks(text):
        refs, errors = cargo_references(source)
        failures.extend(f"{file}:{line}: {error}" for error in errors)
        for reference, bins in refs:
            if is_dynamic(reference):
                dynamic += 1
                continue
            manifest = repo_path(root, reference)
            if manifest is None:
                dynamic += 1
                continue
            checked += 1
            if not manifest.is_file():
                if has_unavailable_note(text, reference):
                    unavailable += 1
                else:
                    failures.append(
                        f"{file}:{line}: missing Cargo manifest {reference!r}; "
                        "mark this command design-only/unavailable or add the checked-in manifest"
                    )
                continue
            for name in bins:
                if is_dynamic(name):
                    dynamic += 1
                    continue
                target = bin_source(manifest, name)
                if target is None or not target.is_file():
                    if has_unavailable_note(text, reference):
                        unavailable += 1
                    else:
                        failures.append(
                            f"{file}:{line}: binary target {name!r} is absent from {reference!r}"
                        )
    return failures, checked, unavailable, dynamic


def tracked_markdown(root: Path):
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def self_test():
    sample = "cargo test --locked --manifest-path " + chr(92) + "\n tools/example/Cargo.toml --bin example_audit"
    refs, errors = cargo_references(sample)
    assert not errors and refs == [("tools/example/Cargo.toml", ["example_audit"])]
    refs, errors = cargo_references('cargo test --manifest-path "$TOOLS_DIR/Cargo.toml"')
    assert not errors and is_dynamic(refs[0][0])
    note = "Design specification only: tools/missing/Cargo.toml is not currently runnable."
    assert has_unavailable_note(note, "tools/missing/Cargo.toml")
    assert not has_unavailable_note("Run this command.", "tools/missing/Cargo.toml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.self_test:
        self_test()
        print("DOC_COMMAND_AUDIT_SELF_TEST_PASS")
        return 0
    try:
        files = tracked_markdown(root)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"DOC_COMMAND_AUDIT_ERROR: cannot enumerate tracked Markdown: {error}", file=sys.stderr)
        return 2

    failures, checked, unavailable, dynamic = [], 0, 0, 0
    for file in files:
        try:
            found, count, missing, dynamic_count = audit_markdown(root, file)
        except (OSError, UnicodeError) as error:
            failures.append(f"{file}: cannot read Markdown: {error}")
            continue
        failures.extend(found)
        checked += count
        unavailable += missing
        dynamic += dynamic_count

    if failures:
        for failure in failures:
            print(f"DOC_COMMAND_AUDIT_FAIL: {failure}", file=sys.stderr)
        print(
            f"DOC_COMMAND_AUDIT_SUMMARY: markdown={len(files)} manifests={checked} "
            f"documented_unavailable={unavailable} dynamic={dynamic} errors={len(failures)}",
            file=sys.stderr,
        )
        return 1
    print(
        f"DOC_COMMAND_AUDIT_PASS: markdown={len(files)} manifests={checked} "
        f"documented_unavailable={unavailable} dynamic={dynamic} errors=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
