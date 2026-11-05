#!/usr/bin/env python3
"""Offline audit of Cargo commands documented in tracked Markdown.

This CI helper does not invoke Cargo, build scripts, candidate code, tests,
graders, or the network. Relative manifest paths are resolved from repo root.
"""
from __future__ import annotations
import argparse
import fnmatch
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
    # flags from one Cargo invocation are never applied to another.
    flattened = re.sub(r"\\\r?\n", " ", source)
    commands = re.split(r"&&|\|\||[;|\n]", flattened)
    manifest_refs = []
    package_refs = []
    errors = []

    for command in commands:
        try:
            tokens = shlex.split(command, comments=True, posix=True)
        except ValueError as error:
            if "--manifest-path" in command or re.search(r"(?<!\w)-p(?:\s|=)", command):
                errors.append(f"cannot tokenize Cargo command: {error}")
            continue

        if "cargo" not in tokens:
            continue

        # Cargo options after -- belong to the invoked program, not Cargo.
        cargo_tokens = tokens[:tokens.index("--")] if "--" in tokens else tokens
        bins = []
        manifests = []
        packages = []
        index = 0
        while index < len(cargo_tokens):
            token = cargo_tokens[index]
            if token == "--manifest-path":
                if index + 1 >= len(cargo_tokens):
                    errors.append("--manifest-path has no path")
                    break
                manifests.append(cargo_tokens[index + 1])
                index += 2
                continue
            if token.startswith("--manifest-path="):
                manifests.append(token.split("=", 1)[1])
            elif token == "--bin":
                if index + 1 >= len(cargo_tokens):
                    errors.append("--bin has no target name")
                    break
                bins.append(cargo_tokens[index + 1])
                index += 2
                continue
            elif token.startswith("--bin="):
                bins.append(token.split("=", 1)[1])
            elif token in {"-p", "--package"}:
                if index + 1 >= len(cargo_tokens):
                    errors.append(f"{token} has no package selector")
                    break
                packages.append(cargo_tokens[index + 1])
                index += 2
                continue
            elif token.startswith("--package="):
                packages.append(token.split("=", 1)[1])
            elif token.startswith("-p") and len(token) > 2:
                packages.append(token[2:].removeprefix("="))
            index += 1

        manifest_refs.extend((manifest, bins) for manifest in manifests)
        anchor = manifests[0] if len(manifests) == 1 else None
        package_refs.extend((package, bins, anchor) for package in packages)

    return manifest_refs, package_refs, errors
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


def has_unavailable_note(markdown: str, path: str, target: str | None = None) -> bool:
    normalized = path.replace("\\", "/").removeprefix("./")
    path_is_named = normalized in markdown
    target_is_named = target is None or target in markdown
    return path_is_named and target_is_named and UNAVAILABLE.search(markdown) is not None


def _path_pattern_matches(path: str, pattern: str) -> bool:
    """Match Cargo workspace member patterns without allowing * to cross /."""
    path_parts = Path(path).parts
    pattern_parts = Path(pattern).parts

    def match_at(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        part = pattern_parts[pattern_index]
        if part == "**":
            return (
                match_at(path_index, pattern_index + 1)
                or (
                    path_index < len(path_parts)
                    and match_at(path_index + 1, pattern_index)
                )
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], part)
            and match_at(path_index + 1, pattern_index + 1)
        )

    return match_at(0, 0)


def _load_manifest(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot parse Cargo manifest {path}: {error}") from error


def tracked_cargo_manifests(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return [
        root / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item == b"Cargo.toml" or item.endswith(b"/Cargo.toml")
    ]


def workspace_package_index(
    root: Path,
    manifests: list[Path],
    anchor_manifest: Path | None = None,
) -> dict[str, list[Path]]:
    """Index packages in the workspace selected by the documented command.

    With no explicit --manifest-path, commands are interpreted from repository
    root. With an explicit manifest, the nearest ancestor workspace is used;
    a standalone package manifest is treated as a one-package workspace.
    """
    anchor = (anchor_manifest or (root / "Cargo.toml")).resolve()
    if not anchor.is_file():
        return {}

    workspace_manifest = None
    current = anchor.parent
    while True:
        candidate = current / "Cargo.toml"
        if candidate.is_file():
            document = _load_manifest(candidate)
            if "workspace" in document:
                workspace_manifest = candidate.resolve()
                break
        if current == root.resolve() or current.parent == current:
            break
        current = current.parent

    if workspace_manifest is None:
        document = _load_manifest(anchor)
        package = document.get("package")
        if not package or not package.get("name"):
            return {}
        return {package["name"]: [anchor]}

    workspace = _load_manifest(workspace_manifest).get("workspace", {})
    members = workspace.get("members", [])
    excludes = workspace.get("exclude", [])
    workspace_root = workspace_manifest.parent
    index: dict[str, list[Path]] = {}

    for manifest in manifests:
        resolved = manifest.resolve()
        if resolved == workspace_manifest:
            is_member = True
        else:
            try:
                relative_dir = resolved.parent.relative_to(workspace_root).as_posix()
            except ValueError:
                continue
            is_member = any(_path_pattern_matches(relative_dir, pattern) for pattern in members)
            if is_member and any(_path_pattern_matches(relative_dir, pattern) for pattern in excludes):
                is_member = False
        if not is_member:
            continue

        package = _load_manifest(resolved).get("package")
        if not package or not package.get("name"):
            continue
        index.setdefault(package["name"], []).append(resolved)

    return index


def resolve_package_selector(index: dict[str, list[Path]], selector: str):
    """Resolve only exact package names; version-qualified specs stay out of scope."""
    if not selector or is_dynamic(selector) or "@" in selector:
        return None, "dynamic_or_version_qualified"
    matches = index.get(selector, [])
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, "not_a_member"
    return None, "ambiguous_package_name"
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


def audit_markdown(
    root: Path,
    file: Path,
    cargo_manifests: list[Path],
):
    text = file.read_text(encoding="utf-8")
    failures, checked, unavailable, dynamic, packages_checked = [], 0, 0, 0, 0

    for line, _info, block in fenced_blocks(text):
        manifest_refs, package_refs, errors = cargo_references(block)
        failures.extend(f"{file}:{line}: {error}" for error in errors)

        for reference, bins in manifest_refs:
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
                    if has_unavailable_note(text, reference, name):
                        unavailable += 1
                    else:
                        failures.append(
                            f"{file}:{line}: binary target {name!r} is absent from {reference!r}"
                        )

        for selector, bins, anchor in package_refs:
            if is_dynamic(selector) or "@" in selector:
                dynamic += 1
                continue
            anchor_path = None
            if anchor is not None:
                if is_dynamic(anchor):
                    dynamic += 1
                    continue
                anchor_path = repo_path(root, anchor)
                if anchor_path is None or not anchor_path.is_file():
                    dynamic += 1
                    continue
            try:
                index = workspace_package_index(root, cargo_manifests, anchor_path)
            except ValueError as error:
                failures.append(f"{file}:{line}: {error}")
                continue

            package_manifest, reason = resolve_package_selector(index, selector)
            if package_manifest is None:
                if reason == "dynamic_or_version_qualified":
                    dynamic += 1
                elif has_unavailable_note(text, selector):
                    unavailable += 1
                else:
                    detail = "not a member of the selected Cargo workspace" if reason == "not_a_member" else "ambiguous in the selected Cargo workspace"
                    failures.append(
                        f"{file}:{line}: Cargo package selector {selector!r} is {detail}; "
                        "use an explicit --manifest-path or document the unavailable selector"
                    )
                continue

            packages_checked += 1
            for name in bins:
                if is_dynamic(name):
                    dynamic += 1
                    continue
                target = bin_source(package_manifest, name)
                if target is None or not target.is_file():
                    if has_unavailable_note(text, selector, name):
                        unavailable += 1
                    else:
                        failures.append(
                            f"{file}:{line}: binary target {name!r} is absent from package {selector!r}"
                        )

    return failures, checked, unavailable, dynamic, packages_checked
def tracked_markdown(root: Path):
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def self_test():
    sample = "cargo test --locked --manifest-path " + chr(92) + "\n tools/example/Cargo.toml --bin example_audit"
    manifests, packages, errors = cargo_references(sample)
    assert not errors and manifests == [("tools/example/Cargo.toml", ["example_audit"])]
    assert packages == []

    multi = (
        "cargo test --manifest-path tools/first/Cargo.toml --bin first\n"
        "cargo test --manifest-path tools/second/Cargo.toml --bin second"
    )
    manifests, packages, errors = cargo_references(multi)
    assert not errors and manifests == [
        ("tools/first/Cargo.toml", ["first"]),
        ("tools/second/Cargo.toml", ["second"]),
    ]
    assert packages == []

    package_flags = (
        "cargo test -p symthaea-core --bin core_audit\n"
        "cargo run --package=symthaea-cli --bin cli_audit"
    )
    manifests, packages, errors = cargo_references(package_flags)
    assert not errors and manifests == []
    assert packages == [
        ("symthaea-core", ["core_audit"], None),
        ("symthaea-cli", ["cli_audit"], None),
    ]

    after_separator = "cargo test --manifest-path tools/a/Cargo.toml -- --bin not_a_cargo_target"
    manifests, packages, errors = cargo_references(after_separator)
    assert not errors and manifests == [("tools/a/Cargo.toml", [])]

    refs, packages, errors = cargo_references('cargo test --manifest-path "$TOOLS_DIR/Cargo.toml"')
    assert not errors and is_dynamic(refs[0][0])

    assert _path_pattern_matches("crates/core/symthaea-core", "crates/core/*")
    assert not _path_pattern_matches("crates/core/nested/symthaea-core", "crates/core/*")
    assert _path_pattern_matches("crates/core/nested/symthaea-core", "crates/**")
    assert not _path_pattern_matches("crates/domains/spark-engine", "crates/domains/*/src")

    fixture = {
        "symthaea-core": [Path("crates/core/symthaea-core/Cargo.toml")],
        "duplicate": [Path("one/Cargo.toml"), Path("two/Cargo.toml")],
    }
    assert resolve_package_selector(fixture, "symthaea-core")[0] == fixture["symthaea-core"][0]
    assert resolve_package_selector(fixture, "missing")[1] == "not_a_member"
    assert resolve_package_selector(fixture, "duplicate")[1] == "ambiguous_package_name"

    note = "Design specification only: tools/missing/Cargo.toml is not currently runnable."
    assert has_unavailable_note(note, "tools/missing/Cargo.toml")
    assert not has_unavailable_note("Run this command.", "tools/missing/Cargo.toml")
    target_note = note + " --bin example_audit has no checked-in source."
    assert has_unavailable_note(target_note, "tools/missing/Cargo.toml", "example_audit")
    assert not has_unavailable_note(target_note, "tools/missing/Cargo.toml", "different_target")

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
        cargo_manifests = tracked_cargo_manifests(root)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"DOC_COMMAND_AUDIT_ERROR: cannot enumerate tracked files: {error}", file=sys.stderr)
        return 2

    failures = []
    checked = unavailable = dynamic = packages_checked = 0
    for file in files:
        try:
            found, count, missing, dynamic_count, package_count = audit_markdown(
                root, file, cargo_manifests
            )
        except (OSError, UnicodeError) as error:
            failures.append(f"{file}: cannot read Markdown: {error}")
            continue
        failures.extend(found)
        checked += count
        unavailable += missing
        dynamic += dynamic_count
        packages_checked += package_count

    if failures:
        for failure in failures:
            print(f"DOC_COMMAND_AUDIT_FAIL: {failure}", file=sys.stderr)
        print(
            f"DOC_COMMAND_AUDIT_SUMMARY: markdown_files={len(files)} "
            f"manifest_references_checked={checked} package_references_checked={packages_checked} "
            f"documented_unavailable_references={unavailable} "
            f"dynamic_references={dynamic} errors={len(failures)}",
            file=sys.stderr,
        )
        return 1
    print(
        f"DOC_COMMAND_AUDIT_PASS: markdown_files={len(files)} "
        f"manifest_references_checked={checked} package_references_checked={packages_checked} "
        f"documented_unavailable_references={unavailable} "
        f"dynamic_references={dynamic} errors=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
