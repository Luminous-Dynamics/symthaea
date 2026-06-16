#!/usr/bin/env python3
"""Fail when the canonical Symthaea workspace has structural drift."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
IGNORED_SYMLINK_ROOTS = (
    ROOT / "data",
    ROOT / "papers",
    ROOT / "docs" / "demos",
    ROOT / "crates" / "symthaea-core" / "docs" / "demos",
    ROOT / "target",
    ROOT / "target_test",
)


def cargo_metadata() -> dict:
    result = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def manifest_path_dependencies(manifest: Path) -> list[tuple[str, str]]:
    with manifest.open("rb") as handle:
        document = tomllib.load(handle)

    dependencies: list[tuple[str, str]] = []
    sections = ["dependencies", "dev-dependencies", "build-dependencies"]
    for section in sections:
        for name, spec in document.get(section, {}).items():
            if isinstance(spec, dict) and "path" in spec:
                dependencies.append((name, spec["path"]))

    for target, target_config in document.get("target", {}).items():
        for section in sections:
            for name, spec in target_config.get(section, {}).items():
                if isinstance(spec, dict) and "path" in spec:
                    dependencies.append((f"{target}:{name}", spec["path"]))
    return dependencies


def is_ignored_symlink(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        any(path == root or root in path.parents for root in IGNORED_SYMLINK_ROOTS)
        or relative.parts[0].startswith("mujoco-")
    )


def main() -> int:
    errors: list[str] = []
    metadata = cargo_metadata()
    workspace_ids = set(metadata["workspace_members"])
    packages = [package for package in metadata["packages"] if package["id"] in workspace_ids]

    names: dict[str, list[str]] = {}
    for package in packages:
        names.setdefault(package["name"], []).append(package["manifest_path"])
        manifest = Path(package["manifest_path"])
        for dependency, raw_path in manifest_path_dependencies(manifest):
            dependency_path = Path(raw_path)
            if dependency_path.is_absolute():
                errors.append(f"{manifest}: absolute path for {dependency}: {raw_path}")
                continue
            resolved = (manifest.parent / dependency_path).resolve()
            if not resolved.exists():
                errors.append(f"{manifest}: missing path for {dependency}: {raw_path}")

    for name, manifests in sorted(names.items()):
        if len(manifests) > 1:
            errors.append(f"duplicate workspace package {name}: {', '.join(manifests)}")

    for directory, dirnames, filenames in os.walk(ROOT, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".git", "target", "target_test"}
        ]
        for name in [*dirnames, *filenames]:
            path = directory_path / name
            if path.is_symlink() and not path.exists() and not is_ignored_symlink(path):
                errors.append(f"broken source symlink: {path.relative_to(ROOT)}")

    app_manifest = ROOT / "Cargo.toml"
    app = next(
        (package for package in packages if Path(package["manifest_path"]) == app_manifest),
        None,
    )
    if app is None:
        errors.append("application package at repository root is not a workspace member")
    else:
        discovered = {
            Path(target["src_path"]).stem
            for target in app["targets"]
            if "test" in target["kind"]
        }
        expected = {path.stem for path in (app_manifest.parent / "tests").glob("*.rs")}
        missing = sorted(expected - discovered)
        if missing:
            errors.append(f"integration tests absent from Cargo metadata: {', '.join(missing)}")

    if errors:
        print("Workspace audit failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        f"Workspace audit passed: {len(packages)} packages, "
        f"{len(app['targets']) if app else 0} application targets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
