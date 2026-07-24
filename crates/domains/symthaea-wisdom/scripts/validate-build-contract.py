#!/usr/bin/env python3
"""Validate the crate's toolchain, feature, and dependency contract without Cargo."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARGO = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
TOOLCHAIN = tomllib.loads((ROOT / "rust-toolchain.toml").read_text(encoding="utf-8"))
LIB = (ROOT / "src" / "lib.rs").read_text(encoding="utf-8")

errors: list[str] = []
package = CARGO.get("package", {})
features = CARGO.get("features", {})
dependencies = CARGO.get("dependencies", {})
toolchain = TOOLCHAIN.get("toolchain", {})

rust_version = package.get("rust-version")
channel = toolchain.get("channel")
if not isinstance(rust_version, str) or not re.fullmatch(r"\d+\.\d+", rust_version):
    errors.append("package.rust-version must be an explicit major.minor string")
elif not isinstance(channel, str) or not channel.startswith(f"{rust_version}."):
    errors.append(
        f"rust-toolchain channel {channel!r} does not pin a patch release for MSRV {rust_version!r}"
    )

if package.get("edition") != "2024":
    errors.append("the crate must remain on Rust edition 2024")
if toolchain.get("profile") != "minimal":
    errors.append("rust-toolchain profile must be minimal")
if sorted(toolchain.get("components", [])) != ["clippy", "rustfmt"]:
    errors.append("rust-toolchain components must be exactly clippy and rustfmt")

if features.get("default") != ["hardened-daemon-startup"]:
    errors.append("default features must expose only hardened-daemon-startup")
if features.get("hardened-daemon-startup") != []:
    errors.append("hardened-daemon-startup must not pull optional dependencies")
if features.get("legacy-direct-startup") != []:
    errors.append("legacy-direct-startup must remain an explicit empty escape hatch")
if features.get("postgres-sync-driver") != ["dep:postgres"]:
    errors.append("postgres-sync-driver must map only to dep:postgres")

postgres = dependencies.get("postgres")
if not isinstance(postgres, dict) or postgres.get("optional") is not True:
    errors.append("postgres must remain an optional dependency")
for name, specification in dependencies.items():
    if isinstance(specification, dict) and specification.get("workspace") is True:
        errors.append(f"dependency {name!r} unexpectedly relies on workspace inheritance")
if "serde" in dependencies:
    errors.append("serde is not used by this crate and must not be reintroduced without code usage")

required_cfg = '''feature = "legacy-direct-startup",\n    not(feature = "hardened-daemon-startup")'''
if LIB.count(required_cfg) < 2:
    errors.append("legacy startup exports are not guarded by hardened-startup precedence")
if LIB.count('#[cfg(feature = "postgres-sync-driver")]') < 4:
    errors.append("PostgreSQL implementation modules/exports are not fully feature-gated")

for path in sorted((ROOT / "src").glob("*.rs")):
    data = path.read_bytes()
    if b"\0" in data:
        errors.append(f"literal NUL byte found in {path.relative_to(ROOT)}")

if errors:
    for error in errors:
        print(f"build-contract error: {error}", file=sys.stderr)
    raise SystemExit(1)

print(
    "validated build contract: "
    f"Rust {channel}, {len(features)} features, {len(dependencies)} dependencies"
)
