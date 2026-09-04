#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fail-closed static contract for Spore boot CI routing.

The contract distinguishes three different facts:

1. a path triggers focused CI;
2. a path has a concrete validation owner in an explicitly audited focused
   qualification script;
3. a path may therefore be excluded from general PR CI.

(1) alone never implies (3). During the pre-boot migration the qualification
script is not yet present on main, so bootstrap may pass but authorizes zero
exemptions. Stage 2 is impossible until the real script exists on the evaluated
head, its exact blob is an audited validation authority, and its unconditional
package ownership is proven.

Any byte change to the qualification script invalidates that authority until the
routing contract is deliberately re-audited. Merely retaining the same PACKAGES
names is not sufficient because loop/command semantics could have changed.

The parser intentionally supports only the small workflow/script shapes used by
this repository. Unsupported forms fail closed rather than being guessed at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

FOCUSED_WORKFLOW = Path(".github/workflows/spore-boot-stack.yml")
GENERAL_WORKFLOW = Path(".github/workflows/ci.yml")
QUALIFICATION_SCRIPT = Path("scripts/check-spore-boot-stack.sh")

# Git blob SHA-1 of the focused qualification implementation audited on
# 2026-09-04 across the current boot integration lineage. Advancing this tuple is
# a governance action: inspect the new script semantics first, then update the
# routing contract in the same review.
AUDITED_QUALIFICATION_SCRIPT_BLOBS = (
    "8ad58e990c57a2f23b5b8f149355a7e48fd61dbd",
)

# Candidate Stage 2a exemptions. Each exact path maps to:
#   (focused workflow trigger owner, unconditional Cargo package owner)
# The broad focused trigger remains useful for discovery, but the ignore side is
# exact so a future crate cannot silently inherit a full-CI exemption.
VALIDATION_OWNERS = {
    "crates/domains/symthaea-boot-protocol/**": (
        "crates/domains/symthaea-boot-*/**",
        "symthaea-boot-protocol",
    ),
    "crates/domains/symthaea-boot-observer/**": (
        "crates/domains/symthaea-boot-*/**",
        "symthaea-boot-observer",
    ),
    "crates/domains/symthaea-quicken-fb/**": (
        "crates/domains/symthaea-quicken-fb/**",
        "symthaea-quicken-fb",
    ),
    "crates/domains/symthaea-boot-control/**": (
        "crates/domains/symthaea-boot-*/**",
        "symthaea-boot-control",
    ),
    "crates/domains/symthaea-boot-input/**": (
        "crates/domains/symthaea-boot-*/**",
        "symthaea-boot-input",
    ),
    "crates/domains/symthaea-boot-ecology-live/**": (
        "crates/domains/symthaea-boot-*/**",
        "symthaea-boot-ecology-live",
    ),
    "crates/domains/symthaea-boot-visual-clock/**": (
        "crates/domains/symthaea-boot-*/**",
        "symthaea-boot-visual-clock",
    ),
    "crates/domains/symthaea-boot-presentation/**": (
        "crates/domains/symthaea-boot-*/**",
        "symthaea-boot-presentation",
    ),
    "crates/core/symthaea-spore-continuity/**": (
        "crates/core/symthaea-spore-continuity/**",
        "symthaea-spore-continuity",
    ),
}

CANDIDATE_BOOT_ONLY = tuple(VALIDATION_OWNERS)

# Explicit diagnostics for surfaces intentionally retained under full CI until a
# later contract version installs and proves an unconditional validation owner.
DEFERRED_FULL_CI = (
    "crates/domains/symthaea-boot-*/**",  # catch-all may match future crates
    "crates/domains/symthaea-boot-render-projection/**",  # package owner conditional
    "crates/core/symthaea-boot-ecology/**",  # own package not unconditionally test/Clippy owned
    "nix/modules/quicken-fb.nix",  # Nix parse currently conditional on nix-instantiate
    "nix/modules/symthaea-boot-*.nix",
    "scripts/check-spore-boot-stack.sh",  # qualification trust root
    "scripts/measure-spore-boot.sh",  # no focused receipt-compatibility theorem yet
    "docs/architecture/BOOT_*.md",
    "docs/architecture/SPORE_*.md",
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    ".github/workflows/**",
    ".github/workflows/ci.yml",
    ".github/workflows/spore-boot-stack.yml",
    ".github/workflows/spore-ci-routing-contract.yml",
)

_LIST_ITEM = re.compile(r"^\s{6}-\s+['\"]([^'\"]+)['\"]\s*$")
_PACKAGE_ITEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def has_pull_request_trigger(text: str) -> bool:
    in_on = False
    for line in text.splitlines():
        if line == "on:":
            in_on = True
            continue
        if in_on and line and not line.startswith("  "):
            break
        if in_on and line == "  pull_request:":
            return True
    return False


def pull_request_list(text: str, key: str) -> tuple[str, ...] | None:
    lines = text.splitlines()
    in_pull_request = False
    in_key = False
    values: list[str] = []

    for line in lines:
        if line == "  pull_request:":
            in_pull_request = True
            in_key = False
            continue
        if in_pull_request and line and not line.startswith("    "):
            break
        if not in_pull_request:
            continue
        if line == f"    {key}:":
            in_key = True
            continue
        if in_key:
            match = _LIST_ITEM.match(line)
            if match:
                values.append(match.group(1))
                continue
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith("      "):
                break

    return tuple(values) if values else None


def unconditional_packages(text: str) -> tuple[str, ...]:
    """Parse only the literal top-level PACKAGES=(...) ownership block."""
    lines = text.splitlines()
    try:
        start = lines.index("PACKAGES=(") + 1
    except ValueError as error:
        raise ValueError("qualification script lacks literal PACKAGES=( block") from error

    packages: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped == ")":
            if not packages:
                raise ValueError("qualification PACKAGES block is empty")
            if len(set(packages)) != len(packages):
                raise ValueError("qualification PACKAGES block contains duplicates")
            return tuple(packages)
        if not stripped or stripped.startswith("#"):
            continue
        if not _PACKAGE_ITEM.fullmatch(stripped):
            raise ValueError(
                "unsupported non-literal entry in qualification PACKAGES block: " + stripped
            )
        packages.append(stripped)

    raise ValueError("qualification PACKAGES block is unterminated")


def fail(message: str) -> None:
    raise ValueError(message)


def validate_texts(
    focused_text: str,
    general_text: str,
    qualification_text: str | None,
    qualification_blob_sha: str | None,
    mode: str,
) -> dict[str, object]:
    if not has_pull_request_trigger(focused_text):
        fail("Spore Boot Stack must retain an explicit pull_request trigger")
    if not has_pull_request_trigger(general_text):
        fail("general CI must retain an explicit pull_request trigger")

    focused = pull_request_list(focused_text, "paths")
    if focused is None:
        fail("Spore Boot Stack must define pull_request.paths")
    if len(set(focused)) != len(focused):
        fail("Spore Boot Stack pull_request.paths contains duplicates")
    if pull_request_list(focused_text, "paths-ignore") is not None:
        fail("Spore Boot Stack may not combine paths with paths-ignore")

    general_paths = pull_request_list(general_text, "paths")
    ignored = pull_request_list(general_text, "paths-ignore")
    if general_paths is not None:
        fail("general CI PR routing must remain unscoped or use only validated paths-ignore")

    missing_trigger_owners = sorted(
        path
        for path, (trigger, _package) in VALIDATION_OWNERS.items()
        if trigger not in set(focused)
    )
    if missing_trigger_owners:
        fail(
            "candidate boot-only paths lost focused trigger ownership: "
            + ", ".join(missing_trigger_owners)
        )

    packages: tuple[str, ...] = ()
    ownership_ready = qualification_text is not None
    if qualification_text is not None:
        if qualification_blob_sha is None:
            fail("qualification script is present but its blob identity is absent")
        if qualification_blob_sha not in AUDITED_QUALIFICATION_SCRIPT_BLOBS:
            fail(
                "qualification script blob is not an audited routing authority: "
                + qualification_blob_sha
            )
        packages = unconditional_packages(qualification_text)
        missing_package_owners = sorted(
            path
            for path, (_trigger, package) in VALIDATION_OWNERS.items()
            if package not in set(packages)
        )
        if missing_package_owners:
            fail(
                "candidate boot-only paths lost unconditional package ownership: "
                + ", ".join(missing_package_owners)
            )
    elif qualification_blob_sha is not None:
        fail("qualification blob identity supplied without qualification script bytes")

    if mode == "bootstrap":
        if ignored is not None:
            fail("bootstrap mode requires general CI to remain unscoped")
    elif mode == "stage2":
        if not ownership_ready:
            fail("stage2 requires the real focused qualification script on the evaluated head")
        if ignored is None:
            fail("stage2 mode requires pull_request.paths-ignore")
        if len(set(ignored)) != len(ignored):
            fail("general CI pull_request.paths-ignore contains duplicates")
        unsafe = sorted(set(ignored) - set(CANDIDATE_BOOT_ONLY))
        if unsafe:
            deferred = sorted(set(unsafe) & set(DEFERRED_FULL_CI))
            unknown = sorted(set(unsafe) - set(DEFERRED_FULL_CI))
            details: list[str] = []
            if deferred:
                details.append("not-yet-validated/full-CI: " + ", ".join(deferred))
            if unknown:
                details.append("unapproved: " + ", ".join(unknown))
            fail("unsafe general-CI ignores: " + "; ".join(details))
    else:
        fail(f"unknown mode: {mode}")

    return {
        "schema": "spore-ci-routing-contract-v3",
        "status": "PASS",
        "mode": mode,
        "focused_path_count": len(focused),
        "general_ignored_path_count": len(ignored or ()),
        "qualification_script_present": ownership_ready,
        "qualification_script_blob_sha1": qualification_blob_sha,
        "audited_qualification_script_blobs": list(AUDITED_QUALIFICATION_SCRIPT_BLOBS),
        "unconditional_package_count": len(packages),
        "unconditional_packages": list(packages),
        "candidate_boot_only_count": len(CANDIDATE_BOOT_ONLY),
        "candidate_boot_only": list(CANDIDATE_BOOT_ONLY),
        "authorized_boot_only": list(CANDIDATE_BOOT_ONLY) if ownership_ready else [],
        "deferred_full_ci": list(DEFERRED_FULL_CI),
    }


def detect_mode(general_text: str) -> str:
    if not has_pull_request_trigger(general_text):
        fail("general CI must retain an explicit pull_request trigger")
    return "stage2" if pull_request_list(general_text, "paths-ignore") is not None else "bootstrap"


def validate(mode: str) -> dict[str, object]:
    focused_text = FOCUSED_WORKFLOW.read_text(encoding="utf-8")
    general_text = GENERAL_WORKFLOW.read_text(encoding="utf-8")
    if QUALIFICATION_SCRIPT.exists():
        qualification_bytes = QUALIFICATION_SCRIPT.read_bytes()
        qualification_text = qualification_bytes.decode("utf-8")
        qualification_blob_sha = git_blob_sha1(qualification_bytes)
    else:
        qualification_text = None
        qualification_blob_sha = None

    resolved_mode = detect_mode(general_text) if mode == "auto" else mode
    receipt = validate_texts(
        focused_text,
        general_text,
        qualification_text,
        qualification_blob_sha,
        resolved_mode,
    )
    receipt["focused_workflow_sha256"] = sha256(FOCUSED_WORKFLOW)
    receipt["general_workflow_sha256"] = sha256(GENERAL_WORKFLOW)
    receipt["qualification_script_sha256"] = (
        sha256(QUALIFICATION_SCRIPT) if qualification_text is not None else None
    )
    return receipt


def list_block(key: str, values: tuple[str, ...]) -> str:
    return f"    {key}:\n" + "".join(f"      - '{value}'\n" for value in values)


def focused_fixture() -> str:
    owners = tuple(dict.fromkeys(trigger for trigger, _package in VALIDATION_OWNERS.values()))
    return "on:\n  pull_request:\n" + list_block("paths", owners)


def qualification_fixture(packages: tuple[str, ...] | None = None) -> str:
    values = packages or tuple(package for _trigger, package in VALIDATION_OWNERS.values())
    return "#!/usr/bin/env bash\nPACKAGES=(\n" + "".join(f"  {p}\n" for p in values) + ")\n"


def self_test() -> None:
    focused = focused_fixture()
    qualification = qualification_fixture()
    audited_blob = AUDITED_QUALIFICATION_SCRIPT_BLOBS[0]
    bootstrap = "on:\n  push:\n    branches: [main]\n  pull_request:\n  workflow_dispatch:\n"
    one_safe = (CANDIDATE_BOOT_ONLY[2],)
    all_safe = CANDIDATE_BOOT_ONLY
    stage2_one = "on:\n  pull_request:\n" + list_block("paths-ignore", one_safe)
    stage2_all = "on:\n  pull_request:\n" + list_block("paths-ignore", all_safe)

    assert detect_mode(bootstrap) == "bootstrap"
    assert detect_mode(stage2_one) == "stage2"
    preboot = validate_texts(focused, bootstrap, None, None, "bootstrap")
    assert preboot["status"] == "PASS"
    assert preboot["qualification_script_present"] is False
    assert preboot["authorized_boot_only"] == []
    ready = validate_texts(focused, bootstrap, qualification, audited_blob, "bootstrap")
    assert ready["qualification_script_present"] is True
    assert len(ready["authorized_boot_only"]) == len(CANDIDATE_BOOT_ONLY)
    assert validate_texts(focused, stage2_one, qualification, audited_blob, "stage2")["status"] == "PASS"
    assert validate_texts(focused, stage2_all, qualification, audited_blob, "stage2")["status"] == "PASS"

    def expect_fail(
        name: str,
        focused_text: str,
        general_text: str,
        qualification_text: str | None,
        qualification_blob_sha: str | None,
        mode: str,
    ) -> None:
        try:
            validate_texts(
                focused_text,
                general_text,
                qualification_text,
                qualification_blob_sha,
                mode,
            )
        except ValueError:
            return
        raise AssertionError(f"negative self-test unexpectedly passed: {name}")

    expect_fail(
        "stage2 before qualification script lands",
        focused,
        stage2_one,
        None,
        None,
        "stage2",
    )
    expect_fail(
        "unaudited qualification script blob",
        focused,
        bootstrap,
        qualification,
        "0000000000000000000000000000000000000000",
        "bootstrap",
    )
    expect_fail(
        "qualification script drops safe package",
        focused,
        bootstrap,
        qualification_fixture(tuple(package for _trigger, package in list(VALIDATION_OWNERS.values())[1:])),
        audited_blob,
        "bootstrap",
    )
    expect_fail(
        "missing general pull_request",
        focused,
        "on:\n  push:\n    branches: [main]\n",
        None,
        None,
        "bootstrap",
    )
    expect_fail(
        "missing focused trigger owner",
        "on:\n  pull_request:\n" + list_block("paths", ("crates/domains/symthaea-quicken-fb/**",)),
        bootstrap,
        qualification,
        audited_blob,
        "bootstrap",
    )
    for name, unsafe in (
        ("broad future boot wildcard", "crates/domains/symthaea-boot-*/**"),
        ("core ecology", "crates/core/symthaea-boot-ecology/**"),
        ("conditional nix", "nix/modules/quicken-fb.nix"),
        ("qualification trust root", "scripts/check-spore-boot-stack.sh"),
        ("measurement script", "scripts/measure-spore-boot.sh"),
        ("cross-cutting lock", "Cargo.lock"),
        ("unknown path", "src/**"),
    ):
        expect_fail(
            name,
            focused,
            "on:\n  pull_request:\n" + list_block("paths-ignore", (unsafe,)),
            qualification,
            audited_blob,
            "stage2",
        )
    expect_fail("bootstrap narrowed early", focused, stage2_one, qualification, audited_blob, "bootstrap")
    expect_fail("stage2 missing ignore", focused, bootstrap, qualification, audited_blob, "stage2")
    expect_fail(
        "positive general paths allowlist",
        focused,
        "on:\n  pull_request:\n" + list_block("paths", ("src/**",)),
        qualification,
        audited_blob,
        "bootstrap",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("auto", "bootstrap", "stage2"), default="auto")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            self_test()
            print("spore-ci-routing-contract: self-test PASS")
        receipt = validate(args.mode)
    except (OSError, UnicodeError, ValueError, AssertionError) as error:
        print(f"spore-ci-routing-contract: FAIL: {error}", file=sys.stderr)
        return 1

    encoded = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    if args.receipt is not None:
        args.receipt.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
