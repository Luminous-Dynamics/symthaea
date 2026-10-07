#!/usr/bin/env python3
"""Fail closed if privileged mutation handlers regain a shell execution boundary.

This is intentionally a small source-structure ratchet. It does not attempt to
prove the entire Rust program. Instead it protects the architectural invariant
that consequential mutation handlers use typed process/filesystem capabilities
rather than the legacy shell-string executor.

The mutation-arm census is explicit so adding or renaming a consequential arm
cannot silently evade the check.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys


SOURCE = Path("crates/domains/symthaea-spore/src/bin/ssh_relay.rs")
MUTATIONS = (
    "install",
    "preserve_data",
    "rollback",
    "switch_generation",
    "service_action",
    "gc_collect",
    "write_config",
    "create_image",
    "restore_image",
    "connect_wifi",
)

ARM = re.compile(r'^            "([A-Za-z0-9_-]+)" => \{$')


def fail(message: str) -> None:
    print(f"privileged-shell-boundary: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if not SOURCE.is_file():
        fail(f"source file missing: {SOURCE}")

    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()

    match_index = next(
        (i for i, line in enumerate(lines) if "match client_msg.command.as_str()" in line),
        None,
    )
    if match_index is None:
        fail("main client command match disappeared")

    arm_indexes: dict[str, int] = {}
    for i in range(match_index + 1, len(lines)):
        match = ARM.match(lines[i])
        if match:
            arm_indexes.setdefault(match.group(1), i)

    missing = [name for name in MUTATIONS if name not in arm_indexes]
    if missing:
        fail(f"mutation arm census missing: {missing}")

    forbidden = (
        "run_cmd(",
        "run_cmd_with_stdin(",
        "privileged_shell_command(",
        'bash -c',
        'sh -c',
        '.arg("-c")',
        '.args(["-c"',
    )

    ordered = sorted(
        ((index, name) for name, index in arm_indexes.items()),
        key=lambda item: item[0],
    )
    for position, (start, name) in enumerate(ordered):
        if name not in MUTATIONS:
            continue
        end = ordered[position + 1][0] if position + 1 < len(ordered) else len(lines)
        body = "\n".join(lines[start:end])
        for needle in forbidden:
            if needle in body:
                fail(f'mutation arm {name!r} contains forbidden shell boundary {needle!r}')

    # Nested interpreters inside generated privileged scripts create a second
    # parsing authority underneath the already-controlled relay interpreter.
    # Keep this global because the relevant helpers live outside the mutation arms.
    nested_interpreters = (
        "chroot /mnt /bin/sh -c",
        "chroot /mnt /bin/bash -c",
        "chroot /mnt sh -c",
        "chroot /mnt bash -c",
    )
    for needle in nested_interpreters:
        if needle in text:
            fail(f"relay source contains forbidden nested interpreter {needle!r}")

    print(
        "privileged-shell-boundary: PASS: "
        f"checked {len(MUTATIONS)} consequential mutation arms; "
        "none invoke the legacy shell executor"
    )


if __name__ == "__main__":
    main()
