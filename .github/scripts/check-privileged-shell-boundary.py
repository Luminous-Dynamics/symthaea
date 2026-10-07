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
        "run_privileged_script(",
        "run_privileged_script_with_args(",
        "privileged_process(",
        "privileged_shell_command(",
        "privileged_script_command(",
        'bash -c',
        'sh -c',
        '.arg("-c")',
        '.args(["-c"',
    )

    # Shell-backed execution is tolerated only in this explicit read-only
    # diagnostic census. Every other websocket command arm must be shell-free.
    # This turns the old "mutation allowlist" into a closed-world policy:
    # a newly added consequential arm cannot silently inherit run_cmd().
    SHELL_ALLOWED_READ_ONLY = {
        "probe_hardware",
        "scan_apps",
        "deep_scan",
        "gc_analyze",
        "diagnose",
        "netboot_info",
        "list_images",
        "inventory",
    }

    ordered = sorted(
        ((index, name) for name, index in arm_indexes.items()),
        key=lambda item: item[0],
    )
    for position, (start, name) in enumerate(ordered):
        end = ordered[position + 1][0] if position + 1 < len(ordered) else len(lines)
        body = "\n".join(lines[start:end])
        has_shell_executor = any(
            needle in body for needle in ("run_cmd(", "run_cmd_with_stdin(")
        )

        if has_shell_executor and name not in SHELL_ALLOWED_READ_ONLY:
            fail(
                f'command arm {name!r} contains shell-backed execution; '
                "only explicitly enumerated read-only diagnostic arms may use it"
            )

        if name in MUTATIONS:
            for required in (
                "mutation_lock.try_lock()",
                "MutationLease::acquire()",
                "admit_mutation_transaction(",
            ):
                if required not in body:
                    fail(
                        f'mutation arm {name!r} lost required mutation authority '
                        f'boundary {required!r}'
                    )
            for needle in forbidden:
                if needle in body:
                    fail(f'mutation arm {name!r} contains forbidden shell boundary {needle!r}')

        if name in SHELL_ALLOWED_READ_ONLY and has_shell_executor:
            if "client_msg." in body:
                fail(
                    f'read-only diagnostic arm {name!r} interpolates command-message '
                    "state into its shell authority surface"
                )

    # The pre-install check is non-destructive, but it accepts a browser-selected
    # disk. Keep that value out of generated shell source: the script must receive
    # it only through argv.
    preflight_start = arm_indexes.get("pre_install_check")
    if preflight_start is None:
        fail("pre_install_check arm census missing")
    next_preflight = next(
        (index for index, name in ordered if index > preflight_start),
        len(lines),
    )
    preflight_body = "\n".join(lines[preflight_start:next_preflight])
    if "run_cmd(" in preflight_body:
        fail("pre_install_check still uses the shell-string executor")
    if "run_privileged_script_with_args(" not in preflight_body:
        fail("pre_install_check lost typed script argv execution")
    if '"{disk}"' in preflight_body:
        fail("pre_install_check interpolates the browser disk into shell source")
    if '&[&disk]' not in preflight_body:
        fail("pre_install_check no longer passes disk as a script argument")

    if "run_privileged_script_with_args(" not in preflight_body:
        fail("pre_install_check lost typed script execution")
    if "privileged_script_command(" in preflight_body:
        fail("pre_install_check arm bypasses the descriptor-bound script runner")

    script_start = text.find("fn open_trusted_script(")
    if script_start < 0:
        fail("open_trusted_script helper disappeared")
    script_end = text.find("\nasync fn run_privileged_script_with_args(", script_start)
    if script_end < 0:
        fail("trusted script helper boundary could not be located")
    script_body = text[script_start:script_end]
    for required in (
        "O_NOFOLLOW",
        "O_CLOEXEC",
        "metadata.uid()",
        "metadata.permissions().mode()",
        "metadata.len() > 256 * 1024",
    ):
        if required not in script_body:
            fail(f"trusted script opener lost required guard {required!r}")

    runner_start = text.find("async fn run_privileged_script_with_args(")
    runner_end = text.find("\nfn create_private_runtime_file(", runner_start)
    if runner_start < 0 or runner_end < 0:
        fail("descriptor-bound script runner could not be located")
    runner_body = text[runner_start:runner_end]
    for required in (
        "open_trusted_script(path)?",
        '.arg("-s")',
        '.arg("--")',
        '.stdin(std::process::Stdio::from(script))',
    ):
        if required not in runner_body:
            fail(f"descriptor-bound script runner lost required primitive {required!r}")

    # Secret credentials must never regain shell-owned cleanup. Native cleanup
    # happens before terminal transaction journaling instead.
    forbidden_secret_cleanup = (
        'rm -f {pw_file}',
        'rm -f "$LUKS_KEYFILE"',
        'format!("trap',
        'map(|path| format!("rm -f -- {}", path))',
    )
    for needle in forbidden_secret_cleanup:
        if needle in text:
            fail(f"secret material cleanup regressed to shell construction: {needle!r}")

    # Typed execution is closed-world: bare program names must first pass
    # through the trusted executable resolver, and the resolver must pin them
    # to the current NixOS system closure instead of PATH search.
    typed_start = text.find("async fn run_privileged_args(")
    if typed_start < 0:
        fail("run_privileged_args helper disappeared")
    typed_end = text.find("\nfn privileged_script_command", typed_start)
    if typed_end < 0:
        fail("typed executor boundary could not be located")
    typed_body = text[typed_start:typed_end]
    if "trusted_typed_executable(program)?" not in typed_body:
        fail("run_privileged_args lost trusted executable resolution")
    stdin_start = text.find("async fn run_privileged_args_with_stdin(")
    if stdin_start < 0:
        fail("run_privileged_args_with_stdin helper disappeared")
    stdin_end = text.find("\n/// nixos-anywhere orchestration stages.", stdin_start)
    if stdin_end < 0:
        fail("stdin typed executor boundary could not be located")
    stdin_body = text[stdin_start:stdin_end]
    if "trusted_typed_executable(program)?" not in stdin_body:
        fail("run_privileged_args_with_stdin lost trusted executable resolution")

    trusted_start = text.find("fn trusted_typed_executable(")
    if trusted_start < 0:
        fail("trusted_typed_executable helper disappeared")
    trusted_end = text.find("\nasync fn run_privileged_args(", trusted_start)
    if trusted_end < 0:
        fail("trusted executable resolver boundary could not be located")
    trusted_body = text[trusted_start:trusted_end]
    if "/run/current-system/sw/bin/" not in trusted_body:
        fail("typed executable resolver no longer pins to /run/current-system/sw/bin/")
    if 'strip_prefix("/nix/var/nix/profiles/system/bin/")' not in trusted_body:
        fail("typed executable resolver lost the validated system-profile exception")

    # There should be no privileged command-construction sites outside the
    # intentionally narrow capability adapters. This keeps helper functions from
    # bypassing the trusted executable resolver while preserving shell compatibility
    # only inside the dedicated shell/script adapters.
    process_calls = [line for line in lines if "privileged_process(" in line]
    if len(process_calls) != 4:
        fail(
            "unexpected privileged_process constructor count: "
            f"expected 4 capability-bound sites, found {len(process_calls)}"
        )
    allowed_constructor_fragments = (
        "fn privileged_process(",
        "privileged_process(shell)",
        "privileged_process(executable.as_ref())",
    )
    for line in process_calls:
        if not any(fragment in line for fragment in allowed_constructor_fragments):
            fail(f"unapproved privileged_process construction: {line.strip()!r}")

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
