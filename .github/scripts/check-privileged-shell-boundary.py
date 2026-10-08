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

    if "run_cmd(" in text:
        fail("legacy run_cmd shell executor regressed into relay source")
    if "privileged_shell_command(" in text:
        fail("legacy privileged_shell_command shell executor regressed into relay source")
    lines = text.splitlines()

    # Source-integrity sentinels: catch accidental partial-blob overwrites before
    # the more specific architectural checks run.
    for marker in (
        "Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics",
        "async fn handle_connection(",
        "fn main(",
        "fn trusted_typed_executable(",
        "fn remove_transaction_artifact_dir_blocking(",
        "fn replace_configuration_atomically_blocking(",
        "async fn create_btrfs_image_archive(",
        "#[cfg(test)]\nmod tests {",
    ):
        if marker not in text:
            fail(f"relay source integrity sentinel disappeared: {marker!r}")

    test_module_start = text.find("#[cfg(test)]\nmod tests {")
    if test_module_start < 0:
        fail("test module boundary disappeared")
    production_text = text[:test_module_start]
    for line_number, line in enumerate(production_text.splitlines(), start=1):
        if "run_cmd(" in line and "async fn run_cmd" not in line:
            fail(
                f"production relay source contains run_cmd() at line {line_number}; "
                "generic shell execution is test-only"
            )
        if "run_cmd_with_stdin(" in line:
            fail(
                f"production relay source contains run_cmd_with_stdin() at line {line_number}; "
                "generic shell execution is test-only"
            )

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

    ordered = sorted(
        ((index, name) for name, index in arm_indexes.items()),
        key=lambda item: item[0],
    )
    for position, (start, name) in enumerate(ordered):
        end = ordered[position + 1][0] if position + 1 < len(ordered) else len(lines)
        body = "\n".join(lines[start:end])
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

    # Generation switching mutates the Nix system profile. Keep its
    # nix-env invocation in the isolated Nix capability so ambient nix.conf,
    # user configuration, and NIX_PATH cannot alter the mutation semantics.
    switch_start = arm_indexes.get("switch_generation")
    if switch_start is None:
        fail("switch_generation arm census missing")
    next_switch = next(
        (index for index, name in ordered if index > switch_start),
        len(lines),
    )
    switch_body = "\n".join(lines[switch_start:next_switch])
    if 'run_privileged_isolated_nix_args("nix-env"' not in switch_body:
        fail("switch_generation lost isolated Nix execution for nix-env mutation")

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
    if "stage_trusted_script(" not in preflight_body:
        fail("pre_install_check lost exact script staging")
    if "run_privileged_script_file(" not in preflight_body:
        fail("pre_install_check lost descriptor-bound script execution")
    if '"{disk}"' in preflight_body:
        fail("pre_install_check interpolates the browser disk into shell source")
    if '&[&disk]' not in preflight_body:
        fail("pre_install_check no longer passes disk as a script argument")
    if "staged_preflight.file" not in preflight_body:
        fail("pre_install_check does not execute the exact staged script descriptor")
    if "preflight script commitment" not in preflight_body:
        fail("pre_install_check does not record its exact script commitment")


    script_start = text.find("fn open_trusted_script(")
    if script_start < 0:
        fail("open_trusted_script helper disappeared")
    script_end = text.find("\nfn create_private_runtime_file(", script_start)
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

    descriptor_start = text.find("fn stage_trusted_script(")
    descriptor_end = text.find("\nfn open_trusted_script(", descriptor_start)
    if descriptor_start < 0 or descriptor_end < 0:
        fail("staged script descriptor capability disappeared")
    descriptor_body = text[descriptor_start:descriptor_end]
    if "sync_parent_directory" not in descriptor_body:
        fail("staged script capability no longer synchronizes its parent directory entry")
    for required in (
        "blake3::hash(contents)",
        "file.sync_all()",
        "file.read_to_end",
        "observed_hash.as_bytes() != expected.as_bytes()",
        "file.seek(SeekFrom::Start(0))",
    ):
        if required not in descriptor_body:
            fail(f"staged script capability lost required exact-byte guard {required!r}")

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

    for required in (
        "std::fs::canonicalize(&requested)",
        'resolved.starts_with("/nix/store/")',
        "let metadata = std::fs::metadata(&resolved)?;",
        "metadata.is_file()",
        "metadata.permissions().mode() & 0o111",
    ):
        if required not in trusted_body:
            fail(
                "trusted executable resolver lost immutable-store binding: "
                f"{required!r}"
            )

    shell_start = text.find("fn trusted_script_shell(")
    shell_end = text.find("\nfn trusted_nix_script_process(", shell_start)
    if shell_start < 0 or shell_end < 0:
        fail("trusted script shell capability disappeared")
    shell_body = text[shell_start:shell_end]
    for required in (
        'trusted_typed_executable("bash")',
        'trusted_typed_executable("sh")',
    ):
        if required not in shell_body:
            fail(f"trusted script shell capability lost immutable resolver call {required!r}")
    for forbidden_shell in (
        '"/bin/bash"',
        '"/bin/sh"',
        '"/run/current-system/sw/bin/bash"',
        '"/run/current-system/sw/bin/sh"',
    ):
        if forbidden_shell in shell_body:
            fail(
                "trusted script shell capability regained mutable pathname interpreter "
                f"{forbidden_shell!r}"
            )

    # There should be no privileged command-construction sites outside the
    # intentionally narrow capability adapters. This keeps helper functions from
    # bypassing the trusted executable resolver while preserving shell compatibility
    # only inside the dedicated shell/script adapters.
    process_calls = [line for line in lines if "privileged_process(" in line]
    if len(process_calls) != 5:
        fail(
            "unexpected privileged_process constructor count: "
            f"expected 5 capability-bound sites, found {len(process_calls)}"
        )
    allowed_constructor_fragments = (
        "fn privileged_process(",
        "privileged_process(shell)",
        "privileged_process(&shell)",
        "privileged_process(executable.as_ref())",
    )
    for line in process_calls:
        if not any(fragment in line for fragment in allowed_constructor_fragments):
            fail(f"unapproved privileged_process construction: {line.strip()!r}")

    env_start = text.find("fn privileged_process(")
    env_end = text.find("\n#[cfg(test)]\nfn privileged_shell_command(", env_start)
    if env_start < 0 or env_end < 0:
        fail("privileged_process helper boundary disappeared")
    env_body = text[env_start:env_end]
    if '"NIX_PATH"' in env_body:
        fail("generic privileged_process regained ambient NIX_PATH configuration input")

    nix_env_start = text.find("fn privileged_nix_process(")
    nix_env_end = text.find("\n#[cfg(test)]\nfn privileged_shell_command(", nix_env_start)
    if nix_env_start < 0 or nix_env_end < 0:
        fail("privileged_nix_process helper boundary disappeared")
    nix_env_body = text[nix_env_start:nix_env_end]
    if '"NIX_PATH"' in nix_env_body:
        fail("generic privileged_nix_process regained target-specific NIX_PATH authority")

    rebuild_start = text.find("fn privileged_nixos_rebuild_process(")
    rebuild_end = text.find("\nasync fn run_privileged_nixos_rebuild_args(", rebuild_start)
    if rebuild_start < 0 or rebuild_end < 0:
        fail("privileged_nixos_rebuild_process helper disappeared")
    rebuild_body = text[rebuild_start:rebuild_end]
    if '"NIX_PATH"' not in rebuild_body:
        fail("nixos-rebuild capability lost its explicit configuration NIX_PATH input")
    if '"nixos-rebuild"' not in rebuild_body:
        fail("nixos-rebuild capability lost fixed executable identity")

    nix_script_start = text.find("fn trusted_nix_script_process(")
    nix_script_end = text.find("\nfn trusted_script_process(", nix_script_start)
    if nix_script_start < 0 or nix_script_end < 0:
        fail("trusted_nix_script_process helper boundary disappeared")
    nix_script_body = text[nix_script_start:nix_script_end]
    if '"NIX_PATH"' not in nix_script_body:
        fail("installer script capability lost its explicit NIX_PATH environment")
    if "trusted_script_shell()?" not in nix_script_body:
        fail(
            "installer script capability no longer resolves its interpreter through "
            "the typed immutable executable resolver"
        )

    process_start = text.find("fn preservation_process_identity(")
    process_end = text.find("\nfn preservation_user_identity", process_start)
    if process_start < 0 or process_end < 0:
        fail("preservation process identity helper disappeared")
    process_body = text[process_start:process_end]
    for required in (
        "parse_proc_effective_uid",
        "status",
        "read_link",
        "exe",
        "uid()",
        "permissions().mode()",
    ):
        if required not in process_body:
            fail(f"preservation process identity lost required discriminator {required!r}")
    if "preservation_process_exists(" in text:
        fail("name-only preservation_process_exists helper regressed into relay source")

    sensitive_start = text.find("fn cleanup_sensitive_file(path: &str) -> Result<(), String> {")
    sensitive_end = text.find("\nfn cleanup_sensitive_files(", sensitive_start)
    if sensitive_start < 0 or sensitive_end < 0:
        fail("sensitive cleanup helper disappeared")
    sensitive_body = text[sensitive_start:sensitive_end]
    for required in (
        "unlink_verified_sensitive_file(",
        "O_NOFOLLOW",
        "fstatat(",
        "AT_SYMLINK_NOFOLLOW",
        "st_dev",
        "st_ino",
        "unlinkat(",
        "sync_all()",
    ):
        if required not in sensitive_body:
            fail(f"sensitive cleanup lost required descriptor/inode primitive {required!r}")
    if "remove_file(path)" in sensitive_body:
        fail("sensitive cleanup regressed to pathname-based unlink")

    # Transaction cleanup is itself a privileged authority boundary. It must stay
    # descriptor-relative and never fall back to pathname-based recursive deletion.
    cleanup_start = text.find("fn remove_transaction_artifact_dir(path: &str)")
    cleanup_end = text.find("\n/// Generate Secure Boot setup commands", cleanup_start)
    if cleanup_start < 0 or cleanup_end < 0:
        fail("transaction cleanup helper disappeared")
    cleanup_body = text[cleanup_start:cleanup_end]
    for required in (
        "remove_transaction_artifact_dir_blocking(",
        "O_NOFOLLOW",
        "openat(",
        "unlinkat(",
    ):
        if required not in cleanup_body:
            fail(f"transaction cleanup lost required descriptor primitive {required!r}")
    if "remove_dir_all(" in cleanup_body:
        fail("transaction cleanup regressed to pathname-based recursive deletion")

    # Recursive rm is especially dangerous inside generated privileged install plans.
    # The test module may mention it in adversarial fixtures; production source may not.
    test_module = text.find("#[cfg(test)]\nmod tests {")
    if test_module < 0:
        fail("test module boundary disappeared")
    production_text = text[:test_module]
    if "rm -rf" in production_text:
        fail("production relay source contains recursive rm -rf")

    static_script_start = text.find("async fn run_privileged_script_source(")
    static_script_end = text.find("\n}\n", static_script_start)
    if static_script_start < 0 or static_script_end < 0:
        fail("run_privileged_script_source helper disappeared")
    static_script_body = text[static_script_start:static_script_end]
    if "script: &'static str" not in static_script_body:
        fail("privileged inline script capability must accept only &'static str")
    if "stdin(std::process::Stdio::piped())" not in static_script_body:
        fail("privileged inline script capability lost explicit stdin binding")

    # Config reads are read-only, but the pathname still crosses a filesystem
    # authority boundary. Require the same descriptor-relative O_NOFOLLOW reader
    # used by restore/postcondition verification.
    read_cfg_start = text.find('            "read_config" => {')
    read_cfg_end = text.find('\n            "', read_cfg_start + 20)
    if read_cfg_start < 0 or read_cfg_end < 0:
        fail("read_config arm disappeared")
    read_cfg_body = text[read_cfg_start:read_cfg_end]
    if 'tokio::fs::read_to_string("/etc/nixos/configuration.nix")' in read_cfg_body:
        fail("read_config regressed to pathname-following read_to_string")
    if 'read_regular_file_bytes_at(' not in read_cfg_body:
        fail("read_config lost descriptor-relative regular-file reader")

    # Generated system_config.nix content must remain an out-of-band staged file.
    patch_start = text.find("fn system_config_patch(")
    patch_end = text.find("\n/// Generate the automated install script", patch_start)
    if patch_start < 0 or patch_end < 0:
        fail("system_config_patch helper disappeared")
    patch_body = text[patch_start:patch_end]
    for forbidden in ("SYSPATCH", "cat >", "NIXCONF"):
        if forbidden in patch_body:
            fail(f"system_config_patch regained inline shell-source generation: {forbidden!r}")

    # Install has one additional monotonic authority requirement: the exact
    # staged script digest must be journal-bound before the background worker starts.
    install_start = arm_indexes["install"]
    install_position = next(
        (position for position, (start, name) in enumerate(ordered) if name == "install"),
        None,
    )
    install_end = (
        ordered[install_position + 1][0]
        if install_position is not None and install_position + 1 < len(ordered)
        else len(lines)
    )
    install_body = "\n".join(lines[install_start:install_end])
    bind_at = install_body.find("bind_execution_commitment(&transaction")
    spawn_at = install_body.find("spawn_privileged_background_script_file")
    digest_at = install_body.find('role: "install-script"')
    if bind_at < 0 or spawn_at < 0 or bind_at > spawn_at or digest_at < 0:
        fail(
            "install mutation must durably bind the install-script commitment "
            "before spawning its privileged worker"
        )

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
