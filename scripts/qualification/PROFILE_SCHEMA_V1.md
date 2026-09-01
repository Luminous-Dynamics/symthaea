# Qualification profile schema v1

`SYMTHAEA_QUALIFICATION_PROFILE_V1` is a deliberately tiny, canonical, line-oriented format for selecting a reviewed qualification profile. It is not a general command language.

## Trust model

A profile name alone is not authority. For v1, `run_capsule.py` pins the SHA-256 of the complete raw bytes of every accepted profile. Changing any byte — including timeout, evidence-input path, step, line ending, whitespace, or comment — requires a reviewed runner/profile revision.

The runner maps step identifiers to fixed argv tuples compiled into the runner. Profile contents are never passed to a shell, `eval`, or an interpreter as executable text.

## Canonical encoding

Profiles are UTF-8 text with LF (`0x0a`) line endings and a final LF. CRLF is forbidden. Leading/trailing whitespace and tab characters are forbidden.

The first line is exactly:

    SYMTHAEA_QUALIFICATION_PROFILE_V1

Shipped v1 profiles then use this canonical order:

1. `profile=<stable-profile-name>`
2. `repository=<owner/repository>`
3. `rust_channel=<exact-rust-release>`
4. `timeout_seconds=<positive-decimal-integer>`
5. one or more `hash=<repository-relative-path>` lines
6. one or more ordered `step=<reviewed-step-id>` lines

Repository-relative hash paths must not be absolute, contain `.` or `..` path components, or contain backslashes. Duplicate hash paths are rejected.

The parser has no extensible unknown-key namespace: an unknown key is a hard failure.

## Raw-byte identity

The authoritative profile identity is:

    SHA256(raw_profile_bytes)

not a normalized parse tree. This intentionally prevents semantic-normalization ambiguities from creating multiple byte encodings for one named `v1` profile.

The current reviewed profile SHA-256 values are also present in `TOOLING_V1.sha256` and compiled into `run_capsule.py`.

## Step identity

Each accepted `step` is resolved through the runner's immutable `STEP_ARGV` table. The ordered step sequence for each profile is independently pinned by the runner's `CONTRACTS` table.

The capsule records a SHA-256 of the NUL-separated argv for every executed step. Display formatting is not command identity.

## Source boundary

Before qualification commands execute, the runner requires:

- an explicit 40-hex expected commit SHA;
- `git rev-parse HEAD` equal to that SHA;
- the expected canonical GitHub repository remote;
- a clean worktree including untracked files;
- every `hash=` evidence input to exist.

Branch names are informational and are not part of source authority.

## Environment boundary

The runner verifies the Rust release and records hashes of `rustc -vV`, `cargo -Vv`, the selected profile, the runner itself, and all `hash=` inputs. When invoked through `run_capsule_nix.sh`, the launcher first verifies `TOOLING_V1.sha256` and executes inside the target worktree's own `.#default` flake environment with `--no-write-lock-file`.

Common credential-bearing environment variables are stripped before Rust/Cargo subprocesses run. The capsule records only the count and a digest of removed variable names, never their values.

## Executor boundary

Executor class is explicit:

- `LOCAL_NIX`
- `GITHUB_HOSTED`
- `SELF_HOSTED_CI`

A successful local/Nix capsule is not equivalent to GitHub-hosted evidence. Independent executor agreement is a stronger evidence state.

## Capsule digest

`SHA256SUMS` contains SHA-256 hashes for evidence payload files in lexical path order. `CAPSULE.sha256` is:

    SHA256("SYMTHAEA_QUALIFICATION_CAPSULE_V1\\0" || SHA256SUMS_bytes)

This is a content digest, not a cryptographic signature and not proof of signer identity.

## Non-claims

A successful capsule establishes only that the declared checks executed successfully for the recorded source/profile/environment/executor tuple. It does not establish scientific truth, epidemiological validity, public-health authority, institutional authorization, or source authenticity beyond the recorded Git/source evidence.
