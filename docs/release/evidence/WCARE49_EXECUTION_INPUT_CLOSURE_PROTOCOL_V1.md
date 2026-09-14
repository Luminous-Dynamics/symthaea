# WCARE-49 — execution-input closure protocol v1

Status: `PREREGISTERED_BEFORE_LOCK_ADMISSION_RESULT_WITH_MONOTONIC_HARDENING`
Authority: `MeasurementOnly`
Tracks: #2948

## Purpose

Close the repository/toolchain execution-input gap left intentionally open by WCARE-47 before the admitted WCARE-42 lock is used as stronger executable evidence.

The base v1 subject, entry theorem, exact WCARE-47Q run identity, and fail-closed requirement were preregistered in #2948 while exact WCARE-47Q run `34871019291` was still queued. After that run completed, source audit found that Cargo hierarchical configuration can be discovered above the Git checkout. The implementation was therefore monotonically hardened before any WCARE-49 execution result was accepted.

That amendment only adds rejection conditions and additional census evidence. It does not change the exact admission event, admit a new candidate, weaken a gate, or promote any claim.

## Exact subject

- WCARE-48 FINAL: `5bc23735f545b1b82044820f0e02ece59be04b4a`
- PREPARED parent: `52d1d9fb741250ab8bcab205113689a8cc9431bb`
- lock blob: `1a9b126e3d5062bd3be5290bfa5930d189760ceb`
- lock SHA-256: `7288b7dd64b533a4e08ae9a66ff120fb69d5885effd80b0e3b7f51455028d976`
- package census: `45`
- WCARE-48V head: `890ab746618f0a57853db1e38cedb7c1b500a89d`
- WCARE-48V result SHA-256 used by WCARE-48H: `19c4eb912fee92e67a8e0071da44582e21aa99f9a0695441538dbe5bf4f78b1c`
- WCARE-48H head: `2a3f2ca7f91911351b817d96565753f0dee06f54`
- WCARE-48H successful run: `34856005699`
- WCARE-47Q head: `8eb8af15af464ae6c49d20de225aa501ec4bedaf`
- WCARE-47Q run id / number / attempt: `34871019291 / 4 / 1`

A retry, replacement run, changed upstream head, or different lock is outside v1.

## Entry state machine

The WCARE-49 harness may classify the exact WCARE-47Q event as:

- `ADMISSION_PRECONDITION_PENDING` — exact run exists but is not completed. No closure execution is permitted.
- `ADMISSION_PRECONDITION_FAILED` — exact run failed, its hosted identity drifted, required job/steps failed, or its emitted WCARE-47 result is not the exact preregistered `LOCK_ADMITTED` theorem. No closure execution is permitted.
- `ADMISSION_PRECONDITION_SATISFIED` — exact hosted run succeeded and its result is cryptographically/result-wise consistent with the preregistered FINAL lock.

Only the third state permits the execution-input theorem to run. A green GitHub run badge is not itself admission evidence; the harness must bind the frozen WCARE-47 JSON result and its `WCARE47Q_RESULT_SHA256` from the exact hosted job log.

## Cargo hierarchical configuration closure

Qualification invokes Cargo from `tools/wcare42_builder_attestation_verifier`. Cargo configuration discovery is therefore treated as a hierarchy, not as a Git-root-local property.

The production census starts at the verifier working directory and walks every ancestor through the filesystem root. At every level it checks both:

- `.cargo/config.toml`
- `.cargo/config`

The expected census of discovered configuration files is empty.

Repository-local candidates may be recorded by repository-relative path. Candidates above the checkout are recorded only as:

- `scope = ambient_parent`;
- ancestor depth above the checkout;
- config name;
- SHA-256 of the normalized absolute path locator.

Raw ambient absolute paths are not emitted in receipts or failure details.

Any existing tracked, ordinary-untracked, ignored-untracked, or ambient-parent Cargo configuration candidate fails closure.

## Checkout cleanliness

The checkout must remain fully clean before and after execution. WCARE-49 combines:

- `git status --porcelain=v1 --untracked-files=all`;
- tracked and cached diff checks;
- ordinary untracked-file enumeration;
- ignored-untracked-file enumeration.

This intentionally closes the gap where an ignored input can be invisible to ordinary status output.

## Environment and Cargo-home closure

The harness rejects non-empty semantic compiler/build/source overrides including:

- `RUSTC`, `RUSTDOC`, `RUSTC_WRAPPER`, `RUSTC_WORKSPACE_WRAPPER`, `RUSTC_BOOTSTRAP`;
- `RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS`, `RUSTDOCFLAGS`;
- `CARGO_BUILD_RUSTC`, `CARGO_BUILD_RUSTC_WRAPPER`, `CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER`, `CARGO_BUILD_RUSTFLAGS`, `CARGO_BUILD_TARGET`;
- any `CARGO_TARGET_*_(RUSTFLAGS|LINKER|RUNNER)` override;
- any `CARGO_PROFILE_*` override;
- any `CARGO_SOURCE_*` override;
- any `CARGO_REGISTRIES_*` override.

Qualification uses a fresh, initially empty `CARGO_HOME` outside the repository and requires both `$CARGO_HOME/config.toml` and `$CARGO_HOME/config` to be absent before and after execution. Cargo may populate cache/registry state during execution; the claim is config closure, not immutable Cargo-home contents.

Build output uses a fresh, initially empty `CARGO_TARGET_DIR` outside the repository.

## Exact execution theorem

After exact admission preconditions pass, the harness must:

1. prove exact FINAL HEAD/parent and immutable WCARE-42 source/toolchain/lock blobs;
2. prove a fully clean checkout, including ordinary and ignored untracked files;
3. derive an empty Cargo hierarchical-config census from verifier cwd through filesystem root;
4. prove the fresh external Cargo home contains no Cargo config before execution;
5. reject the bounded semantic environment override set;
6. use fresh external Cargo home and target directories;
7. require exact Rust 1.96.0 and capture `rustc -Vv` / `cargo -Vv`;
8. capture lock Git blob and SHA-256 before execution;
9. execute `cargo metadata --locked --format-version 1`;
10. execute `cargo test --locked`;
11. execute `cargo test --locked --test golden` explicitly;
12. re-derive source/toolchain/lock identities after execution;
13. re-derive the hierarchical Cargo-config and environment censuses;
14. prove Cargo-home config remains absent;
15. require the full checkout to remain clean, including ignored untracked files;
16. emit a canonical result with the exact admission-result hash and bounded input census.

A pass classification is `EXECUTION_INPUTS_CLOSED`.

## Adversarial requirements

Helper-level self-tests must fail closed for at least:

- wrong WCARE-47Q run identity;
- queued admission precondition;
- duplicate identical hosted result rows without treating them as contradictory evidence;
- repository-local Cargo config;
- ambient-parent Cargo config;
- ordinary untracked repository input;
- ignored untracked repository input;
- `RUSTFLAGS` and `RUSTC_BOOTSTRAP` injection;
- target-specific rustflag/linker injection;
- profile/source override injection;
- Cargo-home config injection;
- external Cargo directories placed inside the repository;
- admission JSON drift.

## Explicit boundedness and non-claims

`EXECUTION_INPUTS_CLOSED` means the exact admitted lock executed under this bounded repository/Cargo/toolchain input envelope. It does **not** mean fully hermetic execution. In particular, v1 does not close arbitrary build-script environment consumption, kernel/OS state, network transport, DNS, runner firmware, hardware, or all possible process-level ambient state.

Even `EXECUTION_INPUTS_CLOSED` keeps all of these false:

- `full_machine_hermeticity_established`
- `trusted_hardware_established`
- `builder_authentication_established`
- `independent_host_reproducibility_established`
- `wcare42_executable_qualification_established`
- `runtime_authority_granted`

It also establishes no consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto/self-preservation authority, or solved alignment.

The governing chain remains:

`HOST_RUN_ATTESTED != LOCK_ADMITTED != EXECUTION_INPUTS_CLOSED != WCARE-42 executable-qualified`.
