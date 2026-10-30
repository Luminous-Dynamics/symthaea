# MEL-MR-EVAL-016 — FluidSynth Process Observation

**Status:** implementation draft stacked on [MEL-MR-EVAL-015 PR #9064](https://github.com/Luminous-Dynamics/symthaea/pull/9064). No compiler, test, formatting or Clippy pass is claimed.

## Problem

MEL-MR-EVAL-015 records a local render receipt, but its first integration only hashed executable and SoundFont paths after the render and treated a returned WAV as a successful process. The FluidSynth runner discarded the child argument vector, exit details and diagnostics. That leaves the receipt unable to distinguish the declared render setup from a directly observed child-process invocation.

## Implemented increment

- `render_midi_to_wav_observed` returns a structured observation even when the child cannot start or exits unsuccessfully.
- The adapter canonicalizes the executable, SoundFont and MIDI paths; builds an explicit ordered argument vector; invokes the executable directly without a shell; and captures stdout/stderr.
- The exact parent environment snapshot is passed to the child after `env_clear`; a deterministic digest commits to the complete environment without persisting potentially sensitive values.
- Full stdout/stderr streams are SHA-256 committed, with bounded 4 KiB previews stored for diagnostics.
- Executable and SoundFont content hashes are taken before and after the child invocation. A receipt-producing render requires both pairs to match.
- A process-output WAV is mastered only after exit code zero and stable pre/post executable and SoundFont hashes.
- Renderer receipt v0.2 binds executable/SoundFont paths, pre/post hashes, ordered argv, environment digest, diagnostic digests/previews, truncation state and actual exit code.
- The Studio FluidSynth candidate path uses this observation rather than asserting exit code zero and hashing assets afterward.

## Fail-closed controls

- Missing binary or SoundFont: no FluidSynth invocation / no FluidSynth receipt.
- Spawn failure, signal termination or nonzero exit: no successful output artifact and no receipt.
- Executable or SoundFont hash changes during the invocation window: output is not accepted for receipt creation.
- Missing process observation, nonzero observed exit, malformed digests, unstable before/after hashes, or mismatch against the receipt's executable/SoundFont digests: verification rejects.
- Arguments are captured in order; a non-UTF-8 argument path is refused rather than lossy-encoded into evidence.
- The process environment is committed but raw values are not persisted.

## Trust boundary

This is stronger **local process evidence**, not cryptographic or remote attestation. The child may load a file through a race that changes and is restored between observations; hashes do not prove the kernel loaded those exact pages. A privileged or malicious host can falsify process observations. Captured process output is the direct child stdout/stderr, not an independent sensor of internal FluidSynth state. Environment hashing is a commitment to the snapshot supplied by this adapter, not a claim that every possible external machine state is captured.

The MIDI file is still checked for mutation across the process window. The next substantial boundary is native-renderer schedule capture plus renderer-specific measured acoustic timing and PCM-locality audit.

## Tests authored

- Environment digest is order-independent and content-sensitive.
- Diagnostic previews are bounded and truncation is explicit.
- FluidSynth receipts require an observed successful process and stable executable/SoundFont hashes.
- Native receipts remain process-observation-free and require a performed schedule.
- Existing receipt replay/tamper tests now cover process observation through the canonical receipt digest.

**Validation status:** Rust tests, `cargo check`, `cargo fmt --check`, Clippy and integration execution have not been run in this environment. These are authored test cases and source changes, not executed passes.
