# MEL-MR-EVAL-016 — FluidSynth Process Observation

**Status:** implementation draft stacked on [MEL-MR-EVAL-015 PR #9064](https://github.com/Luminous-Dynamics/symthaea/pull/9064). No compiler, test, formatting or Clippy pass is claimed.

## Problem

MEL-MR-EVAL-015 records a local render receipt, but its first integration only hashed executable and SoundFont paths after the render and treated a returned WAV as a successful process. The FluidSynth runner discarded the child argument vector, exit details and diagnostics. That leaves the receipt unable to distinguish the declared render setup from a directly observed child-process invocation.

## Implemented increment

- `render_midi_to_wav_observed` returns a structured observation even when the child cannot start or exits unsuccessfully.
- The adapter canonicalizes the executable, SoundFont and MIDI paths; builds an explicit ordered argument vector; invokes the executable directly without a shell; and captures stdout/stderr.
- The child receives a narrow runtime allowlist (`PATH`, `HOME`, locale, dynamic-library and temp/config paths) after `env_clear`; unrelated application credentials are not inherited. A deterministic digest commits to the exact allowlisted values without persisting them.
- Full stdout/stderr streams are drained concurrently and SHA-256 committed incrementally; only bounded 4 KiB previews are retained, so diagnostic memory does not grow with log size.
- Executable, SoundFont and exact MIDI input content hashes are taken before and after the child invocation. A receipt-producing render requires all three pairs to match the receipt's artifact identities.
- A process-output WAV is mastered only after exit code zero and stable pre/post executable and SoundFont hashes.
- Renderer receipt v0.2 binds executable/SoundFont/MIDI paths and pre/post hashes, ordered argv, allowlisted-environment digest, diagnostic digests/previews, truncation state, exit/spawn result and the mastered WAV digest.
- The Studio FluidSynth candidate path uses this observation rather than asserting exit code zero and hashing assets afterward. Every attempted process observation is retained on the candidate and exposed independently at `GET /api/piece/{id}/render-invocation`, including failed attempts and NativeMuse fallback. A successful `/render-receipt` remains a separate artifact and is created only when the output and all evidence checks pass.

## Fail-closed controls

- Missing binary or SoundFont: no FluidSynth invocation / no FluidSynth receipt.
- Spawn failure, signal termination or nonzero exit: no successful output artifact and no receipt.
- Executable, SoundFont or MIDI hash changes during the invocation window: output is not accepted for receipt creation.
- A successful exit with a missing/silent/unmasterable WAV is not a successful render receipt; the invocation record remains available with no mastered-output digest.
- Missing process observation, nonzero observed exit, malformed digests, unstable before/after hashes, or mismatch against the receipt's executable/SoundFont digests: verification rejects.
- Arguments are captured in order; a non-UTF-8 argument path is refused rather than lossy-encoded into evidence.
- Only the child runtime allowlist is passed and committed; unrelated credentials are excluded and raw values are not persisted.
- The process record binds the exact MIDI digest before and after the child run, and receipt validation requires it to match the receipt's MIDI bytes.

## Trust boundary

This is stronger **local process evidence**, not cryptographic or remote attestation. The child may load a file through a race that changes and is restored between observations; hashes do not prove the kernel loaded those exact pages. A privileged or malicious host can falsify process observations. Captured process output is the direct child stdout/stderr, not an independent sensor of internal FluidSynth state. Environment hashing is a commitment to the snapshot supplied by this adapter, not a claim that every possible external machine state is captured.

The MIDI file is checked for mutation across the process window. Failure observations are preserved even when Studio falls back to NativeMuse, but the fallback render does not inherit a FluidSynth receipt. The next substantial boundary is native-renderer schedule capture plus renderer-specific measured acoustic timing and PCM-locality audit.

## Tests authored

- Environment commitment is order-independent and content-sensitive.
- A synthetic environment containing unrelated API-key variables proves that the child allowlist excludes them.
- Fake child processes exercise successful output/argument/diagnostic capture, nonzero exit, spawn failure, missing output, MIDI mutation, SoundFont mutation and stderr draining beyond the bounded preview.
- Renderer receipts reject missing process observations, changed assets, substituted MIDI hashes, nonzero exit and output digest mismatch.
- Native receipts remain process-observation-free and require an explicit performed schedule.
- Studio retains failed FluidSynth observations separately from the NativeMuse fallback and exposes them through the invocation endpoint.

**Validation status:** Rust tests, `cargo check`, `cargo fmt --check`, Clippy and an actual FluidSynth integration render have not been run in this environment. These are authored tests and source changes, not executed passes. The dedicated exact-head workflow is present on the stacked parent PR; no pass is claimed.
