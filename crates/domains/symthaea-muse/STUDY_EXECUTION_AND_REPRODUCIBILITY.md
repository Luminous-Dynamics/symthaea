# Symthaea–Muse V9 Study Execution and Reproducibility

## Purpose

V9 turns the frozen V8.2 methodology into an executable evidence path. It does not change Sonata generation, active inference, candidate ranking, or confirmatory statistics. It controls how already-frozen musical alternatives become study artifacts, how anonymous participants encounter them, how raw events become evidence, and how the complete release is committed.

## Authority order

The study must be executed in this order:

1. Freeze and externally register the V8.2 manifest, methodology, analysis plans, source revision, model checkpoint, renderer, soundfont, and environment.
2. Generate all four policy alternatives under the equal-budget contract.
3. Build the blinded schedule and private codebook from the resulting audio and recipe digests.
4. Create an artifact production plan that maps public presentation IDs to relative files.
5. Seal the artifact bundle from actual bytes on disk. WAV format, duration, peak, clipping, recipe digest, and schedule commitments are verified here.
6. Generate participant-specific schedules.
7. Build one runner package per assigned participant-fixture block.
8. Validate each package against the public schedule, participant assignment, and artifact bundle before deployment.
9. Run `cognitive_study_runner` locally or inside the pinned study environment. The server writes a forward hash-chained session log and rejects invalid transitions before persistence.
10. Compile all finalized session logs into one participant evidence envelope. Excluded and incomplete responses remain present under their preregistered status.
11. Run the frozen Rust and independent reference analyses.
12. Seal the final release root, including preregistration receipt, authorities, artifacts, evidence, analysis outputs, source archive, Nix lock, and toolchain evidence.

## Artifact sealing

The artifact plan contains only public presentation IDs. Policy arms must not appear in its filenames, logs, or bundle.

```sh
cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  seal-artifact-bundle \
  manifest.json methodology.json schedule.json \
  artifact-plan.json ./study-artifacts artifact-bundle.json

cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  validate-artifact-bundle \
  manifest.json methodology.json schedule.json \
  artifact-plan.json artifact-bundle.json ./study-artifacts artifact-issues.json
```

The frozen schedule remains authoritative for audio and recipe hashes. The fixture manifest remains authoritative for renderer and soundfont identities. An artifact does not enter the study merely because it exists or sounds correct.

## Participant packages

```sh
cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  build-runner-package \
  schedule.json participant-schedule.json artifact-bundle.json \
  BLOCK_ID runner-protocol.json runner-package.json

cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  validate-runner-package \
  schedule.json participant-schedule.json artifact-bundle.json \
  runner-package.json runner-package-issues.json
```

A package contains the assigned presentation order, anonymous codes, public audio paths and hashes, durations, and the exact frozen consent and instruction text with verified digests. It contains no policy labels or codebook entries.

## Participant runner

```sh
cargo run -p symthaea-muse --features studio --bin cognitive_study_runner -- \
  runner-package.json ./study-artifacts ./participant-evidence 127.0.0.1:8420
```

The runner verifies the package commitment and every audio hash before opening the server. It resumes only from a valid existing session log.

The state machine requires:

- consent text carried by the package, displayed by the runner, and bound to its frozen digest;
- a separate displayed instruction acknowledgement bound to the frozen instruction digest;
- assigned-order playback;
- sufficient listened duration;
- bounded replay counts;
- one response per presentation;
- response values in the frozen range;
- explicit handling of failed attention checks;
- a complete permutation ranking;
- finalization before inclusion.

Every accepted event commits its sequence, previous digest, server receipt time, client elapsed time, and payload. An invalid event is not appended.

## Cohort compilation

The collection draft references complete runner packages and logs. Compilation requires exactly one session for every participant assignment and one structural outcome for every scheduled presentation.

```sh
cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  seal-runner-collection \
  manifest.json schedule.json participant-schedule.json \
  artifact-bundle.json collection-draft.json participant-evidence.json
```

The result enters the existing V8.2 family-clustered and ranked-preference analyses. V9 does not alter the inferential thresholds.

## Independent verification

The standard-library Python verifier independently checks canonical commitments, artifact bytes, package commitments, and session hash chains without access to the policy codebook.

```sh
python3 scripts/verify_cognition_study_v9.py artifacts \
  ./study-artifacts artifact-bundle.json

python3 scripts/verify_cognition_study_v9.py session \
  runner-package.json BLOCK_ID.session.json
```

The Python verifier is an integrity cross-check, not a replacement for Rust protocol-state validation.

## Final release

The final release plan lists one file for every required authority and output. It also specifies when each file may become public. The codebook and randomization key, when included, must remain withheld until unblinding.

```sh
cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  seal-release release-plan.json ./release release-bundle.json

cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  validate-release release-plan.json release-bundle.json ./release release-issues.json
```

Publish the release bundle digest through the external registration record or another independently timestamped channel.

## Claim boundary

V9 can support the claim that artifacts and responses followed the frozen execution protocol and that the published evidence package is reproducible from committed files. It cannot by itself support claims that:

- temporal cognition improved music;
- Symthaea outperformed the heuristic;
- listeners preferred one policy;
- the study population generalizes to all listeners or composers.

Those claims require the preregistered analyses over collected evidence.

## V10 operational continuation

V10 adds the prospectively registered pilot, independent pilot randomization, privacy-minimized cohort registry, blinded operational monitoring, lifecycle orchestration, independent analysis agreement, and reproduction attestation described in `PILOT_OPERATIONS_AND_INDEPENDENT_REPRODUCTION.md`.
