# Symthaea–Muse V10 Pilot Operations and Independent Reproduction

## Purpose

V10 operationalizes the transition from a validated study platform to an ethically and methodologically controlled pilot. It does not change Sonata composition, temporal HDC/CfC influence, FEP policy selection, candidate generation, theory validation, or the confirmatory estimand.

V10 adds authority over five previously manual boundaries:

1. pilot objectives and permissible amendments;
2. pilot-specific randomization and participant assignment;
3. privacy-minimized recruitment and cohort status;
4. prospective study lifecycle transitions;
5. independent analysis and release reproduction.

A passing V10 pilot can justify freezing a confirmatory protocol. It cannot establish that Symthaea produces better music.

## Pilot and confirmatory separation

The pilot randomization secret must be independent of the confirmatory secret. The public pilot protocol contains only its SHA-256 commitment. Closing or revealing pilot assignments must not expose the future confirmatory order.

Pilot fixtures are the complete set of manifest fixtures labelled `Pilot`. Pilot data may estimate:

- recruitment and completion rates;
- attention-check performance;
- technical failure and exclusion rates;
- session duration and replay burden;
- pooled variance and clustering parameters;
- the sample size required for the frozen confirmatory estimand.

Pilot evidence must never enter the confirmatory outcome dataset. Arm-labelled monitoring is prohibited while pilot collection is open.

## Frozen pilot protocol

Validate the protocol against the V8.2 manifest and methodology:

```sh
cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  validate-pilot-protocol manifest.json methodology.json pilot-protocol.json
```

The protocol freezes:

- all pilot fixture identifiers;
- operational objectives;
- permitted adaptation categories;
- forbidden confirmatory claims;
- minimum completion and maximum enrollment;
- wave size;
- completion, attention, technical-failure, exclusion, and duration thresholds;
- the separate pilot randomization commitment;
- the external registration receipt.

Every amendment is hash-chained from the prior protocol digest, externally receipted, and must attest that the confirmatory manifest remained unchanged and confirmatory outcomes were not inspected.

## Pilot participant scheduling

Build participant-specific Williams schedules from the pilot secret:

```sh
cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  build-pilot-schedule \
  manifest.json methodology.json schedule.json codebook.json \
  pilot-protocol.json pilot-cohort.json pilot-secret.hex \
  pilot-participant-schedule.json pilot-schedule-audit.json
```

Every pilot fixture receives balanced presentation positions and first-order carryover relationships. Public schedules contain only anonymous presentation identifiers. Arm order remains in the private audit.

## Privacy-minimized cohort registry

The cohort registry deliberately excludes names, email addresses, telephone numbers, IP addresses, payment details, and free-form contact notes. It records only:

- pseudonymous participant tokens;
- one-way duplicate-enrollment commitments;
- recruitment source codes;
- eligibility and audio checks;
- consent and instruction document commitments;
- assigned block identifiers;
- participant status, exclusion, withdrawal, and external compensation state.

Contact and payment systems must remain separate from research evidence.

## Pilot runner and sealed collection

Pilot runner packages use the same prospective V9 state machine and hash-chained event log as confirmation, but they are bound to the pilot assignment book.

```sh
cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  build-pilot-runner-package \
  schedule.json pilot-participant-schedule.json artifact-bundle.json \
  BLOCK_ID runner-protocol.json runner-package.json
```

After all assigned blocks are terminal, seal the collection:

```sh
cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  seal-pilot-collection \
  pilot-protocol.json schedule.json pilot-participant-schedule.json \
  artifact-bundle.json pilot-cohort-registry.json \
  2026-07-14T00:00:00Z pilot-sessions.json pilot-collection.json
```

The collection remains blinded. Operational records are derived automatically from validated session logs and contain no arm, rank, recognition, or musical-quality fields.

## Outcome-neutral monitoring

Build a monitoring snapshot only from the sealed operational records:

```sh
cargo run -p symthaea-muse --features theory --bin cognitive_study -- \
  build-pilot-snapshot \
  pilot-protocol.json pilot-collection.json \
  2026-07-14T00:00:00Z pilot-snapshot.json
```

The frozen decision is one of:

- continue the current wave;
- open the next wave;
- pause for technical review;
- close the pilot;
- stop at maximum enrollment.

The monitor cannot inspect policy arms or endpoint effects.

## Pilot review and power recommendation

The final pilot report may contain pooled standard deviations, participant and family intraclass correlations, exclusion inflation, and a simulation-backed confirmatory sample-size recommendation. It may not contain a confirmatory quality claim.

The recommendation must bind:

- simulation source and environment;
- exact simulation input;
- target power and alpha;
- practical effect margin;
- at least 10,000 simulation replicates;
- at least eight confirmatory families;
- at least twelve participants per fixture in a multiple of four.

Any instrument or protocol change requires a newly frozen confirmatory manifest and external registration.

## Prospective lifecycle orchestration

The orchestration log permits only the following sequence:

1. draft;
2. pilot registered;
3. pilot artifacts sealed;
4. pilot collection open;
5. pilot collection closed;
6. pilot reviewed;
7. confirmatory protocol frozen;
8. confirmatory artifacts sealed;
9. confirmatory collection open;
10. confirmatory collection closed;
11. unblinded;
12. analyzed;
13. published.

Each transition is hash-chained, authorized, timestamped, and requires the relevant evidence authorities. Unblinding has no valid transition before confirmatory collection closes. Pilot amendments cannot be added after confirmation is frozen.

## Independent analysis agreement

The primary Rust result and an independently maintained external implementation are normalized into the same comparator schema. The cross-check refuses release unless both engines agree on:

- input and analysis-plan commitments;
- primary endpoint and alpha;
- comparator estimates;
- confidence bounds;
- practical margins;
- raw and adjusted p-values;
- inferential gates;
- the final success decision.

Numerical tolerances are explicit evidence, not hidden implementation constants.

## Independent reproduction attestation

An independent verifier records:

- identity and conflict-of-interest declaration;
- Nix, architecture, toolchain, and command commitments;
- expected and observed digests for source, artifacts, participant evidence, dataset, both analyses, and release root;
- exact or numerical match status;
- limitations and an externally published receipt.

The attestation is invalid if the verifier is not independent, the release root is not reproduced exactly, or the independent analysis cross-check does not pass.

## V10 operations release root

The final operations bundle commits the V9 release plus every V10 authority:

- pilot protocol;
- pilot schedule;
- cohort registry;
- pilot collection;
- final operational snapshot;
- amendment ledger;
- pilot report and sample-size recommendation;
- orchestration log;
- analysis cross-check;
- independent reproduction attestation;
- source archive, Nix lock, and toolchain evidence.

The standard-library verifier can independently validate the V10 commitments:

```sh
python3 scripts/verify_cognition_study_v10.py self-test
python3 scripts/verify_cognition_study_v10.py orchestration orchestration.json
python3 scripts/verify_cognition_study_v10.py amendments pilot-amendments.json
python3 scripts/verify_cognition_study_v10.py crosscheck analysis-crosscheck.json
python3 scripts/verify_cognition_study_v10.py operations-release operations-release.json
```

## Claim boundary

V10 can support claims that:

- the pilot was prospectively registered and operationally monitored without arm-labelled outcome peeking;
- participant assignment was balanced and independent of confirmatory randomization;
- pilot changes were externally receipted and did not alter confirmation silently;
- confirmatory sample size was informed by sealed pilot variance evidence;
- the study lifecycle, analyses, and release were independently reproducible.

V10 cannot support claims that Symthaea improves music, exceeds the heuristic, generalizes to all listeners, or validates consciousness-related interpretations. Those remain empirical confirmatory questions.
