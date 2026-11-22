# MEL Full Event-Universe Forecast Scorer v1

**Status:** standalone Rust scorer with a synthetic fixture. It does not connect to a live Symthaea forecast model, musician, audio engine or physical loopback.

## Purpose

The predictive-transparency v1 evaluator scores probabilities on event slots the candidate chose to forecast. This follow-on defines a candidate-independent opportunity universe first, then scores the candidate against every slot, including omitted predictions.

An omitted prediction is not silently dropped: it is interpreted as zero probability of event presence. This makes missed anticipation measurable and prevents an evaluator from looking well calibrated simply because it only scores events the model voluntarily reported.

## Frozen reference contract

The panel binds:
- opportunity-universe ID, freeze time and exact manifest digest;
- task family and canonical MUSE performance-time map ID/version;
- canonical map source clock and evaluator clock plus a source/target alignment receipt;
- alignment validity interval and uncertainty declaration;
- independently versioned label protocol and adjudication receipt;
- complete opportunity slots with target window, beat index, local beat period, role/instrument class and admissible alternatives;
- truth status: present, absent or ambiguous;
- candidate model/version/payload/input-snapshot/policy digests and forecast commit time;
- candidate probability for each predicted slot and conditional probabilities over the full admissible alternative set.

The opportunity universe must be frozen before the candidate forecast is committed. Reference labels must be adjudicated after all target windows end and after candidate commitment. The candidate cannot submit extra slot IDs outside the frozen universe.

Every probability slot has a mutually exclusive categorical outcome:
- no-event;
- one of the slot's admissible alternatives.

For each slot, the candidate supplies probability of event presence, then a conditional distribution over the alternatives. The joint categorical mass is:

    P(none) = 1 - P(present)
    P(alternative_i) = P(present) * P(alternative_i | present)

The conditional alternative probabilities must enumerate every admissible alternative and sum to one whenever event presence is non-zero. A zero-presence prediction carries no conditional alternatives.

## Scoring

For every independently adjudicated slot (including candidate omissions):

- **Presence Brier:** binary Brier score over present/absent outcomes.
- **Categorical Brier:** multiclass Brier over none plus all admissible alternatives.
- **Categorical log loss:** negative log probability assigned to the adjudicated class (with a documented numerical floor).
- **Top-1 accuracy:** most probable categorical class versus adjudicated outcome.
- **Presence ECE:** ten-bin descriptive expected calibration error over the full opportunity universe.
- **Presence precision/recall/F1:** using a preregistered presence threshold.
- **Omission accounting:** explicit number of all omitted slots and omitted positive opportunities.
- **False positives/negatives:** separately count predicted presence where absent and missed presence where present.
- **Per-role slices:** opportunity count, scored count, omission, false-positive, Brier and recall.

Ambiguous slots remain visible, are counted, and are excluded from proper-score denominators until adjudicated. Their presence does not cause a candidate to be marked wrong.

The categorical score is especially important: a candidate that predicts the correct presence but assigns high probability to the wrong harmony/melody alternative must be penalized. The full-universe event-presence score is different from this within-event alternative score; report both.

ECE bins are descriptive only. Human-study uncertainty must be estimated at participant/session/composition level, not by treating beat slots as independent samples.

## Negative controls

Tests cover:
- an omitted positive event, which remains in the denominator and incurs full-universe loss;
- a prediction for an unknown slot;
- duplicate predictions for one slot;
- label leakage (candidate commitment after adjudication);
- candidate commitment before the opportunity universe is frozen;
- inadmissible reference alternative;
- conditional probabilities that do not sum to one;
- ambiguous slots retained but not scored;
- zero-presence prediction without conditional alternatives;
- source/target clock-domain mismatch.

## Run

    cargo +1.96.0 run --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml \
      --bin mel-event-universe-audit -- score \
      tools/mel-judge-grounding-audit/fixtures/valid-event-universe-v1.json \
      report.json

    cargo +1.96.0 test --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --all-targets -- -D warnings

## Claim ceiling

The fixture's opportunity universe, labels, model predictions and receipts are synthetic. Receipt digests are shape-checked, not authenticated. The scorer validates candidate-independent manifest structure and timestamp order, but it does not independently recruit adjudicators, authenticate the label panel or prove the universe covers every possible musical action. The reported ECE has no cluster-aware confidence interval. Real calibration claims require an independently frozen, rights-cleared, musician/composition-held-out opportunity corpus and preregistered adjudication.

The event-universe scorer complements #13306; it does not replace the randomized preview/agency study in #13308.
