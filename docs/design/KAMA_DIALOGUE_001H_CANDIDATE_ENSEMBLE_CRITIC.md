# KAMA-DIALOGUE-001H — Candidate Ensemble and Independent Style Critic

Status: source candidate; not qualified.

## Purpose

Separate creative candidate generation from critique/selection so adult fantasy or romantic dialogue quality does not depend on one unconstrained completion.

This layer stores candidate metadata and opaque content references/commitments, not raw intimate content.

## Candidate binding

Every candidate binds:

- candidate ID;
- exact session ID and epoch;
- exact KAMA-DIALOGUE-001J style-proposal epoch;
- exact topic ID;
- generator identity;
- opaque content reference;
- opaque content commitment.

A candidate whose scope does not match the active style proposal is hard-disqualified.

## Hard gates

The V1 critic can report hard failures for:

- scope binding;
- hard boundary adherence;
- fantasy/reality namespace;
- identity/likeness policy;
- privacy;
- anti-coercion;
- stop/pacing requirements.

Any hard failure from any admitted independent critic is disqualifying.

```text
excellent creativity + hard failure = DISQUALIFIED
```

Hard failures are never averaged against creative quality.

## Soft quality dimensions

Only after hard-gate admission, critics score:

- style fit;
- pacing fit;
- continuity;
- persona consistency;
- naturalness;
- repetition avoidance;
- callback quality;
- novelty.

All soft metrics must be present and bounded in `[0, 1]`.

The default selection profile compares admissible candidates lexicographically in the order above. Novelty therefore cannot compensate for worse style or pacing fit under the default profile.

## Independent critics and disagreement

The default profile requires two distinct critics per candidate. Reusing one critic identity for the same candidate fails closed.

For each soft metric, the selector retains critic disagreement. If max-minus-min exceeds the configured threshold, the candidate is `Uncertain`, not automatically selected.

A critic may also mark material uncertainty explicitly.

Therefore:

```text
critic disagreement != hard failure
critic disagreement != automatic selection
```

If no candidate remains admissible, the receipt sets `revision_required=true` and selects nothing.

## Privacy

The selection receipt contains candidate IDs, critic-derived assessments, and the selected candidate ID. Candidate content remains behind opaque references/commitments.

This layer does not persist or expose raw intimate dialogue.

## Non-objectives

No engagement-duration, dependency, compliance, purchase, or session-length optimization metric exists in the V1 critic vocabulary.

## Tests

The integration suite covers:

- hard failure defeating perfect soft scores;
- critic disagreement retained as uncertainty;
- insufficient independent critics blocking auto-selection;
- exact style/session/topic scope binding;
- lexicographic quality selection;
- duplicate critic identity failing closed;
- material uncertainty without fabricated hard failure;
- incomplete soft-score reports rejected.

## Nonclaims

This tranche does not generate dialogue, prove critic correctness, establish universal taste, authorize physical behavior, establish consent, or prove product-quality superiority.
