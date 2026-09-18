# EKM-062 — Historical Replay Eligibility

Status: **draft / unqualified**

## Purpose

EKM-060 proves that a supplied EKM-057 mutation-evidence-seal sidecar is semantically equivalent to the EKM-056 capsule digest protected by EKM-059. EKM-061 separately proves, under an explicitly current-head provider contract and bounded validity window, that the exact EKM-059 checkpoint is current.

EKM-062 composes those two read-only facts into one immutable review receipt.

The resulting claim is deliberately narrow:

> the exact protected mutation-time evidence census is eligible for a later **isolated historical projection review**.

It is **not** authority to replay history, construct an EKM-056 capsule from untrusted bytes, hydrate writable state, mutate trusted state, activate a restart, or swap live epistemic state.

## Inputs

`HistoricalReplayEligibilityReceiptV1::evaluate` requires:

- the exact restart-v2 snapshot;
- the exact EKM-057 mutation-evidence-seal wire snapshot;
- the exact EKM-044 restart validation receipt;
- the exact verified EKM-059 mutation-seal checkpoint;
- the exact EKM-060 protected-sidecar admission receipt;
- the exact verified EKM-061 current-head receipt;
- a later review cycle.

## Re-verification

EKM-062 does not trust detached booleans.

It re-checks:

1. EKM-059 internal integrity;
2. EKM-060 against the exact restart/sidecar/receipt/checkpoint inputs;
3. EKM-061 internal integrity;
4. exact EKM-059 checkpoint digest agreement across EKM-060 and EKM-061;
5. exact restart capture-cycle agreement;
6. exact protected EKM-056 seal-capsule digest agreement;
7. exact EKM-059 sequence, verification-cycle and expiry binding carried by EKM-061.

## Temporal rule

EKM-060 binds the cycle at which protected-sidecar admission originally occurred. Re-verifying that receipt at a later EKM-062 review cycle would incorrectly alter the admission identity.

Therefore EKM-062:

- reproduces EKM-060 using `admission.observed_at_cycle()`;
- requires `reviewed_at_cycle >= admission.observed_at_cycle()`;
- requires `reviewed_at_cycle >= currentness.verified_at_cycle()`;
- rejects review at or after the EKM-061 currentness expiry.

The EKM-062 receipt then binds `reviewed_at_cycle` separately into its own digest.

## Claim decomposition

A successful receipt records:

- `protected_source_equivalence_verified = true`;
- `checkpoint_currentness_verified = true`;
- `isolated_historical_projection_review_eligible = true`.

It simultaneously records:

- `historical_replay_authorized = false`;
- `capsule_construction_authorized = false`;
- `writable_hydration_authorized = false`;
- `activation_authorized = false`.

This distinction is structural, not commentary.

## Digest binding

The domain-separated EKM-062 receipt digest binds:

- restart capture cycle;
- seal capture cycle;
- mutation count;
- exact EKM-059 checkpoint digest;
- exact protected EKM-056 seal-capsule digest;
- exact EKM-060 admission digest;
- exact EKM-061 currentness statement digest;
- exact EKM-061 proof digest;
- currentness attestation cycle;
- currentness expiry cycle;
- EKM-062 review cycle;
- all claim/authority booleans.

## Non-claims

EKM-062 does not claim:

- that a historical ledger projection has been reconstructed;
- that an EKM-026 firewall replay has executed;
- that mutation-time evidence chronology beyond the protected EKM-028 census has been inferred;
- that a writable support store or revision history has been hydrated;
- that a restart is safe to activate;
- that trusted checkpoints have been advanced;
- that generic signatures prove currentness;
- that legacy `TemporalFact::confidence` semantics are migrated;
- that legacy causal-language edges are admitted into the formal causal DAG.

## Qualification boundary

GitHub Actions remains the executable authority for format, compilation, Clippy, tests and runtime checks.

At the time this contract was frozen, parent EKM-061 exact-head CI run #7290 remained queued. Static review is not qualification evidence.

## Next safe tranche

After executable qualification, the next authority-preserving step is a **read-only historical projection**:

- construct only a temporary mutation-time ledger view from the protected EKM-028 seal census;
- compare that projection against the final append-only restart ledger;
- prove no post-seal evidence leaks into the historical view;
- do not execute support mutations yet;
- expose only a projection digest/report, not a mutable ledger handle.

Actual historical firewall replay should remain a separate later tranche.