# Engineering Trust — authority-bound clock policy V3

## Scope

This note records the protocol correction discovered while reviewing accepted-clock-basis admission.

The previous `ClockEvaluationPolicyV2` bound transition horizon, minimum eligible clock-authority keys, and whole-envelope algorithm diversity, but it did **not** bind the full quorum-verification semantics used later by the clock verifier.

That left these caller-selectable:

- minimum distinct sources;
- maximum observations;
- maximum uncertainty;
- maximum consensus width;
- candidate-window algorithm diversity.

The new governing theorem is:

```text
policy-bound permit
!= authority-bound quorum semantics

verifier success under caller-selected quorum criteria
!= authorized clock acceptance
```

## V3 model

The independent reference introduces two distinct content-addressed records.

### `ClockQuorumPolicyRevisionV1`

Commits the exact candidate-verification semantics:

```text
schema
minimum_distinct_sources
maximum_observations
maximum_uncertainty_ms
maximum_consensus_width_ms
require_algorithm_diversity
```

### `ClockEvaluationPolicyV3`

Commits:

```text
schema
policy_record_digest
exact ClockQuorumPolicyRevisionV1 ID
max_transition_ms
minimum_eligible_clock_authority_keys
require_eligible_algorithm_diversity
```

This separates:

```text
which keys are eligible across the whole transition envelope
!=
what quorum/uncertainty/consensus theorem the candidate clock must satisfy
```

The externally authenticated bootstrap claim continues to bind an exact `clock_evaluation_policy_id`; changing the quorum policy therefore necessarily changes the evaluation-policy ID and the authenticated bootstrap lineage.

## Frozen corrected chain

```text
ClockQuorumPolicyRevisionV1
4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304

ClockEvaluationPolicyV3
7a782b8adbf4e66a352c4a26ae614a458971d78c33c136cb8eeee6c6c28433b4

ClockBootstrapClaimV2
840d076db077b2df95277923b4be3b38f1e2a7027222298876041c036456d212

ClockBootstrapAuthorityEvidenceV2
0f3169321314c398a6ca68be67447b14ffd3d4409d228662d1effaa3d60f55a6

VerifiedClockBootstrapAuthorityV2
a0286a04ec3f4c0c6f8f2cc4bf97c7638bed43f913c09456e0931ea9c49e5f7c

ClockBootstrapAnchorV2
61fe43c253ca8dde9c152b641208f4eddc2ddbbfc263124d9b2a712e1e23d9e5

ClockEvaluationPermitV3
716c65376a721574bb53d2415e6b58f83d9c7483666b3799ca8fc4fe214f2cce
```

## Negative controls

The oracle proves that changing only `maximum_uncertainty_ms` from `5000` to `10000` changes the quorum-policy ID. The old evaluation policy then rejects that altered quorum policy.

If a new evaluation policy is intentionally created around the weaker quorum semantics, its ID changes; the old bootstrap claim rejects it. Only a newly authenticated bootstrap lineage can authorize that change.

The same identity divergence is checked for `maximum_consensus_width_ms`.

## Exact-byte reference evidence

The final source bytes were executed locally with Python 3.13.5:

```text
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         72f4fc7fa01976f148fc9d594743e6a58efbe1d3067400d052445be556ff91ca
locally computed Git blob  f70eb559312f015988adf8b7042c01e144c6f4fa
GitHub stored Git blob      f70eb559312f015988adf8b7042c01e144c6f4fa
```

The checked-in bytes therefore exactly match the executed reference subject. Repository CI remains a separate qualification layer.

## Production migration rule

Before the current #2408 accepted-clock candidate can mature:

1. add a typed/content-addressed quorum-policy revision;
2. supersede `ClockEvaluationPolicyV2` with V3 semantics that bind that quorum-policy ID;
3. re-key the bootstrap claim/authority lineage intentionally;
4. supersede `ClockEvaluationPermitV2` with V3, carrying both exact evaluation-policy and quorum-policy identities;
5. remove the free `ClockQuorumPolicy` argument from accepted-basis admission;
6. reconstruct the verifier policy from the permit-authorized quorum-policy revision;
7. re-freeze the accepted-basis lineage after those changes.

Because the affected trust/time PRs remain draft and Rust-unqualified, this should be treated as a protocol correction now rather than compatibility debt later.

## Deliberate nonclaims

This reference does not establish any hardware root, key, clock source, trust snapshot, provider implementation, policy-migration authority, trust-rotation authority, accepted clock basis, ETK currentness, requirement satisfaction, design qualification, deployment approval, or physical actuation authority.

Related: #1738, #2226, #2315, #2316, #2367, #2386, #2398, #2405, #2407, #2408.
