# Promotion Stack Effect Provenance v1

This tranche prevents the compact stacked effect set from becoming the only terminal evidence representation.

## Provenance-bearing effect

Each PromotionStackEffectEvidenceV1 retains:

    compact effect
    source delivery ID
    source payload digest
    source hook ID
    source event type
    source repository
    source authentication class

These source fields remain evidence lineage. They do not become an independent causal proof.

## Provenance-bearing set

PromotionStackEffectEvidenceSetV1 is ordered according to the reserved stack.

Construction requires:

- exact one-to-one PR correspondence with the reserved stack;
- unique delivery ID per effect;
- unique PR per effect;
- repository equal to the reserved repository;
- pull_request event type;
- webhook-hmac-verified source authentication;
- valid effect/head/merge-commit values;
- exact operation identity digest.

The input observation order may differ from stack order; the resulting evidence set is deterministically normalized to the reserved order.

## Compact projection

The provenance-bearing set can produce the existing PromotionStackEffectSetV1.

The compact projection is a view, not a replacement for the retained source evidence.

Terminal storage should therefore preserve both the provenance-bearing evidence set and the compact effect view when later reconciliation or audit requires tracing an effect back to an authenticated delivery.

## Fail-closed cases

Reject:

- missing hook, delivery, or payload provenance;
- duplicate delivery IDs across effects;
- duplicate PR numbers;
- repository mismatch;
- event mismatch;
- source-authentication mismatch;
- effect/head disagreement;
- incomplete provenance while the compact effect set appears complete.

## Causality boundary

Retained webhook provenance strengthens traceability only.

It does not establish that the local async operation caused the observed effect and does not establish provider-side topology CAS.

Those remain independent causal and provider-boundary theorems.

## Claim ceiling

This establishes only provenance-preserving effect projection semantics in the provider-free reference model.

It does not establish provider truthfulness, provider-side topology CAS, causal attribution, governance legitimacy, production atomicity, or successful external promotion.

Related: #7119, #7136, #7138, #7149, #7150, #7152, #7153, #7154, #7155.