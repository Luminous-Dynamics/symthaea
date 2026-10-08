# Promotion Pull-Request Merge Effect Observation v1

This tranche converts an authenticated GitHub pull-request webhook payload into a normalized merge-effect observation.

## Admission order

The reference adapter requires:

    authenticated webhook receipt
        ->
    exact payload bytes
        ->
    strict semantic parsing
        ->
    merged pull-request effect observation

Authentication happens first. A payload is never treated as semantic evidence before its delivery signature and context are valid.

## Required semantics

The normalized observation records:

    delivery ID
    repository
    pull-request number
    event type
    action
    merged state
    pull-request head SHA
    observed merge commit SHA
    payload digest

For a merged effect, the payload must represent:

    event_type = pull_request
    action = closed
    merged = true
    repository = expected repository
    PR number = expected PR
    head SHA = expected reserved head
    non-empty merge commit SHA

The adapter also requires the payload's nested pull-request number and top-level number to agree.

## Effect conversion

A valid observation can produce:

    PromotionStackEffectV1

for the corresponding PR.

This allows authenticated webhook evidence to become one member of the existing exact observed effect set without inventing additional effect data.

## Causality remains separate

A valid webhook merge observation establishes:

    authenticated delivery
    + semantic merged-PR effect

It does not establish:

    this async provider operation caused the merge

In particular, a valid `pull_request:closed` webhook cannot by itself reconstruct an expired asynchronous merge UUID or prove the operation-to-effect edge.

## Fail-closed cases

Reject:

- invalid HMAC;
- payload tampering after receipt capture;
- wrong event type;
- wrong action;
- closed but unmerged pull request;
- repository mismatch;
- PR-number mismatch against the reserved subject;
- requested head mismatch;
- missing merge commit SHA;
- contradictory top-level and nested PR numbers;
- payload digest mismatch;
- webhook evidence used as direct async-operation causality.

## GitHub interpretation

GitHub documents pull-request webhook payloads with repository and pull-request identity, and stacked pull requests additionally carry stack context in the pull-request webhook payload. This model consumes only the minimal merged-effect fields needed for the current effect-observation theorem and leaves stack-topology normalization to #7130/#7133.

## Claim ceiling

This establishes only deterministic parsing of authenticated pull-request webhook evidence into an effect observation in the provider-free reference model.

It does not establish provider-result causality, provider-side topology CAS, provider truthfulness, governance legitimacy, production atomicity, or successful external promotion.

Related: #7138, #7139, #7149, #7150, #7151.
