# Promotion Stack Identity v1

This tranche makes the stack-operation identity from #7096 executable in the provider-free
reference model.

## Identity

A single PR head is not sufficient to identify a stacked merge operation.

The reference identity binds:

    repository
    requested PR number
    requested PR head SHA
    base ref
    observed base tip SHA
    ordered stack entries
    each entry's PR number
    each entry's head SHA
    each entry's base ref
    each entry's base head SHA
    merge method
    merge action
    trust-root generation
    governance generation

The resulting canonical JSON bytes are SHA-256 addressed. Ordering is semantic: reversing stack
membership changes the identity rather than being normalized away.

## Why this is distinct from provider CAS

GitHub's asynchronous stacked-PR merge endpoint can merge or queue every open PR in the stack up
to the requested PR. The provider's expected-head check therefore protects the requested PR's
head, but that is not equivalent to protecting the entire local stack identity or the current
base tip.

A deterministic local operation digest lets the authority core detect:

    changed lower-stack head
    changed stack membership/order
    changed base tip
    changed requested PR/head
    changed merge parameters
    changed trust/governance generation

before a reservation or dispatch is reused.

It does not claim that GitHub consumes or enforces this digest.

## Adversarial invariants

The reference tests require:

- identical identities produce identical canonical bytes and digest;
- stack order changes the digest;
- a lower-stack PR head change changes the digest;
- base-tip movement changes the digest;
- merge-method or merge-action changes the digest;
- trust-root or governance generation changes the digest;
- requested PR number or requested head SHA changes the digest.

## Claim ceiling

This establishes only deterministic identity semantics in the provider-free reference model.

It does not establish:

- that GitHub's observed stack topology is truthful;
- that the provider atomically honors the local digest;
- production implementation correctness;
- external merge causality;
- governance legitimacy;
- successful promotion.

Related: #7096, #7097, #7101.
