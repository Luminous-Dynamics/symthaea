# ASSURE-002B convergence qualification and claim boundary

This note is part of the ASSURE-002B candidate tree. It describes the theorem the source is designed to establish; it does **not** self-assert that any Git head has passed qualification. Qualification is external evidence produced by the permanent read-only workflows for an exact candidate head.

## Convergence lineage

ASSURE-002B is constructed from two independently qualified parents:

- campaign parent: `8f619541528362a30e7caf432eaf256a64e302e8`;
- shared-semantics parent: `a95bbb2d0eac395b1d734e1031f9209240f3b8d6`.

The candidate product tree contains the `crates/core/symthaea-assurance-semantics` subtree byte-for-byte from the qualified shared-semantics parent. The campaign-local semantic commitment implementation is removed; the shared schema-bearing semantic commitment becomes authoritative.

The product delta contains no `.github/workflows/` mutation. Temporary writer/staging branches used while deriving earlier candidate trees are historical process evidence only and are not product ancestry or qualification authority. The final candidate is an exact two-parent Git commit over the qualified inputs, and exact-head product qualification remains read-only.

## Evidence-context boundary

An evidence artifact is admissible only when its committed context matches the exact campaign context before evidence-kind, duplicate, timing, admission-order, or evidence-root mutation checks:

```text
evidence.subject_id == campaign.subject_core_id

evidence.claim_digest == campaign.claim_digest
```

A registered evidence kind by itself is therefore insufficient to establish that evidence belongs to the campaign subject or claim.

## Temporal provenance boundary

The base theorem is commitment ordering:

```text
terminal registration in supplied view
    < exact evidence-artifact commitment
    < evidence admission
```

This establishes that the exact evidence commitment was durably ordered after the terminal registration represented by the supplied view. It does **not** establish that the underlying evidence bytes were first produced after registration.

Therefore:

```text
CommittedAfterTerminalRegistrationInView
    != ProductionWitnessedAfterRegistration
```

A stronger production theorem requires causal execution evidence binding the exact registration, execution request, real execution, and exact output/evidence commitment with authority outside the candidate worker.

## Currentness boundary

ASSURE-002B resolves a unique terminal unwithdrawn registration only inside the registration/withdrawal view supplied to the resolver.

Therefore:

```text
TerminalRegistrationInView
    != CurrentThroughVerifiedCheckpoint
```

The base kernel does not claim global completeness, authoritative external currentness, or absence of a later registration outside the supplied view.

## Ordering and authorization boundary

A valid ordering receipt establishes the bounded ordering relation defined by its committed validation profile. It does not prove that the ordered event was authorized by an external policy authority.

Therefore:

```text
OrderedEvent
    != AuthorizedEvent
```

## Canonicalization and arithmetic boundary

ASSURE-002B uses the shared semantic-set canonicalizer, explicit support-tier ranking, minimal unsigned-decimal ASCII for public `u64` wire fields, and checked evidence-ordinal advancement. Ordinal overflow fails closed before statement or ledger transition identity can wrap.

## Qualification requirement

No source-tree statement substitutes for exact-head execution. A candidate is not qualified until the permanent read-only workflows establish, for that exact head, at minimum:

```text
exact checkout binding
Rust 1.96 formatting
cargo check
focused tests
strict Clippy -D warnings
committed Cargo.lock parity
tracked-checkout immutability
```

Because ASSURE-002B also adds ASSURE-002F evidence-context accessors to `symthaea-assurance-core`, the final convergence head must satisfy both the existing `ASSURE-000 Qualification` and `ASSURE-002 Qualification` lanes triggered by the product delta.

A passing campaign does not establish universal AI safety, certification, regulatory conformity, global currentness, causal production provenance, or authorization beyond the precisely implemented and tested relations above.
