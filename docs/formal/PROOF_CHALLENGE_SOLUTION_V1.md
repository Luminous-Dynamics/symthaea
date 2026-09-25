# Proof Challenge / Solution V1

Status: trust-topology architecture only  
Tracking issue: #5779  
Parent: Proof Observability V1 / #5778

## Purpose

Separate the trusted statement of what must be proved from the candidate artifact that attempts to prove it.

```text
trusted challenge
!= candidate solution
```

The challenge fixes theorem/spec identity, permitted assumptions, subject identity, coverage requirement, claim ceiling, negative controls, and checker profile before a candidate proof is considered.

The solution may satisfy that contract; it may not rewrite it.

## Challenge

`ProofChallengeV1` binds:

- obligation ID;
- human claim;
- exact formal statement and digest;
- permitted axioms and trust roots;
- exact source/model subject digest;
- evidence class sought;
- required domain-coverage digest;
- required negative controls;
- explicit claim ceiling;
- admitted checker profile.

The canonical challenge digest is computed over these semantics.

## Solution

`ProofSolutionV1` binds:

- exact challenge digest;
- proof/certificate artifact digests;
- producer toolchain identity;
- checker receipts;
- result;
- producer warnings.

The solution does not contain an independently editable theorem statement or claim ceiling. Those come from the challenge.

## Checker profiles

### ProducerOnly

The producing verifier reports success, but no admitted independent replay/certificate checker is retained.

### ChallengeComparator

An independent comparator establishes that the candidate targets the exact challenge statement/subject/axiom policy.

### KernelReplay

The accepted proof artifact is replayed through an admitted proof-kernel/checker path.

### IndependentCertificate

The producer emits a certificate that is verified by a separately pinned checker where the lane supports this topology.

These profiles are not interchangeable and do not imply identical trust bases.

## Admission policy

Challenge/solution separation is intended for high-risk lanes such as:

- cryptographic/security authority theorems;
- safety-critical authority claims;
- AI-produced or substantially AI-rewritten proofs;
- external SMT results used for strong claims;
- proof results that materially widen an evidence or authority label.

Ordinary low-risk proof maintenance may remain on the normal capsule path.

## Negative controls

A candidate must be rejected when:

1. its challenge digest does not match;
2. it changes the theorem statement;
3. it requires an axiom not permitted by the challenge;
4. it targets a different source/model subject;
5. it presents a certificate for a different problem;
6. producer PASS conflicts with an admitted independent checker FAIL;
7. checker identity/version drifts without qualification;
8. advertised human claim exceeds the challenge claim ceiling.

## Trust rule

Multiple checkers reduce some classes of trust concentration but do not magically remove shared assumptions.

```text
kernel replay
!= theorem-intent correctness

challenge/solution agreement
!= source refinement

independent certificate check
!= runtime authority

multiple checkers
!= elimination of shared assumptions
```

## AI-generated proofs

An AI system may propose a solution, proof script, lemma structure, tactic sequence, or certificate-producing workflow. It may not mutate the trusted challenge in the same qualification step.

This makes AI assistance easier to admit because review can focus separately on:

```text
Did we ask the right theorem?
Did the candidate prove exactly that theorem?
Did an admitted checker accept the result?
```

## V1 scope

V1 defines and validates the content-addressed challenge/solution contract with synthetic fixtures. It does not yet claim production Lean comparator integration, `lean4checker` execution, Alethe/Carcara execution, or independent qualification of any checker. Those become separate adapter tranches.
