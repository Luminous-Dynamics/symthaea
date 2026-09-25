# Proof Review Capsule V1

Tracking issue: #5773

## Purpose

Formal proof review should not require a human to reread every unchanged proof, log, translation artifact, and dependency after every update. The machine should re-check the exact subject; humans should review the semantic delta that can change the meaning or trust boundary.

This contract adds a compact, content-addressed review surface over the existing formal-verification architecture. It does not create a new proof system and does not replace the proof-obligation DAG, evidence composition rules, proof audit, axiom gate, or independent qualification work.

## Core rule

**Prove broadly; review narrowly.**

Machine verification remains exhaustive for the admitted subject. Human review is optimized around the semantic delta frontier:

- claim or theorem statement;
- assumptions, axioms, and trust roots;
- exact source/model subject;
- dependency/composition identities;
- domain-coverage argument;
- negative-control behavior;
- claim ceiling.

Unchanged details remain available by progressive disclosure but do not need to be manually reread solely because a downstream capsule was regenerated.

## What a capsule answers

The default view should fit on one screen and answer seven questions:

1. What is claimed?
2. What exact subject is covered?
3. Why does the proof cover the stated domain?
4. What is assumed or trusted?
5. What changed since the previous qualified capsule?
6. What machine checks and negative controls ran?
7. What does this not prove?

A capsule is review metadata. It is not itself a proof.

## Progressive disclosure

The preferred review order is:

1. human claim and exact formal statement;
2. assumptions, axioms, and trust roots;
3. dependency/composition DAG;
4. domain-coverage witness;
5. counterexamples and negative controls;
6. checker/toolchain receipts;
7. generated/extracted artifacts;
8. raw proof source and logs.

The fast path hides unchanged detail; it never removes it.

## Content-addressed identity

The capsule digest is SHA-256 over canonical JSON with sorted keys and compact separators, excluding any derived digest field.

Semantically relevant input includes the claim, formal statement, subject identity, assumptions, axioms, trust roots, dependencies, coverage witness, receipts, negative controls, and claim ceiling.

If any of these changes, the capsule identity changes.

A parent capsule that references a changed child identity becomes stale until the composition is re-qualified. This is how review effort follows the changed dependency frontier instead of the entire historical graph.

## Infinite and parametric domains

Infinite domains are not reviewed by enumerating their members. A proof capsule must record the finite or generic reason the theorem covers the whole domain.

### Universal proof

A theorem directly quantified over the domain may cover it with one generic proof: for all x in D, P(x).

### Induction or well-founded decomposition

Use a basis plus a preservation theorem and a constructor/well-founded coverage theorem. For natural numbers, a base case and successor step can cover infinitely many values because the induction principle covers every constructor-generated natural.

### Partition cover

For D = union over i of D_i, prove the branch theorem parametrically in i and separately prove that the indexed family covers D. The index set itself may be infinite.

### Generator plus closure

Show that a basis generates D, prove P on the basis, and prove every admitted generator/composition operation preserves P. This is useful for algebraic objects, protocol traces, syntax trees, and other constructor-generated domains.

### Quotient or symmetry reduction

Prove representative coverage and prove the property is invariant under the equivalence relation or group action. A finite representative basis can then cover an infinite symmetry class when the quotient theorem is exact.

### Coinduction or bisimulation

For infinite streams, reactive systems, or unbounded traces, prove a relation or invariant is preserved by every next transition instead of enumerating executions.

### Finite approximation plus limit transfer

This is valid only when the mathematics supplies both convergence and a theorem transferring the property to the limit. Many finite tests do not imply an infinite or limiting theorem.

## Coverage witness

Every nontrivial domain claim records:

- coverage kind;
- coverage scope;
- domain identity;
- basis or generic theorem family;
- coverage theorem;
- preservation/closure theorems when required;
- explicit exceptional or boundary cases.

Allowed scopes are finite, exact-universal, bounded, quotient-relative, and asymptotic.

A finite-case proof may not be relabeled exact-universal without a coverage theorem.

## Composition vocabulary

The v1 review vocabulary is deliberately small:

- DependsOn
- Implies
- Conjunction
- Refines
- PartitionCover
- Induction
- GeneratorClosure
- QuotientInvariant
- Bisimulation
- LimitTransfer

These labels describe how already-qualified evidence is related. They may preserve or narrow authority; they may not invent a stronger evidence class.

## Claim review versus proof review

Kernel acceptance is necessary but not sufficient for meaningful review. A machine can prove a theorem whose statement is weaker, differently scoped, or simply unrelated to the human claim.

The fast path therefore places the human claim beside the exact formal theorem/spec identity. Statement drift is a semantic change and invalidates a stale proof receipt.

Lean's `#print axioms` and Symthaea's proof-audit/axiom-gate machinery remain machine checks. They expose transitive axiom dependence but do not decide whether the theorem statement expresses the intended claim.

## External solver receipts

When external SMT is used, prefer independently checkable proof or certificate artifacts when practical rather than retaining only a solver verdict. The admissible checker and trust boundary must remain explicit. This is an admission preference, not a requirement to add a new solver/checker to every proof lane.

## Semantic delta review

A capsule is fast-path reviewable only when all of the following are unchanged or explicitly re-reviewed:

- human claim and formal statement;
- assumptions/axioms/trust roots;
- exact subject identity;
- prerequisite capsule identities and composition relations;
- coverage witness;
- negative-control contract;
- claim ceiling.

Machine checks may still need to rerun even when the semantic delta is empty.

## Required negative controls

The v1 validator must reject at least these classes:

- theorem statement changed while a stale statement-bound receipt is retained;
- assumption added but omitted from the capsule;
- dependency identity changed while the parent is treated as current;
- claim ceiling widened without semantic review;
- finite cases promoted to exact-universal coverage;
- partition branches without a coverage theorem;
- induction without basis or preservation;
- finite approximants promoted to a limit theorem without convergence and transfer;
- proof/checker receipt bound to a different formal statement.

## Nonclaims

- proof capsule != proof
- proof DAG consistency != theorem truth
- compact review != reduced verification
- finite basis != universal coverage without a coverage theorem
- many finite tests != infinite-domain proof
- composition metadata != a stronger evidence class

## Intended ergonomics

The system should make the rigorous path easy to inspect and the ordinary path cheap to approve. A reviewer who trusts unchanged, independently qualified prerequisites should usually need to inspect only the claim, semantic delta, coverage witness, assumptions, claim ceiling, and fresh machine receipts.

The result is not weaker review. It is less repeated review of unchanged material.
