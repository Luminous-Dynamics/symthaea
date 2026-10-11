# Symthaea Algorithm Discovery Architecture v1

Status: proposed implementation contract
Tracking: #1318

## Purpose

Symthaea already contains multiple optimization, evolutionary, benchmarking, equality-saturation, and evidence systems. This architecture defines the smallest common contract needed to make algorithm discovery reproducible without granting discovery code production or repository authority.

## Non-equivalences

The following are intentionally distinct:

```text
Problem
!= Algorithm family
!= Implementation
!= Evaluation
!= Evidence
!= Promotion
!= Runtime authority
```

A discovered candidate is not production code. A benchmark win is not correctness evidence. A correct implementation is not globally optimal. A promoted implementation is not permission for an autonomous search process to modify the canonical repository.

## Trust boundary

Discovery is measurement-only. It may emit candidate artifacts, patches, manifests, lineage, and evaluation receipts. It may not directly mutate protected repository state, merge pull requests, authorize safety-critical behavior, or replace cryptographic/security primitives.

For exact algorithms, correctness is a hard admission gate. Performance objectives are considered only after the semantic contract passes.

Non-finite objective values are invalid evidence and may not participate in Pareto ordering.

## Core objects

- `ProblemSpec`: semantic problem identity and requirements.
- `AlgorithmRecord`: an algorithm family or strategy that can solve a problem.
- `ImplementationRecord`: one concrete implementation of an algorithm.
- `AlgorithmLineage`: parent candidates and transformations used to derive a candidate.
- `EvaluationSpec`: exact evaluator, oracle, corpus/generator, environment, and objectives.
- `EvaluationReceipt`: exact measured result bound to a candidate and environment.
- `PromotionRecord`: separate review decision; not part of v1 discovery authority.

All persistent identities are deterministic content identities over canonical fields. Human-readable names are metadata, not authority.

## Evidence model

Evaluation should compose with `symthaea-evidence-plane` rather than define a rival evidence hierarchy. At minimum every receipt must bind:

- problem identity;
- implementation identity;
- evaluator/oracle identity;
- input corpus or generator identity and seeds;
- repository/source revision;
- compiler/toolchain profile;
- target architecture/hardware profile when performance is measured;
- correctness verdict;
- finite objective measurements;
- run/evidence identity.

A candidate that fails correctness, violates a hard constraint, or emits non-finite measurements is ineligible for Pareto ranking.

## Discovery strategies

The registry is deliberately independent of candidate generation. Candidate generators may include:

- deterministic parameter sweeps;
- evolutionary search;
- mutation/recombination;
- equality saturation / rewrite extraction;
- constraint solving or synthesis;
- superoptimization;
- external or model-assisted code generation.

A generator has no promotion authority. New generators should implement the same candidate/lineage protocol instead of bypassing it.

## Initial consumer: HDC kernels

The first real discovery target should be exact, deterministic HDC kernels with existing scalar/SIMD references and benchmark coverage. Suitable initial problems include Hamming distance/similarity, binding, bundling reductions, and batch nearest-neighbor scoring.

HDC is intentionally chosen before cryptography, robotics, medicine, or other safety-critical domains because correctness can be checked against deterministic reference implementations and performance can be measured with existing Criterion harnesses.

## Forge migration

The existing `symthaea-forge` experiment directly edits tracked source and can commit a perceived optimization win. Preserve it as historical prototype behavior, but do not extend that authority model.

The successor flow is:

```text
immutable baseline
  -> isolated candidate workspace
  -> candidate artifact/patch
  -> correctness gate
  -> measured evaluation
  -> candidate archive
  -> independent promotion review
```

Never:

```text
search -> mutate canonical working tree -> self-score -> auto-commit
```

## Initial PR stack

- AR-0/1: architecture and typed registry contracts.
- AR-2: reproducible evaluation receipts and evidence-plane bridge.
- AR-3: finite-safe reusable Pareto kernel.
- AR-4: isolated candidate archive/search protocol with zero repository authority.
- AR-5: bridge equality saturation into candidate lineage.
- AR-6: HDC kernel discovery laboratory.
- AR-7: independent verification adapters where tractable.
- AR-8: retire direct-mutation Forge behavior behind the sandbox protocol.

## Non-goals for v1

- autonomous source replacement;
- autonomous merge/commit authority;
- invention or promotion of cryptographic primitives;
- safety-critical controller promotion;
- claiming benchmark results transfer across hardware without remeasurement;
- claiming an algorithm is universally best when only a Pareto frontier is established.
