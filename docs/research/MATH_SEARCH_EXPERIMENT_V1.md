# MATH-EXP-001 Manifest v1

Status: preregistration contract for controlled mathematical-search experiments.

## Question

Does Symthaea's structural HDC machinery improve mathematical search outcomes beyond conventional retrieval under equal resources?

This contract measures **search effectiveness**, not mathematical truth.

```text
HDC similarity != proof
Phi != correctness
retrieval quality != formal authority
```

Every mathematical result produced by an experiment remains subject to the normal MATH-SPEC / MATH-EVID / MATH-VERIFY authority chain.

## Frozen identity

Before held-out evaluation, freeze SHA-256 commitments for:

- exact challenge set;
- permitted corpus snapshot;
- knowledge-boundary/contamination policy;
- statistical analysis plan;
- equal-budget contract;
- model/toolchain identities;
- human-intervention policy;
- structural encoder;
- random seeds;
- primary and secondary endpoints;
- experiment arms, negative controls, and manipulation checks.

`frozen_before_evaluation` must be true. A changed encoder, corpus, endpoint, budget, or challenge set starts a new evidence lineage.

## Experimental arms

v1 requires at least:

- **A — prover/search baseline:** no retrieval augmentation beyond required definitions;
- **B — conventional retrieval:** non-HDC retrieval under the same total context/resource budget;
- **C — structural HDC:** structural HDC retrieval/analogy, no negative-search memory;
- **D — structural HDC + negative-search memory:** optional until MATH-SEARCH-001 is available, but if present it must enable that intervention explicitly.

Evolutionary search is forbidden in v1. It is intentionally deferred so an HDC effect cannot be confounded with a second search intervention.

Every arm must bind exactly the same challenge set, corpus, knowledge boundary, toolchain, encoder identity, and budget contract. The encoder identity is shared even for non-HDC arms so the experiment manifest itself cannot silently mutate between arms; non-HDC arms simply do not consume the structural representation for retrieval.

## Primary endpoints

Primary endpoints are chosen before evaluation from a closed set including:

- formally solved rate;
- verified useful-lemma rate;
- valid counterexamples;
- proof calls per solved challenge;
- search nodes per solved challenge;
- normalized compute per solved challenge;
- time to first useful lemma;
- cross-domain transfer rate;
- repeated-failure rate;
- false-pruning rate.

Phi/elegance may be recorded as a secondary diagnostic but is deliberately absent from the primary endpoint enum.

## Required negative controls

At minimum:

```text
RandomRetrieval
LexicalRetrieval
ShuffledHdcVectors
PermutedChallengeAssociations
MajorityStrategy
```

The historical surface-token HDC null remains evidence and must not be discarded merely because a structural encoder is introduced later.

## Required manipulation checks

A downstream null is interpretable only if the HDC intervention actually changed search behavior. v1 therefore requires:

```text
RetrievalDiffersFromBaseline
StructuralNeighborShift
StrategyDistributionShift
SearchTrajectoryShift
```

If these fail, the correct conclusion is that the intended intervention was not demonstrated—not that structural mathematical analogy has been disproved.

## Equal-budget law

All arms bind the same budget-contract digest. That contract should cover, at minimum:

- wall/normalized compute budget;
- context/retrieved-byte budget;
- Lean calls;
- SMT/SAT/CAS calls;
- candidate count;
- theorem-decomposition budget;
- maximum search nodes;
- timeout policy.

Unused budget after an early solve is recorded as savings and may not be invisibly reallocated to improve that arm.

## Contamination law

Every arm binds the same frozen knowledge-boundary manifest. Blind rediscovery must remain distinguishable from retrieving a known solution. Any external source access, human intervention, or corpus change after freeze starts a new lineage unless explicitly permitted by the frozen policy.

## Machine checks

- `.github/schemas/math-search-experiment-v1.schema.json` defines the closed interchange shape.
- `.github/scripts/validate-math-search-experiment.py` enforces cross-arm equality, intervention isolation, mandatory negative controls/manipulation checks, seed discipline, and preregistration state.

The stdlib validator is normative for invariants JSON Schema cannot fully express.

## Failure interpretation

If HDC is null or worse:

1. preserve the result;
2. inspect preregistered manipulation checks;
3. do not change endpoints post hoc;
4. improve the encoder only in a new lineage;
5. rerun on fresh held-out problems.

If HDC is positive:

1. replicate with new problems/seeds;
2. compare with a stronger conventional retriever;
3. ablate structural channels;
4. verify the gain is not extra context or compute;
5. retain formal proof verification as a separate authority gate.

## Nonclaims

A valid manifest does not establish that:

- the experiment executed;
- HDC improves mathematical search;
- any theorem is true;
- a discovered result is novel;
- contamination was absent at runtime.

It only freezes the protocol needed to make those later empirical claims interpretable.
