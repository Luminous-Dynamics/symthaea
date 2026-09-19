# MATH-EXP-001 Manifest v2

Status: successor preregistration contract for factorized mathematical-retrieval/search experiments.

Predecessor: MATH-EXP-001 v1 at exact head `9bf1795ed8c7a7b2a99e2ed88fd8f9c3ee136810`.

v1 remains preserved. v2 exists because the research question has become richer than “same encoder, different retrieval policy.”

## Why v2 exists

MATH-EXP-001 v1 correctly froze challenge/corpus/budget/toolchain state, but it required every arm to share one `encoder_sha256`.

That prevents a clean causal comparison among now-distinct interventions such as:

- syntax-only conventional structure;
- syntax-only HDC;
- exact-normal-form conventional structure;
- exact-normal-form HDC;
- syntax + normal-form fusion.

v2 therefore separates **shared experimental invariants** from **arm-specific representation and retrieval identity**.

The change is not a relaxation of control. It makes the intervention axes explicit.

```text
shared mathematical objects / corpus / contamination boundary
                         |
shared normalization contract + exact normalizer implementation
                         |
shared toolchain / human-intervention policy / statistical plan
                         |
ONE shared hard resource budget
                         |
       +-----------------+------------------+
       |                 |                  |
       v                 v                  v
     Syntax          Normal form          Fusion
 conventional/HDC  conventional/HDC   conventional/HDC
       |                 |                  |
       +-----------------+------------------+
                         |
                preregistered contrasts
                         |
                     measurement
                         |
                    evidence plane
```

## Authority boundary

The manifest is always:

`authority = MeasurementOnly`

and preserves:

```text
retrieval similarity != mathematical equivalence
normal-form identity != theorem authority
HDC advantage != theorem truth
search success != novelty
Phi != correctness
```

Formal mathematical authority remains in MATH-SPEC / MATH-EVID / MATH-VERIFY.

## Shared contract

Every arm inherits exactly one root-level shared contract containing digests for:

- challenge set;
- corpus snapshot;
- knowledge/contamination boundary;
- source-object contract;
- normalization receipt contract;
- exact normalization implementation;
- toolchain/model manifest;
- human-intervention policy.

An arm cannot override any of these fields because they do not exist in the closed arm grammar.

This is particularly important for exact normal forms: syntax-only arms may not consume the normalizer, but the exact normalizer candidate is still frozen globally before evaluation so normal-form/fusion arms cannot drift independently.

## Explicit equal-budget contract

v2 replaces “same opaque budget digest per arm” with one shared root budget plus an accounting-policy digest.

The frozen hard ceilings include:

```text
retrieved_items_max
retrieved_item_bytes_max
retrieval_context_bytes_max
total_context_bytes_max
normalized_compute_units_max
wall_time_ms_max
proof_calls_max
solver_calls_max
search_nodes_max
candidate_count_max
retrieval_queries_max
```

and:

`unused_budget_reallocation = false`

Arms have no budget fields.

Consequences:

- a fusion arm cannot receive twice the item count;
- a fusion arm cannot receive twice the retrieval-context bytes;
- HDC cannot receive extra search nodes or proof calls;
- unused resources after an early solve are recorded as savings rather than reassigned;
- a baseline may leave retrieval budget unused without granting another arm extra resources.

The execution report must later bind actual consumption. The preregistration only freezes the ceilings/accounting law.

## Representation axes

Each arm independently freezes:

- `representation_channels`
  - `Syntax`
  - `ExactNormalForm`
  - or both;
- `representation_family`
  - `None`
  - `Lexical`
  - `CanonicalSparse`
  - `HDC`
  - `FusionConventional`
  - `FusionHDC`;
- exact `representation_sha256`;
- retriever family and exact `retriever_sha256`;
- retrieval-index manifest;
- fusion-policy digest when both channels are used.

`representation_sha256` is now **supposed** to differ between causal representation arms.

That difference is no longer mislabeled as experimental drift.

## Fusion law

Any arm consuming both syntax and exact-normal-form channels must:

- use exactly those two channels;
- use a fusion representation family;
- carry a `fusion_policy_sha256`;
- use the same root resource ceilings as every other arm.

Non-fusion arms are forbidden from carrying a fusion-policy digest.

This prevents an apparently stronger fusion arm from gaining an untracked second retrieval budget.

## Negative controls

v2 requires explicit arms for:

```text
LexicalRetrieval
RandomRetrieval
ShuffledHdcVectors
PermutedChallengeAssociations
```

rather than merely listing their names.

`MajorityStrategy` remains a required strategy control.

Control arms are mutually exclusive with negative-search-memory augmentation.

## Negative-search memory as a one-variable ablation

A `MemoryAugmented` arm must:

- set `negative_search_memory = true`;
- identify `parent_arm_id`;
- use the same representation channels/family/digest;
- use the same retriever family/digest;
- use the same index;
- use the same fusion policy;
- use no control intervention.

The parent must have `negative_search_memory = false`.

Q0 representation retrieval forbids negative-search-memory arms entirely. Memory belongs in later search experiments after representation mechanics qualify.

Thus:

```text
parent retrieval intervention
          |
          +-- no search memory
          |
          +-- same intervention + negative-search memory
```

is machine-checkable.

## Q0 endpoints

v1 had a Q0 stage but its closed primary-endpoint set was dominated by downstream theorem-search metrics.

v2 adds retrieval-native primary endpoints:

```text
StructuralNeighborRecallAtK
EquivalentFamilyRecallAtK
MeanReciprocalRank
NormalizedDiscountedCumulativeGain
PositiveRankMargin
FalseEquivalentNeighborRate
RetrievalLatency
RetrievalNormalizedCompute
```

Q0 permits retrieval-native primary endpoints only.

Q1+ must contain at least one downstream search endpoint such as formally solved rate, useful-lemma rate, proof/search efficiency, false pruning, harmful transfer, or cross-domain transfer.

This prevents a representation benchmark from being retroactively judged by a theorem-search metric that was not part of its actual stage.

## Planned contrasts

v2 preregisters comparisons explicitly.

Every planned contrast freezes:

- left arm;
- right arm;
- primary endpoint;
- direction;
- intended difference.

Allowed intended differences are:

```text
RetrievalAugmentation
SyntaxVsNormalForm
HdcSpecificRepresentation
FusionBenefit
RetrieverMechanism
NegativeSearchMemory
Control
```

The semantic validator verifies that the named arms really exhibit the declared difference.

Examples:

```text
SyntaxVsNormalForm:
    {Syntax}  vs  {ExactNormalForm}

HdcSpecificRepresentation:
    same channel(s)
    HDC vs conventional representation family

FusionBenefit:
    one single-channel arm vs one Syntax+ExactNormalForm arm

NegativeSearchMemory:
    same retrieval intervention; only memory flag differs
```

This prevents choosing the most favorable comparison after results are known.

## Manipulation checks

All stages require:

```text
RetrievalDiffersFromBaseline
RepresentationInterventionApplied
BudgetAccountingComplete
```

Q0 additionally requires:

`StructuralNeighborShift`

Q1+ additionally requires:

```text
StrategyDistributionShift
SearchTrajectoryShift
```

If the intervention did not actually change representation/retrieval/search behavior, a downstream null cannot be interpreted as evidence that the mathematical idea itself failed.

## Recommended representation ladder

The current research sequence should distinguish at least:

```text
A   no retrieval
L   lexical retrieval
S   canonical sparse syntax
H   structural HDC syntax
N   exact-normal-form conventional
F   syntax + normal-form conventional fusion
FH  syntax + normal-form HDC fusion

plus:
R      random retrieval
SHUF   shuffled HDC vectors
PERM   permuted challenge associations
```

This supports separately answerable questions:

```text
Does retrieval help at all?
Does structure beat lexical surface?
Does exact normalization help?
Does HDC add value beyond conventional structure?
Does fusion add value beyond either channel alone?
Does negative-search memory add value after retrieval is held fixed?
```

No single overall “best arm” claim is required.

## Historical evidence preservation

The earlier 22/31 learned-cascade HDC result matching the 22/31 majority baseline remains negative evidence.

A structural or normalized representation does not erase that result. It begins a new intervention lineage designed to test the hypothesis that the earlier surface representation was too weak.

Likewise, MATH-REP v1/v2 normalization failures and repairs remain preserved in their own lineages.

## Validator

`.github/scripts/validate-math-search-experiment-v2.py` is stdlib-only.

It checks, among other things:

- closed field grammar;
- preregistration state;
- shared-contract digests;
- explicit budget consistency;
- no per-arm budget/normalizer/source override;
- stage-appropriate endpoints;
- required control arms;
- representation/channel consistency;
- fusion-policy requirements;
- one and only one baseline;
- memory-parent equivalence;
- planned-contrast semantics;
- required manipulation checks;
- seed uniqueness;
- evolutionary-search exclusion.

The validator's self-test includes adversarial field-smuggling attempts specifically to ensure fusion or another treatment cannot inject local resource/normalizer overrides.

## Nonclaims

A valid v2 manifest does not establish that:

- any implementation compiles;
- any experiment executed;
- any arm passed;
- HDC improves retrieval;
- normalization improves retrieval;
- fusion improves theorem search;
- negative-search memory helps;
- any mathematical claim is true;
- any result is novel;
- runtime contamination was absent.

It only makes later empirical comparisons causally interpretable.
