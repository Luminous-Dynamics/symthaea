# IG-INTEGRATION-001 — protocol v0.1

**Status:** deterministic synthetic qualification design; no LLM campaign executed.  
**Owner:** [Symthaea #12775](https://github.com/Luminous-Dynamics/symthaea/issues/12775)

## Question

When important evidence has already been surfaced, does the system correctly integrate it into a structured decision representation and final choice?

## Research basis

Toxtli & Delaflor, *Hidden Profile Decision Making in Multi-Agent LLM Groups*, CSCW Companion '26, DOI 10.1145/3785651.3831525, report an exploratory study crossing synchronous/asynchronous communication with shared/hidden information for three-agent groups from four model families.

The authors report that the tested models could surface many unique facts while still failing to integrate those facts into the optimal decision. One model family benefited strongly from asynchronous communication (13/15 correct versus 4/15 synchronous); this is an early result, not a universal effect. Other model families remained below the three-candidate chance baseline despite high information-sharing rates. The paper proposes a dedicated synthesizer producing a candidate-by-attribute matrix and an integration-audit interface.

The paper's sample is limited to one task domain, one group size and 15 sessions per cell (with five valid sessions in one asynchronous cell). We use it as a method/design comparator, not as a transferable expected effect size.

## Core decomposition

Measure these capabilities separately:

`P(correct) = P(surface) \times P(integrate \mid surfaced)`

This is an operational decomposition for the benchmark, not a universal theorem.

### 1. Surfacing

An evidence atom is surfaced only when a valid delivery event establishes that it reached the group/synthesizer. Reachability in a graph or mention in a model-generated summary is not delivery. Duplicate mentions do not create new evidence.

### 2. Integration matrix

Before making its final choice, the synthesizer emits one row per unique evidence atom it uses:
- evidence ID;
- source agent ID;
- candidate/entity ID;
- attribute ID;
- polarity (supports/opposes/neutral);
- signed decision contribution;
- unresolved mapping where the model cannot determine the relation.

Ground-truth candidate/attribute/polarity/contribution labels remain evaluator-only. Unknown facts remain unknown. The synthesizer must not infer source authority from confidence or repeated mention.

### 3. Decision

The system emits a choice or abstention and binds it to the matrix it used. The evaluator separately checks:
- whether the matrix ranking is internally consistent with the decision;
- whether the matrix faithfully represents the gold evidence structure;
- whether the final choice is correct under the synthetic oracle;
- whether decisive evidence was complete enough to justify a choice.

A system can be internally consistent but wrong because its matrix is wrong. This must remain visible.

## Synthetic fixture model

Each fixture includes:
- candidates/options;
- a unique evaluator-only evidence catalog;
- source identity, candidate, attribute, polarity and signed gold contribution per atom;
- surfaced evidence IDs;
- synthesizer matrix rows and selected option;
- required decisive evidence IDs;
- expected per-metric scores and disposition.

The synthetic oracle uses an explicit additive contribution rule only within these fixtures. This is a bounded scoring construction, not a claim that real decisions or HiddenBench tasks are universally additive.

## Metrics

### Surfacing
- unique evidence coverage;
- decisive-evidence coverage;
- message/source diversity (deferred until integration with delivery receipts);
- repeated-mention inflation.

### Integration
- matrix recall over surfaced atoms;
- source attribution accuracy;
- candidate/entity mapping accuracy;
- attribute mapping accuracy;
- polarity accuracy;
- signed contribution exactness/error;
- omission of decisive positive or negative evidence.

### Decision
- matrix/choice consistency;
- gold-choice accuracy when all required decisive evidence is surfaced;
- correct abstention when decisive evidence is missing or the oracle is tied;
- unsupported-evidence / provenance rejection.

Never use mention frequency, agreement, HDC similarity, Φ or synchrony as substitutes for evidence integration.

## Treatment campaign

After reproducing the HiddenBench baseline under #12762, compare at matched provider/model, task, token, tool and latency budgets:
1. synchronous serialized debate;
2. asynchronous independent contributions;
3. length-matched serialized contributions;
4. dedicated synthesizer with structured evidence matrix;
5. synthesizer plus independent integration audit/challenge;
6. ablations removing source attribution, polarity, contribution weighting and deduplication.

The CSCW Companion result is a separate three-agent/one-task-domain comparator. Do not pool its scores with HiddenBench's 65 tasks. If reproducing its task, prompts, and source materials, pin the exact release, checkpoint IDs, substrate and valid/failed session accounting.

## Qualification targets — proposed, not current results

- reject 100% of fabricated evidence IDs and source-lineage mismatches in deterministic controls;
- achieve 100% invariance to duplicate mention rows in deterministic fixtures;
- at least 99% entity/attribute/polarity mapping on unambiguous structured fixtures;
- at least 95% oracle-choice accuracy when all decisive evidence is surfaced and the oracle is unique;
- no forced-choice reward when decisive evidence is missing or the gold matrix is tied;
- before any model claim, show a statistically supported integration improvement over standard debate at equal budgets.

## Statistical and claim boundaries

The independent unit is the task × randomized treatment block. Agents and message rows are nested observations. Keep failed/timeout/missing synthesis records. Report task-level confidence intervals and quality/cost Pareto frontiers. A local fixture pass does not establish model ability, human decision validity, consciousness, legitimacy or production readiness.
