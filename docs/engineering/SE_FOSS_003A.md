# SE-FOSS-003A — Internal Cognition → Engineering Proposal Contract

Issue: #6195  
Parent: #6194  
Reference corpus: `docs/engineering/data/se_foss_003a_reference_v1.json`  
Canonical corpus SHA-256 (exact compact UTF-8 bytes): `ee1a66b8fa4fa3ccb312e14252363eb6c05c960f0d195de1623a3c770f6f59b3`

## Status

This is a docs/data source contract only. It defines deterministic synthetic semantics for projecting internal Symthaea cognitive outputs into engineering as proposals, hypotheses, retrievals, priorities, or requests.

It establishes no cognitive correctness, engineering truth, causal truth, requirement acceptance, proof validity, experiment authority, fabrication authority, procurement authority, or actuation authority.

## Governing theorem

```text
cognitive output
!= engineering model fact
!= causal fact
!= admitted evidence
!= requirement satisfaction
!= qualified design
!= physical authority
```

and:

```text
proposal usefulness
!= proposal truth
!= evidence strength
!= execution authority
```

## Producer families

The frozen source recognizes these producer families:

- Broca / language reasoning;
- HDC associative reasoning;
- LTC/CfC temporal reasoning;
- FEP / active-inference style epistemic prioritization;
- causal reasoning;
- provenance-bound engineering memory;
- global workspace / attention;
- world-model / counterfactual reasoning;
- operations research;
- formal-proof suggestion.

Producer identity does not itself establish correctness.

## Proposal boundary

A claim-bearing production implementation should eventually bind, directly or by exact reference:

- schema/profile;
- producer kind and exact implementation/profile identity;
- exact producer input snapshot;
- exact engineering subject/configuration generation;
- output/proposal kind;
- exact target refs;
- structured proposed content;
- rationale/provenance refs;
- ambiguity/uncertainty where available;
- memory/model/observation dependencies;
- requested deterministic next step;
- intended-use scope;
- expiry/currentness trigger;
- authority ceiling;
- deterministic proposal identity.

Free-form rationale may accompany a proposal but cannot carry claim-critical semantics by itself.

## Required downstream routing

```text
RequirementDraft
  -> requirement acceptance process

RelationHypothesis
  -> deterministic semantic / analytical / review path

CausalHypothesis
  -> evidence-bound claim obligations
  -> optional bounded causal-model admission

MaterialOrProcessCandidate
  -> MAT-ENG demand/target boundary
  -> scientific evaluation
  -> ENG-MAT projection

ModelRevisionRequest
  -> SE-MODEL calibration / validation / review path

MeasurementRequest / ExperimentRequest
  -> planning / safety / human authorization
  -> execution separately

FormalProofRequest
  -> Lean / Verus / Creusot / model checker
  -> checked proof receipt separately

ScenarioRequest
  -> SE-SYNTH exact scenario identity / execution
```

There is no generic proposal-to-authority transition.

## HDC boundary

```text
high HDC similarity
!= exact semantic equality
!= physical equivalence
!= functional interchangeability
!= causal support
!= evidence independence
```

HDC may improve search and retrieval. It cannot promote itself to engineering evidence.

## Memory boundary

Engineering memory composes ENG-LEARN #5002.

```text
remembered
!= observed physically

replayed many times
!= independently confirmed many times
```

Physical and simulated source classes remain distinguishable. Hidden-evaluator contamination remains a campaign failure under strict policy.

## Temporal / prospective boundary

For LTC/CfC and other predictive faculties:

```text
prediction committed before outcome
-> eligible prospective prediction record

prediction created after outcome
-> retrospective interpretation only
```

A later contradiction creates discrepancy evidence; it does not rewrite the historical prediction.

## FEP / epistemic-action boundary

FEP-style utility may rank information-seeking actions:

```text
measure X
run analysis Y
inspect interface Z
```

but:

```text
epistemic utility
!= evidence strength
!= experiment authorization
!= actuation authority
```

## Causal boundary

Compose ENG-EVID-CLAIM #5771.

```text
association
!= causal claim

causal-model edge
!= evidence-qualified causal fact

model-relative counterfactual
!= observed intervention
```

A bounded causal-model admission is specific to one exact model/profile.

## Global-workspace boundary

Broadcasting, repetition, salience, or priority cannot change epistemic or authority class.

```text
broadcast count
!= evidence multiplicity
!= authority
```

## Optimization boundary

Operations-research outputs remain candidates:

```text
Pareto optimal
!= validated
!= qualified design
```

Hard constraints and qualification remain external.

## Formal-proof boundary

```text
proof idea
!= formal obligation
!= checked proof receipt
```

Cognition may discover valuable proof obligations; the formal toolchain remains proof authority.

## Currentness

Every proposal binds an exact target engineering generation.

```text
proposal@G1
+ target now G2
!= current proposal@G2
```

Changed generations require review/rebinding or a fresh proposal.

## Reference corpus

The canonical corpus contains 24 ordered synthetic cases C01…C24 spanning:

- Broca drafts;
- HDC analogies/material proposals/common-source duplication;
- physical vs simulation memory retrieval;
- hidden-evaluator contamination;
- prospective/retrospective/contradicted temporal predictions;
- FEP measurement/experiment/priority suggestions;
- correlation vs bounded causal-model admission;
- counterfactual non-observation;
- global-workspace rebroadcast;
- optimization candidates;
- formal-proof requests;
- synthetic scenario requests;
- model-revision requests;
- stale-target review;
- synthetic PASS authority ceiling.

## Independent output dimensions

There is no universal `accepted`, `correct`, `engineering_ready`, or `trusted` flag.

The frozen corpus uses independent dimensions:

```text
ProposalDisposition
TargetCurrentnessDisposition
EvidencePromotionDisposition
CausalAdmissionDisposition
LearningUseDisposition
ProspectivePredictionDisposition
ExecutionAuthorityDisposition
```

## Qualification sequence

```text
#6195 source/data freeze
-> independent stdlib-only oracle
-> hosted exact-head qualification
-> common proposal/reference types
-> HDC + memory adapters
-> LTC/CfC prospective adapter
-> causal adapter
-> FEP epistemic-action adapter
```

The independent oracle must derive outcomes from raw inputs before comparing stored expectations.

## Claim ceiling

A future exact-head PASS may establish only faithful deterministic software semantics for keeping internal Symthaea cognitive outputs proposal-bounded and routing them to explicit downstream evaluation/acceptance paths.

It establishes no cognitive correctness, real engineering competence, physical truth, causal truth, scientific truth, requirement acceptance, checked proof, experiment authority, qualification, certification, fabrication, procurement, build, or actuation authority.
