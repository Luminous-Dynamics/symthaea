# SYM-INT-001A — Integral → Symthaea analytical crosswalk

Status: architecture/mapping only. No production or governance authority claim.

Tracks Symthaea #6172. Companion semantic crosswalk: Mycelix #3194 / PR #3195 (`MYC-INT-007C`).

Observed Integral source snapshot: 2026-09-27.

## 1. Placement theorem

Mycelix remains the semantic and institutional root. Symthaea is optional analytical infrastructure.

```text
Mycelix
= semantic identity
+ source state
+ evidence/provenance
+ governance lineage
+ authority
+ execution/effect receipts

Symthaea
= analysis
+ simulation
+ prediction
+ design search
+ optimization candidates
+ diagnostics
+ recommendations
+ verifiable computation
```

Therefore:

```text
Symthaea output
!= Mycelix source fact
!= Integral governance decision
!= authorization
!= executed effect
```

This crosswalk intentionally prevents an AI/analysis engine from becoming a shadow governance system.

## 2. Integral source treatment

Use the same source-authority ladder as `MYC-INT-007C`:

```text
White Paper
    conceptual / normative source

Development Guide
    builder proposal / seam map

Technical Specifications
    builder-facing contract at each source status

Decision Record
    project decision provenance

Q&A / system pages / node guide
    explanatory / operational context unless elevated
```

Symthaea never receives authority merely because a statement appears in a higher-authority Integral document. Document authority determines **how the adapter interprets a source**, not whether Symthaea can act.

## 3. Adapter shape

Preferred flow:

```text
Integral/Mycelix source-owned state
            ↓
read-only AnalysisRequest
            ↓
Symthaea bounded model/profile
            ↓
AnalysisArtifact
            ↓
Mycelix evidence/provenance admission
            ↓
Integral consumer
```

The result path must never shortcut the institutional path:

```text
Symthaea Recommendation
        X
        └──────────────► external effect
```

Instead:

```text
Recommendation
    ↓
CDS/OAD/COS/ITC review
    ↓
Decision / admission
    ↓
Authorization if needed
    ↓
Effect attempt
    ↓
Receipt
```

## 4. Common analysis request contract

A future read-only bridge should bind at minimum:

- request semantic identity;
- exact input `SemanticRef`s;
- exact evidence/source frontier;
- source-currentness profile;
- analysis purpose;
- permitted model/profile identity;
- assumptions supplied by caller;
- prohibited data/authority surfaces;
- resource/time budget;
- privacy profile;
- required output kinds;
- required uncertainty/calibration profile;
- applicability envelope;
- request provenance.

The request should not contain an implicit `execute = true` lane.

## 5. Common analytical outputs

Prefer typed outputs rather than one generic assistant response.

Candidate result kinds:

- `AnalysisResult`
- `Prediction`
- `CounterfactualResult`
- `DiagnosticFinding`
- `DesignCandidate`
- `OptimizationCandidate`
- `RiskOrConstraintFinding`
- `Recommendation`
- `ProofReceipt`
- `Abstention`
- `Unknown`

Each should preserve:

- exact request ref;
- exact input refs/frontier;
- model/profile/version;
- method/derivation identity;
- assumptions;
- output values/artifacts;
- uncertainty/calibration fields when meaningful;
- applicability envelope;
- known limitations;
- claim ceiling;
- optional proof receipt;
- `authority = none` as the default invariant.

## 6. Integral five-system crosswalk

### 6.1 CDS — Collaborative Decision System

Symthaea role: **decision support, not decision maker**.

Potential capabilities:

- simulate candidate policies/options under declared assumptions;
- generate bounded counterfactuals;
- compare consequences across declared criteria;
- sensitivity analysis;
- uncertainty propagation;
- constraint-conflict detection;
- identify missing evidence;
- surface trade-offs and dissent-relevant consequences;
- generate scenarios the deliberative body may have missed;
- post-decision comparison of observed outcomes to preregistered intent.

Useful output path:

```text
CDS Issue
+ options
+ evidence
+ constraints
+ review basis
        ↓
Symthaea Counterfactual/Prediction profile
        ↓
Prediction[]
TradeoffFinding[]
Recommendation[]
Abstention/Unknown
        ↓
Mycelix provenance
        ↓
CDS deliberation
```

Never:

- choose the accepted option;
- cast or weight votes;
- assign standing/expertise/governance authority;
- turn model confidence into authority;
- rewrite the original decision's success criteria after outcomes arrive.

Core theorem:

```text
better decision support
!= automated governance
```

### 6.2 OAD — Open Access Design

Symthaea role: **design/discovery engine**.

This is likely the strongest non-FRS fit.

Potential capabilities:

- generative design-space exploration;
- engineering constraint solving;
- topology/geometry alternatives;
- materials candidate discovery;
- material/property prediction under declared models;
- manufacturability checks;
- process-design exploration;
- robustness/disturbance analysis;
- energy/resource efficiency analysis;
- lifecycle/ecological scenario analysis;
- formal/numerical verification assistance;
- failure-mode generation;
- design trade-off explanations;
- later aesthetics/symbolic criteria as explicit objective channels rather than hidden preferences.

Proposed flow:

```text
OAD DesignProblem
+ requirements
+ constraints
+ evidence
        ↓
Symthaea design/discovery profiles
        ↓
DesignCandidate[]
OptimizationCandidate[]
RiskOrConstraintFinding[]
Prediction[]
ProofReceipt[]
        ↓
OAD evaluation/certification
```

Critical boundary:

```text
Symthaea DesignCandidate
!= Integral CertifiedDesign
```

A candidate may include evidence supporting certification, but certification remains owned by Integral/OAD rules and authorized reviewers.

### 6.3 ITC — Integral Time Credits

Symthaea role: **economic scenario and anomaly analysis**, never ledger authority.

Potential capabilities:

- stress test Integral's declared contribution/access rules;
- sensitivity analysis over decay/access parameters;
- detect unusual concentration/access patterns;
- analyze role bottlenecks;
- simulate contribution/access pressure;
- identify possible coercion or preference-pattern anomalies for human/system review;
- compare alternative policy envelopes under declared metrics;
- forecast capacity/access pressure from COS/OAD/FRS observations.

Never:

```text
Symthaea forecast
→ balance mutation
```

or:

```text
model score
= labor value
```

Preserve:

```text
LaborEvent
!= model inference

ITCLedgerEntry
!= Symthaea analysis

anomaly finding
!= violation established

policy recommendation
!= policy decision
```

Symthaea should consume the exact Integral ITC semantics from the adapter, not reinterpret ITC as MYCEL/SAP/TEND or invent a substitute value system.

### 6.4 COS — Cooperative Organization System

Symthaea role: **operations and production decision support**.

Potential capabilities:

- production scheduling candidates;
- resource-allocation scenarios;
- bottleneck detection;
- anomaly detection;
- predictive-maintenance candidates;
- process optimization;
- material/energy efficiency analysis;
- robotics task/planning candidates;
- disturbance/robustness simulation;
- capacity forecasting;
- alternate work-order sequencing;
- supply/material risk analysis;
- quality/failure prediction under qualified models.

Critical separation:

```text
Prediction(machine failure)
!= observed machine failure

OptimizationCandidate
!= authorized production plan

recommended robot action
!= actuation authorization

simulation result
!= implementation receipt
```

Physical-action authority remains downstream in Mycelix/COS/Xenia or another explicit control layer.

### 6.5 FRS — Feedback & Review System

Symthaea role: **primary optional analytical engine**.

Potential capabilities:

- multimodal signal fusion;
- source-preserving pattern recognition;
- temporal dynamics;
- anomaly detection;
- system-health modeling;
- hypothesis generation;
- causal-model candidates;
- forecasting;
- counterfactual review;
- uncertainty/sensitivity analysis;
- outcome-vs-intention assessment;
- recommendation generation;
- change-point/regime detection where qualified;
- private/verifiable analysis using ZK primitives where the computation fits supported circuits.

Canonical chain:

```text
SourceObservation(s)
        ↓
FRS/Symthaea analysis request
        ↓
DerivedFinding
Prediction
Counterfactual
Recommendation
        ↓
Mycelix provenance / FRS packet
        ↓
CDS/OAD/COS/ITC consumer
```

Required non-equivalence:

```text
SourceObservation
!= DerivedFinding
!= Prediction
!= Recommendation
!= Decision
!= Authorization
```

FRS is the best initial place to integrate Symthaea because the institutional separation is naturally explicit: sensing/analysis informs governance and operations without replacing them.

## 7. Mapping current Integral data structures to Symthaea

### SPEC-DS-01 — CertifiedDesign

Symthaea may consume the design and supporting evidence for:

- performance prediction;
- robustness analysis;
- manufacturability analysis;
- lifecycle/ecological analysis;
- supersession candidate generation.

It may emit new `DesignCandidate` or `RiskOrConstraintFinding` artifacts.

It must not rewrite certification state.

### SPEC-DS-02 — LaborEvent

Symthaea may use labor events as source-owned inputs to:

- capacity forecasting;
- role bottleneck analysis;
- workload-pattern analysis;
- economic simulations.

It must not infer an unobserved LaborEvent and insert it as fact.

### SPEC-DS-03 — MaterialConsumptionEvent

Symthaea may use these observations for:

- resource-intensity analysis;
- production efficiency;
- ecological/lifecycle estimates;
- redesign suggestions;
- demand forecasts.

Source ownership remains COS/operational domain.

### SPEC-DS-04 — ITCLedgerEntry

Symthaea may analyze ledger history/current-state witnesses for:

- trends;
- anomalies;
- stress scenarios;
- policy counterfactuals.

Never output a ledger entry as an analytical side effect.

### SPEC-DS-05 — FRSSignalPacket

Best alignment.

A packet may contain references to Symthaea-derived:

- diagnostic findings;
- predictions;
- recommendations;
- analysis profiles;
- proof receipts;
- uncertainty/applicability metadata.

The packet should reference, not erase, the source observations that fed the analysis.

### SPEC-DS-06 — DecisionPacket

Symthaea may consume a decision packet after adoption to establish:

- intended outcomes;
- review triggers;
- constraints;
- assumptions;
- monitoring targets.

Later analysis can compare outcome observations against those preregistered fields.

Symthaea must not turn the DecisionPacket into an Authorization.

## 8. Mapping the twelve cross-system seams

Symthaea is not itself a thirteenth mandatory Integral system. It sits behind selected analytical seams.

| Integral seam | Possible Symthaea role | Result type | Authority boundary |
|---|---|---|---|
| OAD → COS | simulate manufacturability/production alternatives before admission | `Prediction`, `RiskOrConstraintFinding` | COS decides/admit separately |
| OAD → ITC | estimate labor/material scenarios for analysis | `Prediction` | estimate != ITC valuation |
| OAD → FRS | analyze expected design behavior | `Prediction`, `DesignCandidate` | prediction != observation |
| FRS → OAD | propose recalibration/supersession candidates | `Recommendation`, `DesignCandidate` | OAD certification decision remains external |
| COS → ITC | detect anomalies/capacity patterns around source events | `DiagnosticFinding` | finding != ledger mutation |
| COS → FRS | primary operational-analysis input | `AnalysisResult`, `Prediction` | source observations remain COS-owned |
| ITC → FRS | stress/anomaly/access-pattern analysis | `DiagnosticFinding`, `Prediction` | FRS/Symthaea do not reconstruct authoritative account state from partial history |
| FRS → CDS | provide bounded evidence-backed analysis | `Recommendation`, `Prediction`, `CounterfactualResult` | no decision/authority |
| CDS → FRS | establish monitoring/review basis | analysis profile derived from decision | review profile cannot rewrite decision history |
| CDS → OAD | test design-mandate feasibility | `AnalysisResult` | mandate/admission remain external |
| CDS → COS | simulate production-mandate feasibility/capacity | `Prediction`, `OptimizationCandidate` | recommendation != work authorization |
| CDS → ITC | simulate policy consequences | `CounterfactualResult`, `Prediction` | no policy or balance mutation |

## 9. Symthaea capability families relevant to Integral

Use existing capability families only where their own evidence ceilings permit them.

### Core cognition / temporal modeling

The current Symthaea architecture exposes a four-phase cognitive loop:

```text
Perceive → Evolve → Measure → Act
```

Integral adapters should constrain this to a **read-only analysis mode** unless a separate authority layer explicitly permits action. For most Integral work the relevant shape is effectively:

```text
Perceive → Evolve → Measure → report artifact
```

not direct actuation.

HDC and CfC components may support representation and temporal modeling, but model-specific capability claims must inherit Symthaea's existing qualification/evidence state.

### Engineering / materials / discovery

Existing Symthaea research includes substantial domain modules and a materials/discovery evidence program. These can seed OAD experiments only under their individual claim ceilings.

Preserve:

```text
computational candidate
!= validated material
!= certified design
!= manufacturable product
```

### Spatial / robotics / control

Existing spatial, control and robotics theorem/research lines can support COS simulation and optimization.

Preserve:

```text
robust under declared synthetic model
!= physical-world guarantee
!= actuation authority
```

### Formal/proof infrastructure

Formal verification can be useful for bounded invariants such as:

- analysis artifact cannot mint authority;
- hidden evaluator/oracle data cannot enter runtime input;
- source observations remain source-addressable;
- result model/profile identity is immutable;
- expired/stale input profile cannot be represented as current;
- recommendation cannot serialize as Decision/Authorization.

Do not attempt to formally prove political legitimacy, economic desirability, or social optimality.

### ZKP / verifiable computation

Symthaea currently includes working research primitives for proving bounded computations/state transitions. For Integral, the useful pattern is:

```text
private inputs
+ declared computation profile
        ↓
Symthaea computation
        ↓
public result/commitment
+ ProofReceipt
```

Possible later examples:

- prove an eligibility predicate without exposing full identity attributes;
- prove a bounded aggregate/threshold calculation;
- prove a stated analysis transformation was run on committed inputs;
- prove a model/state transition followed a committed circuit/profile.

Critical theorem:

```text
proof that computation C was executed correctly
!= proof that C's assumptions/model match reality
```

## 10. Uncertainty, calibration and abstention

Integral should not consume a single undifferentiated `confidence` value from Symthaea.

Where meaningful, distinguish:

- model uncertainty;
- data/source uncertainty;
- evidence completeness;
- sensitivity to assumptions;
- calibration profile;
- applicability/domain-shift state;
- source freshness;
- numerical approximation error;
- `Unknown` / `Abstention`.

Never collapse these into governance authority.

A useful result envelope may carry:

```text
result_state:
  EstablishedUnderProfile
  NotEstablishedUnderProfile
  Unknown
  Inapplicable
  InvalidEvidence
```

only when the underlying analysis profile justifies those states.

## 11. Preventing circular evidence

The adapter must detect and reject/flag circularity such as:

```text
Symthaea Recommendation A
    ↓
CDS adopts policy using A
    ↓
policy-generated record
    ↓
record fed back as independent evidence that A was correct
```

or:

```text
FRS summary generated from source S
    ↓
summary re-imported as if independent source
    ↓
confidence amplified
```

Preserve derivation ancestry and dependence groups through Mycelix provenance.

## 12. Human legibility

Every consequential Symthaea artifact used by Integral should support a participant-facing explanation containing at least:

- what was asked;
- what inputs were used;
- what inputs were unavailable;
- model/profile version;
- assumptions;
- output;
- uncertainty/limitations;
- whether the artifact is observation, prediction, finding or recommendation;
- whether it has any authority (normally none);
- how to contest/recompute/review it.

Broca/NLG may render explanations, but generated wording must not become the evidence record itself.

## 13. First pilot — I0 water-system analytical companion

Reuse the existing Mycelix/Integral I0 synthetic water scenario rather than inventing another demo.

### Input

Read-only bundle could include:

- water-flow observation refs;
- purity/pressure observations;
- actor reports;
- evidence provenance;
- current issue/options;
- declared operating constraints;
- historical source observations available before cutoff.

No oracle answer and no future outcome should be visible.

### Symthaea task

Produce at least:

1. `DiagnosticFinding` — e.g. likely issue class under declared profile;
2. `Prediction` — expected consequences of each permitted option;
3. `CounterfactualResult` — what changes under alternative assumptions;
4. `Recommendation` — optional, explicitly advisory;
5. `Unknown/Abstention` where evidence is inadequate.

### Hidden evaluator

Future synthetic outcomes remain evaluator-only.

After a CDS decision and synthetic outcome observation, score:

- prediction calibration/accuracy;
- false positive/negative diagnostics;
- whether uncertainty was honest;
- whether recommendation quality improves over simple baselines;
- whether explanation cites the correct source/evidence frontier;
- whether any output crossed its authority boundary.

Do **not** score which political/governance choice was 'best'. Score analytical fidelity under preregistered synthetic truth/criteria.

## 14. OAD pilot — design/discovery companion

After the read-only FRS pilot, the highest-value second pilot is OAD.

Use a bounded engineering fixture with:

- explicit requirements;
- known constraints;
- synthetic or independently held-out property data;
- manufacturability limits;
- lifecycle/resource metrics;
- hidden evaluation values.

Compare:

```text
baseline design search
vs
Symthaea-assisted design search
```

on preregistered technical metrics only.

Retain:

```text
better search under fixture
!= universal design superiority
!= certification
```

## 15. COS pilot — production/disturbance companion

Later, construct a synthetic production cell with:

- jobs;
- machine capabilities;
- maintenance states;
- material constraints;
- stochastic/bounded disturbances;
- energy/resource budget;
- hidden realized failures.

Symthaea emits schedules/forecasts/recommendations; COS simulator executes independently.

Measure throughput/resource/failure prediction against baselines without granting runtime actuation authority.

## 16. What Symthaea should **not** own

Do not add Integral-specific ownership for:

- CDS decisions;
- voting/standing/delegation;
- OAD certification;
- ITC balances or valuation policy;
- COS source-of-truth production state;
- FRS source observations;
- local authorization;
- federation authority;
- execution receipts.

Those remain Mycelix/Integral/domain concerns.

Likewise, do not create a new `symthaea-integral-cds`, `symthaea-integral-frs`, etc. cognitive architecture unless a concrete reusable capability gap is first demonstrated.

## 17. Recommended implementation generations

```text
SYM-INT-001A
this source-aware mapping
        ↓
SYM-INT-001B
read-only AnalysisRequest / AnalysisArtifact schemas
        ↓
SYM-INT-001C
I0 water synthetic hidden-outcome corpus
        ↓
SYM-INT-001D
uncertainty / calibration / abstention profile
        ↓
SYM-INT-001E
selected verifiable-computation receipt
        ↓
SYM-INT-001F
OAD engineering/design-search pilot
        ↓
SYM-INT-001G
COS production/disturbance pilot
```

Executable work should reuse Mycelix semantic refs/provenance across the bridge rather than inventing Symthaea-local identities for external institutional objects.

## 18. Qualification gates

Before making an external claim that Symthaea improves an Integral workflow, require:

- exact source/input generation frozen;
- oracle/future outcomes hidden from candidate;
- baseline comparator frozen before results;
- analysis model/profile identity frozen;
- source/evidence frontier recorded;
- calibration/abstention policy frozen;
- mutation controls proving candidate cannot see evaluator answers;
- independent result scoring;
- authority-boundary tests;
- negative results retained;
- exact environment/evidence capsule.

Claim ladder:

```text
architecture mapped
< source implemented
< internally executed
< internally qualified
< directly compared
< externally reproduced
```

Do not skip levels in external communication.

## 19. Nonclaims

This crosswalk:

- does not endorse or oppose Integral's governance/economic choices;
- does not make Symthaea a governance actor;
- does not claim current experimental research modules are production-ready;
- does not claim AI analysis improves real-world governance outcomes;
- does not convert proof-of-computation into proof-of-world correctness;
- does not authorize physical action;
- does not make Mycelix dependent on Symthaea.

The target architecture is **institutional sovereignty with optional verifiable intelligence**, not algorithmic sovereignty.