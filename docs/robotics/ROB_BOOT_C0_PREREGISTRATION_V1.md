# ROB-BOOT C0 Preregistration Template v1

Status: **DRAFT TEMPLATE — CAMPAIGN NOT STARTED**

Authority: protocol/preregistration semantics only.

This document does **not** establish prerequisite qualification, fabrication authority,
commissioning, experiment authorization, physical safety, or an R2 result.

Related work:

- ROB-BOOT-001 #4818
- ROB-BOOT-000 #5078
- ROB-BOOT-000A #5107
- ROB-DESIGN-001C0 #5084
- ROB-JOINTLAB-001 #4814
- ENG-MEAS-001 #4919
- ENG-MAT-001 #5074
- ENG-LOAD-001 #5003
- ENG-MASSPROP-001 #4999
- ROB-REALIZE-001 #4859
- SE-VV-005A #4922
- ENG-LEARN-001 #5002

## 1. Benchmark question

The first C0 campaign asks only:

> Under one frozen JointLinkCoupon interface, load, material/process,
> measurement, controller, and safety profile, can a bounded redesign reduce
> realized physical mass by a preregistered engineering-relevant amount while
> remaining non-inferior in held-out physical deflection and without protected
> regression?

A successful C0 campaign is a calibration of the design-to-evidence pipeline.
It is not evidence of arbitrary CAD competence, general structural optimization,
humanoid competence, or superior search intelligence.

## 2. Protocol identity

All semantic fields below must be frozen before `CampaignStarted`.

| Field | Value |
|---|---|
| protocol version | `rob-boot-c0-preregistration-v1` |
| generation ID | **UNSET** |
| protocol commitment SHA-256 | **UNSET** |
| readiness snapshot | **UNSET** |
| baseline subject root | **UNSET** |
| search-domain root | **UNSET** |
| material/process root | **UNSET** |
| load/support profile root | **UNSET** |
| measurement-system root | **UNSET** |
| safety/consequence profile root | **UNSET** |
| analysis-plan root | **UNSET** |
| proposal-memory snapshot | **UNSET** |
| development partition root | **UNSET** |
| validation partition root | **UNSET** |
| confirmatory partition root | **UNSET** |

`UNSET` is fail-closed. No placeholder value may be interpreted as current evidence.

## 3. C0 design subject

Preferred first article: `JointLinkCouponV1`, initially restricted to a
solid rectangular section.

Frozen across every candidate in this generation:

- link length;
- root/fixture datum;
- load-application datum;
- load direction;
- deflection metrology datum;
- section orientation convention;
- material axes/reference frame;
- controller semantics;
- hard-stop/guard keep-outs;
- experiment load/support profile.

Allowed V1 design variables:

| Variable | Lower bound | Upper bound | Unit |
|---|---:|---:|---|
| section width | **UNSET** | **UNSET** | **UNSET** |
| section height | **UNSET** | **UNSET** | **UNSET** |

Any semantic change to the frozen interface or load/metrology datums creates a
new generation rather than another candidate in this generation.

## 4. Primary and protected claims

### Primary target

Physical as-built article mass.

Target direction: lower is better.

Minimum engineering-relevant reduction:

```text
minimum_mass_reduction = UNSET
```

The value must be frozen in native units before candidate search and justified
independently of candidate outcomes.

### Protected structural metric

Physical deflection at the frozen metrology datum under the exact frozen load.

Maximum allowed non-inferiority regression:

```text
maximum_allowed_deflection_regression = UNSET
```

The final protection decision uses physical measurement evidence, not analytical
prediction alone.

### Other protected constraints

| Constraint | Frozen bound/profile |
|---|---|
| geometry/envelope | **UNSET** |
| fixture/interface conformity | **UNSET** |
| material/process applicability | **UNSET** |
| fabrication capability profile | **UNSET** |
| stored-energy/consequence profile | **UNSET** |
| temperature/environment profile | **UNSET** |
| measurement validity/currentness | **UNSET** |

No composite score may rescue a hard-constraint violation.

## 5. Measurement-system readiness

Every final metric requires a qualified ENG-MEAS profile before candidate search.

### Mass measurement

| Field | Value |
|---|---|
| instrument identity | **UNSET** |
| calibration/currentness | **UNSET** |
| range/resolution | **UNSET** |
| zero/tare rule | **UNSET** |
| stabilization rule | **UNSET** |
| repeatability evidence | **UNSET** |
| between-session/intermediate precision | **UNSET** |
| uncertainty method | **UNSET** |
| run-level estimator | **UNSET** |

### Deflection measurement

| Field | Value |
|---|---|
| instrument identity | **UNSET** |
| calibration/currentness | **UNSET** |
| datum/frame identity | **UNSET** |
| range/resolution | **UNSET** |
| zero/reference rule | **UNSET** |
| settle/warm-up duration | **UNSET** |
| analysis window | **UNSET** |
| filter/derivation profile | **UNSET** |
| repeatability evidence | **UNSET** |
| remount/session reproducibility | **UNSET** |
| uncertainty method | **UNSET** |
| run-level estimator | **UNSET** |

If the measurement system cannot resolve the preregistered effect, the campaign
disposition is `MeasurementSystemDominatesEffect` or `Indeterminate`; the effect
threshold may not be relaxed after candidate results are observed.

## 6. Material and fabrication profile

| Field | Value |
|---|---|
| design material-state subject | **UNSET** |
| property-evidence root | **UNSET** |
| material lot/batch policy | **UNSET** |
| manufacturing process | **UNSET** |
| process orientation/state | **UNSET** |
| machine/process capability assumption | **UNSET** |
| witness-coupon evidence, if required | **UNSET** |
| actual fabrication manifest policy | **UNSET** |

Search-time material/process assumptions are not proof that the realized article
achieved that state.

## 7. Load/support profile

| Field | Value |
|---|---|
| ENG-LOAD subject | **UNSET** |
| support condition | **UNSET** |
| applied load magnitude/profile | **UNSET** |
| load direction | **UNSET** |
| load application datum | **UNSET** |
| gravity/orientation profile | **UNSET** |
| environment/temperature profile | **UNSET** |

The physical campaign must preserve deviations as discrepancy evidence rather
than rewriting the planned load subject.

## 8. Search budget and comparison baseline

| Field | Value |
|---|---:|
| maximum candidate proposals | **UNSET** |
| expensive evaluations | **UNSET** |
| physical prototypes allowed | **UNSET** |
| development physical runs | **UNSET** |
| shortlist size | **UNSET** |
| compute/wall-clock budget | **UNSET** |
| material/monetary budget | **UNSET** |

Where practical, retain an exhaustive/grid/random baseline under a matched
search domain/budget.

C0 may establish that the engineering pipeline works even when a simple baseline
finds the same candidate. Search-method superiority is a separate claim.

## 9. Evidence partitions

### Development

May be used for:

- measurement-system characterization;
- baseline characterization;
- system identification/model calibration;
- material/process characterization;
- optimizer tuning;
- candidate search;
- DOE exploration;
- sample-size/power planning.

Development baseline data are not fresh confirmatory baseline evidence.

### Validation

Optional. If used, exact allowed uses must be frozen here:

```text
validation_allowed_uses = UNSET
```

### Confirmatory

Fresh evidence unavailable to candidate design, model fitting, threshold
selection, and proposal-visible engineering memory.

After candidate selection freezes, collect fresh baseline and candidate evidence
under the same confirmatory protocol.

Confirmatory leakage requires a new generation/fresh holdout.

## 10. Experimental unit and replication

Distinguish explicitly:

```text
sensor sample != run != remount/session != fabricated article != fabrication lot
```

| Field | Value |
|---|---|
| claim scope: exact articles or process population | **UNSET** |
| baseline articles | **UNSET** |
| candidate articles | **UNSET** |
| runs per article | **UNSET** |
| remount/session blocks | **UNSET** |
| sample-size/power rule | **UNSET** |

Thousands of samples from one time series do not become thousands of physical
replicates.

## 11. Run order, blocking, and blinding

| Field | Value |
|---|---|
| run-order randomization rule | **UNSET** |
| blocking variables | **UNSET** |
| warm-up/thermal-state rule | **UNSET** |
| remount/reset rule | **UNSET** |
| opaque article/run ID policy | **UNSET** |
| analysis-label blinding policy | **UNSET** |
| known blinding limitations | **UNSET** |

Potential blocks include session/day, fixture remount, thermal state, calibration
epoch, article identity, and material/fabrication lot.

## 12. Frozen analysis rules

Before confirmatory exposure, freeze:

- raw-data parser identity;
- missing-sample rule;
- saturation rule;
- outlier/exclusion rule;
- filtering rule;
- run-level estimator;
- uncertainty method;
- baseline/candidate contrast estimator;
- conformity/guard-band decision rule;
- multiplicity policy;
- stopping rule;
- abort classification;
- final conjunction rule.

No analysis window, exclusion threshold, stopping rule, or primary metric may be
chosen after viewing confirmatory outcomes.

## 13. Candidate freeze

Before confirmatory exposure, bind:

| Field | Value |
|---|---|
| candidate RobotDesignId | **UNSET** |
| candidate geometry root | **UNSET** |
| candidate material/process subject | **UNSET** |
| candidate as-built article root | **UNSET** |
| topology/calibration/commissioning root | **UNSET** |
| controller profile | **UNSET** |
| prediction artifact root | **UNSET** |
| candidate-selection rationale root | **UNSET** |
| search/model/memory snapshot root | **UNSET** |

No redesign after confirmatory exposure under the same generation.

## 14. Safety and stored-energy profile

Bind the exact JointLab/SE-VV profile before physical confirmation.

| Field | Value |
|---|---|
| experiment energy/consequence class | **UNSET** |
| electrical stored-energy bound | **UNSET** |
| mechanical kinetic/potential bound | **UNSET** |
| compliance/spring-energy bound | **UNSET** |
| travel/speed/current/temp/load limits | **UNSET** |
| independent power-removal path | **UNSET** |
| containment/guard profile | **UNSET** |
| abort predicates | **UNSET** |
| achieved-safe-state observation rule | **UNSET** |

A lower-voltage system is not automatically a lower-consequence system.

## 15. Prediction preregistration

Before confirmatory testing, freeze predicted ranges/intervals for:

| Quantity | Prediction |
|---|---|
| physical mass | **UNSET** |
| physical deflection | **UNSET** |
| key material/model discrepancy | **UNSET** |
| relevant protected metric(s) | **UNSET** |

Preserve separately:

```text
physical improvement evidence
prediction calibration evidence
mechanism-attribution evidence
search-method performance
```

A physical win does not establish the other three.

## 16. Confirmatory decision

Default R2 conjunction:

```text
mass reduction >= frozen minimum engineering-relevant effect
AND
physical deflection satisfies frozen non-inferiority decision rule
AND
all protected hard constraints remain satisfied
AND
measurement evidence is adequate
AND
confirmatory partition remains uncontaminated
AND
candidate does not introduce a protected safety/fabrication regression
```

Allowed dispositions include:

- `QualificationBlocked`
- `ProtocolNotStarted`
- `NoFeasibleCandidate`
- `NoCandidateBeatsBaseline`
- `MeasurementSystemDominatesEffect`
- `InstrumentationInconclusive`
- `TargetImprovementTooSmall`
- `DeflectionProtectionNotEstablished`
- `ProtectedMetricRegressionObserved`
- `TradeoffEstablished`
- `SimulationImprovementDidNotTransfer`
- `PhysicalImprovementButPredictionPoor`
- `HeldOutImprovementSupportedExactArticles`
- `HeldOutImprovementSupportedProcessProfile`
- `ConfirmatoryProtocolContaminated`

## 17. Recomputable evidence capsule

Final reporting must retain enough content-bound evidence to recompute the
published metric/disposition from raw observations:

- protocol/preregistration root;
- baseline/candidate subject roots;
- raw instrument artifacts;
- calibration/metrology roots;
- parsed/normalized dataset;
- derived-measurement dataset;
- primary analysis;
- independent analysis crosscheck when available;
- predictions/model artifacts;
- fabrication/as-built lineage;
- Nix/flake/toolchain/execution environment;
- exact commands/configuration;
- final disposition root.

Track reproduction claims separately:

```text
NotReplayed
SelfRecomputed
EnvironmentReproduced
IndependentAnalysisCrosschecked
IndependentPhysicalReplication
```

Self-recomputation is not independent reproduction.

## 18. Start gate

`CampaignStarted` is forbidden while any required field above remains `UNSET`
or any required predecessor is not executable-qualified/current.

Current source qualifiers that are merely queued do not satisfy this gate.

Before start, review must explicitly confirm:

- [ ] exact benchmark generation identity sealed;
- [ ] all required precursor qualifications current;
- [ ] design/search domain frozen;
- [ ] material/process profile frozen;
- [ ] load/support profile frozen;
- [ ] measurement system can resolve the target effect;
- [ ] numeric target/non-inferiority thresholds frozen;
- [ ] search/resource budget frozen;
- [ ] replication/power/stopping plan frozen;
- [ ] safety/stored-energy profile current;
- [ ] evidence partitions and memory snapshot sealed;
- [ ] confirmatory analysis/decision rule frozen;
- [ ] candidate-selection freeze mechanism defined;
- [ ] raw-data/recomputation package defined.

## 19. Amendment rule

After campaign start:

```text
semantic protocol amendment
-> close/abort current generation
-> create a new generation identity
```

Display-only corrections must be recorded as explicitly non-semantic amendments.
Protocol history is append-only.

## 20. Metrology references

Implementation should remain compatible with established measurement-science
practice, including:

- NIST/SEMATECH measurement-process characterization for repeatability,
  reproducibility, stability, calibration, and uncertainty;
- JCGM Guide to the Expression of Uncertainty in Measurement (GUM);
- explicit conformity/decision rules that account for measurement uncertainty.

Exact adopted versions and methods must be content-bound when the campaign is
instantiated.
