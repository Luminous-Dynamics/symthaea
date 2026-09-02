# Affective Emergence v0.2 — Minimal Exploratory Scenario Battery

Status: **normative design-only / blocked on Native Interoception v0.1 qualification**

This contract defines the smallest initial exploratory scenario battery intended to discriminate the finite E00–E11 candidate set from `V02_MINIMAL_EXPLORATORY_CANDIDATE_SET.md` while preserving causal, information, history, weighting, aggregation, and temporal controls.

The objective is not broad ecological realism. It is high information gain per deterministic scenario.

## 1. Principle

Every scenario family exists because it disambiguates at least one prospectively declared competing explanation.

Do not add scenarios merely for variety. Do not remove a scenario because it produces an inconvenient equivalence or null result.

The later machine-readable `ExploratoryScenarioBatteryManifest` freezes exact parameter values, cut points, content hashes, generator/source identities, and candidate discrimination obligations before exploratory outputs are inspected.

## 2. Battery structure

The first exploratory battery contains twelve required scenario families, X00–X11.

A family may contain a matched pair, crossed pair, or small deterministic tuple. The manifest must minimize duplicated content while preserving the required contrasts.

The preferred implementation target is **no more than 24 primary scenario arms** before malicious fixtures and forecast-policy sensitivity diagnostics. If a required discriminator cannot be satisfied within that budget, document why and supersede the battery identity before data inspection.

## 3. Required scenario families

### X00 — neutral fixed-point control

Construct a stable native state inside all preferred bands with zero drive and no intervention.

Requirements:

- current viability burden neutral;
- realized change neutral after stabilization;
- drive baseline zero;
- drive-persistence forecast self-consistent;
- no projected breach within horizon;
- candidate availability states predictable.

Primary uses:

- E00 null floor;
- neutrality gates for E01/E03/E04/E05/E06/E08;
- typed no-breach behavior for E07;
- basic forecast self-consistency;
- canary/preprocessing/evaluator controls.

### X01 — same current burden, different current drive

Create a matched pair with equal complete current native state/configuration at the comparison cut point but different current drive vectors/magnitudes supplied to the forecast/evaluation boundary.

The drive difference must be legal prefix information and must not yet have changed the matched current state.

Primary discriminations:

- E02 vs E01;
- E04/E05/E07/E08 vs E01;
- forecast simulation vs current-state-only burden.

Causal interpretation: diagnostic/prefix forecast sensitivity unless the native execution contrast separately establishes a realized total effect.

### X02 — same drive magnitude, different viability consequence

Create a matched pair with equal Euclidean drive magnitude but different current viability consequence because of different prospectively declared channel state/margin/importance geometry.

Prefer to hold drive direction comparable while changing distance to preferred/viable boundaries through valid initial/native-state construction.

Primary discriminations:

- E01 vs E02;
- E07/E08 vs E02;
- nuisance load magnitude vs regulatory consequence.

### X03 — crossed realized-change / forecast-residual signs

Construct two deterministic cases:

A. worsening but better than predicted:

`E03 < 0` while `E04 > 0` under the frozen sign conventions.

B. improving but worse than predicted:

`E03 > 0` while `E04 < 0`.

Exact inequalities/tolerances are frozen in the machine manifest.

Primary discrimination:

- E04 vs E03.

This family is an anti-collapse test; inability to realize the crossed cases is a design finding, not permission to redefine either candidate.

### X04 — temporal forecast decomposition tuple

X04 contains two small deterministic subcases so E05 is separately identifiable from both E04 and E06 without creating a new scenario family.

#### X04-A — boundary-only rolling turnover

Construct a transition where the overlapping absolute future support remains self-consistent while the dropped/entering horizon terms differ.

Required expectation:

- E05 aligned future revision neutral/equivalent;
- E06 rolling-horizon change non-neutral.

Primary discrimination:

- E05 vs E06.

#### X04-B — outlook revision with neutral one-step residual

Construct a transition where the prior one-step forecast for the realized current point is accurate/equivalent, while information legitimately available at the current cut point changes the forecast for later **shared absolute future points**.

Required expectation:

- E04 one-step forecast residual neutral/equivalent;
- E05 aligned overlapping-future revision non-neutral;
- E06 may move but is not the identifying quantity.

Primary discrimination:

- E05 vs E04.

The current-cut-point forecast change may arise from a legally observed current-state/current-drive change; it may not use unseen future protocol information.

A failure of the expected neutrality relations indicates temporal alignment/fixture failure rather than an affect result.

### X05 — same current burden, different breach latency

Construct a matched pair with equal E01 current viability burden but different projected first-breach latency under `ObservedDrivePersistence`.

Use a legal difference in current drive, channel state distribution, or native recovery configuration while preserving the matched current aggregate burden as declared by the causal contrast.

Include a stricter subcase when feasible with matched current drive magnitude but different projected breach latency due to state geometry/distribution.

Primary discriminations:

- E07 vs E01;
- E07 vs E02 in the strict matched-drive subcase;
- urgency vs current burden.

### X06 — similar projected cumulative exposure, different urgency profile

Construct a pair whose E08 projected discounted cumulative exposure is equal/equivalent within a frozen tolerance while first-breach latency or peak-threat timing differs materially.

Possible construction: concentrated near-term threat vs diffuse longer-horizon burden with matched cumulative exposure.

Primary discrimination:

- E08 vs E07.

This prevents cumulative burden and urgency from being treated as synonyms.

### X07 — precision/confidence orthogonalization

Construct a matched pair with identical raw deviations, preferred/viable geometry, importance weights, current drive, and native dynamics configuration, but different valid precision/confidence values.

Required expectations:

- E01 viability-only burden unchanged;
- E09 confidence changes;
- E10 legacy precision×importance burden changes when the deviated channels carry changed precision.

Primary discriminations:

- E09 vs E01;
- E10 vs E01.

This family directly tests the semantic concern tracked separately for the legacy weighting hypothesis.

### X08 — healthy-channel denominator dilution

Hold the deviated channel state and its own viability weight fixed while changing only one or more healthy in-band channel weights in a legal state fixture.

Required interpretation:

- raw deviated-channel evidence unchanged;
- A3 weighted mean may change because its denominator changes;
- this change cannot be described as intrinsic recovery of the deviated channel.

Primary uses:

- aggregation/denominator audit for E01/E10;
- malicious M11 control;
- guard against interpreting healthy-channel reweighting as improved regulatory severity.

This is primarily a measurement-semantics discriminator rather than a candidate-ranking world.

### X09 — matched current state with controlled prior histories

X09 contains two matched-history subcases sharing the same complete current native state/configuration and same current drive at cut point `t`.

#### X09-A — immediate-change discriminator

Construct two prior paths that converge to the same current state/current drive but have different immediately preceding realized burden.

Required expectations:

- E01 equal;
- E02 equal;
- E03 different;
- H0 forecast candidates equal when all of their current allowed inputs are equal.

Primary discriminations:

- E03 vs E01;
- E03 vs E02.

#### X09-B — deeper-history discriminator with matched immediate change

Construct two prior paths that converge to the same current state/current drive **and** have equal/equivalent immediately preceding burden so E03 is equal, while the earlier trailing-16 burden history differs.

Required expectations:

- E01 equal;
- E02 equal;
- E03 equal;
- E11 different by construction.

Primary discriminations:

- E11 vs E01;
- E11 vs E03.

For both subcases, subsequent native execution from the matched current state/configuration under identical future inputs must be exactly equal.

A difference in restarted native future execution is a native-state sufficiency defect, not evidence of hidden mood.

### X10 — exact-prefix divergent-future twins

Create two source executions/protocol variants whose allowed evidence is byte-identical through cut point `t` and whose unseen suffixes diverge strongly afterward.

Required expectations through `t`:

- identical prefix digest;
- identical preprocessing state/parameters;
- identical E00–E11 candidate payloads/availability where defined;
- outer full-trace provenance may differ.

Primary use:

- prefix-causality and suffix-provenance firewall;
- malicious M01/M02/M12 controls.

This family is integrity-blocking, not candidate-discrimination evidence.

### X11 — forecast-policy disagreement diagnostic

Construct at least one prefix where:

- `ObservedDrivePersistence`;
- `NativeZeroInputRecovery`; and
- `KinematicVelocity`

produce materially different prospective trajectories under their own valid semantics.

Primary exploratory candidates remain computed using `ObservedDrivePersistence` only.

The alternate policies are sensitivity diagnostics used to report policy dependence of E04/E05/E06/E07/E08-like quantities without multiplying the primary candidate set.

Primary use:

- forecast-policy sensitivity;
- detect whether an apparent candidate advantage exists only under one simple forecast assumption.

Oracle future schedules may be added only as separately marked upper-bound diagnostics.

## 4. Reuse and arm-count minimization

A single concrete scenario arm may satisfy multiple families only when the machine manifest explicitly records each obligation and the relevant matched/causal constraints are simultaneously valid.

Do not reuse an arm if doing so introduces conditioning that changes the intended causal estimand.

Prefer paired construction over broad parameter grids.

The initial battery should optimize for orthogonality, deterministic reproducibility, low arm count, and clear failure interpretation—not visual richness or anthropomorphic plausibility.

## 5. Required candidate-coverage matrix

Before exploratory execution, the scenario manifest must prove coverage for at least:

| Candidate/contrast | Required families |
| --- | --- |
| E00 neutrality/null floor | X00 |
| E01 vs E02 | X01, X02 |
| E03 vs E01 | X09-A |
| E03 vs E02 | X09-A |
| E04 vs E03 | X03 |
| E04 vs E02 | X01/X03 with matched nuisance control |
| E05 vs E04 | X04-B |
| E05 vs E06 | X04-A |
| E07 vs E01 | X05 |
| E07 vs E02 | X05 strict matched-drive subcase |
| E08 vs E01 | X01/X05 with equal current burden and divergent prospective exposure |
| E08 vs E02 | X02 |
| E08 vs E07 | X06 |
| E09 vs E01 | X07 |
| E10 vs E01 | X07 |
| E11 vs E01 | X09-B |
| E11 vs E03 | X09-B |
| prefix causality | X10 |
| primary forecast-policy sensitivity | X11 |
| aggregation denominator interpretation | X08 |

Every non-null E candidate must also have at least one valid non-neutral case somewhere in the locked battery so comparison with E00 is meaningful.

If coverage cannot be proven, the battery is incomplete even if the raw scenario count is large.

## 6. Cut-point discipline

Each family declares cut points/windows prospectively.

Do not search a trajectory after execution for the most dramatic cut point.

Allowed cut-point definitions include:

- fixed step index from scenario start;
- fixed offset before/after a prospectively declared intervention;
- first occurrence of a mechanical condition defined entirely without candidate output or semantic arm identity.

Candidate-dependent cut-point selection is forbidden for primary exploratory comparison.

## 7. No fitted preprocessing

The minimal E00–E11 set uses no fitted preprocessing.

Scenario values therefore cannot influence candidate scaling/threshold fitting.

Structural fixed transformations and typed unit conversions remain part of candidate identity.

## 8. Discovery vs later confirmation

X00–X11 define the **exploratory design vocabulary**, not the future confirmatory holdout itself.

After exploratory analysis:

- candidate reduction/equivalence classes are frozen;
- a new confirmatory candidate/baseline set is selected prospectively;
- confirmatory scenarios must be newly materialized/content-hash audited and not near-duplicate copies of exploratory cases under the locked overlap policy;
- the confirmatory cohort receives a new root/analysis identity.

Do not reuse the exact exploratory arms as untouched confirmation.

## 9. Machine-readable battery manifest

A future `ExploratoryScenarioBatteryManifest` should bind:

- schema/version;
- design-contract-registry digest;
- minimal candidate-set manifest digest;
- ordered X00–X11 family/subcase definitions;
- exact concrete scenario/arm digests;
- matched-pair/group identities;
- causal-contrast manifest digests where mechanistic language is intended;
- prospective cut points/windows;
- required candidate discrimination obligations;
- expected invariant/equality relations where known;
- generator/source identities and seeds if stochastic generation is later introduced;
- total primary arm count;
- canonical SHA-256.

Validation rejects:

- missing required family/subcase;
- unregistered extra primary family;
- missing required discriminator coverage;
- candidate-dependent cut point;
- semantic-arm-dependent construction in blinded primary artifacts;
- future-dependent matched-group membership;
- exact/content-near duplicate violation across discovery/calibration/confirmatory cohorts.

## 10. Claim boundary

This battery is a deterministic discrimination instrument. It is designed to reveal whether candidate behavior can be explained by current state, load, actual change, prediction error, future revision, rolling-window turnover, urgency, cumulative prospective exposure, confidence, legacy weighting, denominator effects, recent external history, or forecast-policy assumptions.

It does not model the full richness of biological emotion and cannot establish subjective affect, feeling, mood, suffering, sentience, or consciousness.