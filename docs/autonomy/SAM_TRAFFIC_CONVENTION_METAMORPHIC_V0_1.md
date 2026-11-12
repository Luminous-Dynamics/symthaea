# SAM Traffic Convention and Mirror-Metamorphic Benchmark v0.1

Status: research contract / synthetic conformance only  
Composes: SAM-TRAFFIC-CONVENTION-001 (#11123) and SAM Scene Evidence Contract v0.1

## 1. Why this exists

Road coordinates and traffic rules are related, but they are not the same thing.

The current vehicle model defines positive steering, lateral offset and curvature as leftward. The road geometry abstraction represents centerline segments and curvature. Neither alone tells a planner which side of a carriageway is the lawful ego lane, which adjacent lane is oncoming, what side is kerbside, or which passing-side rule applies.

This distinction matters when comparing systems across road jurisdictions. ISO 34502:2022 is a scenario-based ADS safety evaluation framework scoped to limited-access highways. SAE J3164_202301 describes normal operating maneuvers but explicitly assumes left-hand-drive vehicles and says right-hand-drive vehicles / infrastructure require adjustment. These are useful references, not a universal coverage guarantee.

References:
- ISO 34502:2022: https://www.iso.org/standard/78951.html
- SAE J3164_202301: https://saemobilus.sae.org/standards/j3164_202301-ontology-lexicon-automated-driving-system-ads-operated-vehicle-behaviors-maneuvers-routine-normal-operating-scenarios
- NIST Measurement Science for Automated Vehicles: https://www.nist.gov/programs-projects/measurement-science-automated-vehicles-0

## 2. Separation of concerns

Keep these axes orthogonal:

| Field | Meaning | Must not imply |
|---|---|---|
| Traffic side | Which side of the roadway traffic uses (left/right) | Steering wheel side |
| Steering-wheel position | Driver control layout (left/right/center/unknown) | Road-frame handedness |
| Travel frame | x forward, y left, z up in the declared ego/road frame | Jurisdiction or legal lane |
| Keep/kerbside sign | Legal and physical side in the travel frame | Passing rule |
| Passing/overtaking side | Rule-profile-specific allowed side | A global inference from traffic side |
| Lane role | Ego travel lane / oncoming / bus / cycle / shoulder / unknown | A lane role from pixel-left alone |
| Rule profile | Exact jurisdictional rule version and applicability | Universal law or automatic GPS inference |

The profile schema is in `SAM_TRAFFIC_CONVENTION_PROFILE_SCHEMA_V0_1.json`; the synthetic registry fixture is in `fixtures/SAM_TRAFFIC_CONVENTION_PROFILE_FIXTURE_V0_1.json`.

The scene envelope should carry a profile ID and exact version. A profile marked unverified or unknown may not authorize rule-sensitive lane selection, passing or overtaking.

## 3. Stable coordinate frame

The initial comparison uses a right-handed local road frame:
- x: ego travel direction / forward;
- y: left when looking along positive x;
- z: up;
- metres, seconds, radians.

A physical mirror about the longitudinal x-axis uses:
- x' = x;
- y' = -y;
- z' = z;
- heading' = -heading;
- signed curvature' = -curvature;
- lateral offset' = -lateral offset;
- lateral velocity' = -lateral velocity;
- yaw rate' = -yaw rate;
- steering angle/command' = -steering angle/command.

Longitudinal quantities, metric distances, speed magnitude, throttle and brake are invariant.

This transform does not itself alter the semantic value of steering-wheel position. A driver layout is a vehicle-configuration property, not a road-frame sign convention.

## 4. Paired synthetic benchmark

The canonical fixture contains one crossing-actor road scene and its mirrored twin:
- one profile declares left-hand traffic and right-hand-drive layout;
- the other declares right-hand traffic and left-hand-drive layout;
- the ego travel lane and oncoming lane swap lateral signs but preserve their semantic lane roles;
- keep/kerbside sign changes with traffic side;
- passing side is explicitly bound to a versioned rule profile;
- ego pose, velocity, yaw, curvature and steering command obey the mirror transformation.

A third control profile demonstrates that left-hand traffic can be paired with a left-hand-drive vehicle layout without changing road-side sign semantics. The profile is a synthetic test control, not an assertion about a particular jurisdiction.

The fixture is stored in `fixtures/SAM_TRAFFIC_CONVENTION_METAMORPHIC_V0_1.json`.

## 5. Conformance gates

The checker must verify:
1. unique profile IDs;
2. profile traffic-side signs are internally consistent;
3. scene profile IDs resolve to the profile registry;
4. map / traffic-rule profile mismatch remains a conflict;
5. paired scenes share exact scenario and pair identity;
6. longitudinal geometry and metric magnitudes are preserved;
7. lateral, yaw, curvature and steering signs are reflected;
8. lane semantic roles are preserved while lane-side signs reverse;
9. passing-side values are explicitly present and transform in the fixture;
10. steering-wheel position is independent of traffic-side and road-frame signs.

The JSON Schema is complemented by semantic checks: structural validation alone cannot establish that two scenes are true mirror pairs or that their identities match.

## 6. Negative controls

The fixture includes:
- left-hand geometry with a right-hand traffic profile;
- right-hand geometry with a left-hand traffic profile;
- mirror transform without steering sign inversion;
- steering-wheel-layout-only change;
- missing passing-rule identity;
- unknown traffic-side with a rule-sensitive decision request.

Required outcome is reject or unresolved for contradictory / missing jurisdictional evidence. A layout-only change should not alter road semantics.

## 7. Benchmark comparison protocol

Run paired tests under the same frozen:
- model/checkpoint and software build;
- scene geometry and actor policy;
- initial conditions and exogenous disturbance stream;
- action set and candidate-generation budget;
- sensor privilege, observation timing and compute budget;
- map version, coordinate frame and scoring code.

Report separate measures by traffic side and scenario family:
- collision / risky-interaction counts;
- road and lane-rule violations;
- route completion / progress;
- comfort and kinematic feasibility;
- unresolved / abstention rate;
- p50/p95/p99 runtime and deadline misses;
- transform/profile admission failures.

Do not aggregate these into one universal score. A system may improve one side while regressing on the other. Report worst-side and paired-difference results, confidence intervals under the correct episode/cluster sampling design, and the full denominator.

## 8. Prospective targets (not measured SAM results)

- 100% of schema/profile contradiction negatives reject or remain unresolved.
- 100% of paired synthetic transform vectors satisfy the exact declared sign and magnitude invariants.
- 0 silent inference of traffic side from steering-wheel position, GPS, map appearance or image pixel orientation.
- 0 decision comparisons across mismatched map, coordinate frame, rule profile, vehicle configuration or action-set identities.
- No side-specific performance claim without disjoint held-out scenario families and a frozen sampling design.

## 9. Boundaries

This benchmark is a synthetic metamorphic conformance test. It does not prove country-specific law, map accuracy, physical perception, legal compliance, real-world transfer, roadworthiness, collision-risk reduction, public-road authority or SAE L4/L5 capability.
