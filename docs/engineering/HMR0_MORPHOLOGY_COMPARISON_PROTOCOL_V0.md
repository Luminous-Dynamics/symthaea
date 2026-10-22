# HMR-0 Morphology Comparison Protocol v0.1

**Status:** benchmark protocol proposal. No robot profile has been measured by this document.

## 1. Research question

When does a full biped humanoid provide enough task-level value to justify its additional balance, power, fall, maintenance, integration and verification burdens?

HMR-0 must not assume the biped wins because the surrounding hospital was built for people. Compare three candidates under matched conditions:

- **M1 — fixed/bench manipulator:** fixed workstation and repeatable fixture tasks.
- **M2 — wheeled mobile manipulator with humanoid upper body:** rolling navigation plus human-height manipulation, avoiding biped balance as a prerequisite.
- **M3 — full biped humanoid:** stairs/step-over and foot-placement capability, if demonstrated; incurs extra locomotion and fall constraints.

A candidate should be ineligible for a task when it cannot satisfy the hard physical/task constraints. That is not a zero score and must not be hidden in a weighted average. Tasks outside a morphology's capability are reported as not applicable or infeasible with the reason.

## 2. Industry reference: compare relevant dimensions, not marketing demonstrations

Fraunhofer IPA's May 2026 humanoid benchmark uses six application-related dimensions: basic capabilities, complex capabilities, cleanroom/cleanability, functional safety, cybersecurity, and energy. These are sensible cross-platform dimensions for HMR-0. NIST's robotics performance framework adds an important measurement principle: compose task-driven evidence across perception, mobility, dexterity and safety instead of evaluating these pieces only in isolation.

HMR-0 adds healthcare-workflow dimensions that do not imply clinical capability: mock supply accountability, tool-attachment identity, contamination/cleaning evidence, interrupted-task recovery, human interventions, and traceable task handoff.

The July 2026 Nature study of a humanoid laparoscopic teleoperation framework is a comparator for a specialized surgical platform: it reports benchtop characterization, dry-lab user studies and in-vivo porcine work. It is not an apples-to-apples baseline for nonclinical hospital logistics, and should not be used to claim that general-purpose autonomous humanoids are clinically ready.

References:
- Fraunhofer IPA, *Benchmark for humanoid robots* (27 May 2026): https://www.ipa.fraunhofer.de/en/press-media/press_releases/benchmark-for-humanoid-robots.html
- NIST, *Performance Assessment Framework for Robotic Systems*: https://www.nist.gov/programs-projects/performance-assessment-framework-robotic-systems
- NIST, *Grasping, Manipulation, and Contact Safety Performance*: https://www.nist.gov/programs-projects/grasping-manipulation-and-contact-safety-performance-robotic-systems
- Liang et al., *In vivo feasibility study of humanoid robots in surgery*, Nature 657 (2026): https://doi.org/10.1038/s41586-026-10796-x

## 3. Matched task set

The first benchmark must use mock objects and nonclinical tasks only.

| Task ID | Task | Principal dimensions | Comparability rule |
|---|---|---|---|
| HMR-T01 | Transfer sealed mock supply tote over a level route | Completion, time, Wh, operator intervention, tote stability | Same tote mass, centre of mass, route, staging points and task contract |
| HMR-T02 | Read synthetic inventory labels across shelf heights | Read precision/recall, unreadable abstention, time, privacy violations | Same labels, lighting, occlusion, shelf geometry and information budget |
| HMR-T03 | Dock a passive exchange tray | Misalignment, successful retention, cycle time, recovery | Same tray geometry, tolerances, load and target fixture |
| HMR-T04 | Navigate a narrow corridor / doorway / mock elevator | Clearance margin, route deviation, stalls, interventions, Wh | Same swept-volume constraints, route map and dynamic obstacles |
| HMR-T05 | Recover from localization loss or changed world state | Correct stop/abort, stale-plan rejection, recovery time | Same injected fault, evidence revision and safe recovery contract |
| HMR-T06 | Change to a verified tool at a station | Tool-change success, identity verification, duration, wrong-tool rejection | Same tool identities, station and revision sequence |
| HMR-T07 | Threshold / step transition | Success/failure disposition, stability, recovery, energy | Run only where a test facility, physical risk plan and applicable safety controls allow; not required for early HMR-0 acceptance |
| HMR-T08 | Cleaning/contamination process audit on inert mock surfaces | Coverage of required steps, unhandled regions, time, false clean claims | No pathogen challenge and no sterility claim; physical cleaning efficacy is a separate validated study |

HMR-T01 through T06 can begin in simulation. HMR-T07 stays in simulation until separately reviewed physical test controls exist. T08 first tests process records and mock geometry; actual cleaning efficacy is never inferred from a simulated surface.

## 4. Measurements and evidence classes

For each episode bind:
- morphology/build revision and hardware bill-of-materials digest;
- tool ID, revision, geometry digest, attachment/calibration identity;
- task contract, object/fixture, environment and route digests;
- sensors/software/controller revisions and policy configuration;
- exact scenario ID, trial ID, and pre-registered faults;
- evidence class and measurement artifact digest.

Report per task and evidence class:
- completed tasks / attempts with confidence intervals;
- cycle time and energy per successful task;
- human interventions per 100 attempts;
- tool-change and fault-recovery success;
- measured grasp/fixture force and stability where instrumented;
- localization/pose error where relevant;
- collision, unsafe proximity, fall, lost-tool and stale-plan events;
- maintenance and downtime burden.

Evidence classes must remain separate: `synthetic_contract`, `simulation`, `bench_measurement`, `hardware_in_the_loop`, and `controlled_field_observation`. Synthetic data validates the evaluation harness only. Do not promote a result to a higher evidence class because a receipt has a valid hash.

## 5. Hard gates — never average away

Any of the following blocks the episode regardless of task completion, energy or time:
- stale/unknown robot, tool, map or calibration identity;
- invalid or missing task permission and operator authorization;
- tool not detected/verified at the attachment;
- unresolved localisation, state freshness or recovery constraint;
- missing required interlock or independent stop boundary;
- unintended contact or exceeded task/fixture force constraint;
- a physical-actuation request when the configured gate is simulation-only;
- malformed or unknown event/manifest fields silently accepted.

A hard-gate violation is a separate safety/process result, not a penalty term.

## 6. Fair comparison and ranking policy

- Freeze the task definitions, route, object/fixture, trial protocol, score code, and build revisions before comparing candidates.
- Compare a candidate only within the same task contract, relevant fixture, evidence class, and resource conditions.
- Never treat missing measurements as zero or as success. Mark the candidate `unranked_missing_evidence`.
- Never rank an unqualified or hard-gate-failing candidate on performance alone.
- Report the Pareto frontier for duration, energy, operator intervention, maintenance burden and task success. Do not create one aggregate score across incompatible tasks.
- Publish task-level deltas with confidence intervals and retain all failures, aborted trials, exclusions and protocol deviations.
- Include established task-specific equipment as baseline where applicable, not only the other HMR-0 morphologies.
- A superiority claim requires a pre-registered matched experiment and independent evaluation; synthetic fixture comparisons cannot establish real-world superiority.

## 7. Provisional targets

The only immediate quantitative targets are software-harness integrity:
- 100% of malformed/missing measurement fields are rejected or marked unranked rather than imputed.
- 100% of stale build/tool/scenario identity mutations invalidate comparability.
- 0 hard-gate failures are hidden by aggregate scores.
- 0 performance rankings are emitted for candidates with only synthetic fixture data.
- 100% of accepted benchmark records bind trial identity, task identity, morphology/build revision, tool revision and measurement artifacts.

Do not pick a universal “humanoid beats wheeled robot by X%” target before the first defensible measurements exist. Pre-register the sample size and confidence bounds once the task, expected effect size, variability and safety constraints are known. For rare high-severity failures, mean completion time is not an adequate risk metric.

## 8. Current state and claim ceiling

HMR-0 currently has candidate morphologies and parameterized tool concepts, but no measured robot performance profiles, no physical robot, no measured task trials, no verified cleanability, and no safety case. This protocol defines a fair path to comparison. It does not establish which morphology is best, and it does not authorize physical actuation, patient contact or clinical functions.
