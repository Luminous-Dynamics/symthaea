# Evidence-qualified traffic-rule compliance v0.1

**Status:** offline rule/trajectory evaluator implemented; no perception or closed-loop benchmark results  
**Code:** `symthaea_vehicle::rule_compliance`  
**External benchmark:** TrafficSignBench (2026)  
**Decision-measurement reference:** NIST Measurement Science for Automated Vehicles (2026)

## Motivation

Aggregate driving score, destination rate and collision count do not establish that a planner obeyed an applicable traffic rule. TrafficSignBench makes this concrete: its held-out planner results report 2.9–9.0% Sign-Compliant Destination for several evaluated planners, while rule-supervised fine-tuning raises one PlanT-2 configuration from 5.9% to 72.3%. The benchmark paper describes 34 sign types, 29 scenario types and 5,800 test scenes; its public repository currently presents 25 sign-profile IDs and a 5,800-episode held-out results table. Preserve the distinction between paper taxonomy and released profile registry.

Reference:
- Paper: https://arxiv.org/abs/2609.38463
- Public code/data: https://github.com/emb-ai/traffic-sign-bench

## Current executable scope

The v0.1 evaluator supports three rule primitives:
- **SpeedLimit:** compare sampled speed against a declared maximum plus an explicit measurement tolerance.
- **MandatoryDirection:** compare wrapped trajectory heading against a declared heading and angular tolerance.
- **StopBeforeLine:** require low-speed samples before line crossing for a minimum observed dwell duration, with bounded sample gaps and plausible route-progress movement during the stationary sequence.

Other rule families can be retained as `Unsupported` and yield `Indeterminate`. An empty unsupported-family name is malformed and also remains indeterminate. They must not be silently dropped or counted as compliant. Initial unsupported families include yield/right-of-way, crosswalk priority, signal phase, no-entry, prohibited turns and other route-direction signs.

## Evidence input semantics

Every rule includes:
- stable rule ID;
- active route-progress interval;
- typed rule parameters and units;
- evidence ID;
- source ID and source artifact digest when the source is available or conflicted;
- status: `Uncontradicted`, `Conflicted` or `Unavailable`.

An `Unavailable` source does not require a fabricated source ID or digest; its unavailable state is explicit.

**Uncontradicted does not mean authenticated or correct.** It only means the caller supplied one rule assertion without a declared conflict. The evaluator does not perform image perception, traffic-sign recognition, map matching, jurisdictional law interpretation, provenance authentication or source truth adjudication.

Conflicted or unavailable evidence yields `Indeterminate`, not `Compliant`. Unsupported rule semantics also yield `Indeterminate`.

## Episode outcomes

- `Compliant`: every sampled condition inside the applicable interval satisfies the implemented rule predicate.
- `Violation`: the sampled trajectory provides a counterexample to the implemented rule predicate.
- `Indeterminate`: evidence is missing/conflicted/malformed, a rule is malformed, or the trajectory cannot support the predicate.
- `NotEncountered`: the rule's active route interval was not entered in the supplied trajectory.
- `Pending`: the episode entered a stop-control zone but ended before crossing the stop line; no compliance or violation is claimed yet.

Non-finite numeric input is rejected explicitly because it cannot be represented faithfully in the JSON report. Finite but malformed trajectories (e.g. non-monotonic time or route progress) are retained as indeterminate findings. Speed-limit and mandatory-direction evaluations additionally require the trace to bracket the complete declared active route-progress interval, at least two samples inside it, and no intersecting segment whose time gap exceeds the rule's maximum. A trace that overlaps only part of the rule zone or skips the full zone between samples is `Indeterminate`, not `Compliant` or `NotEncountered`. Route progress must be unwrapped and monotonic; a loop-wrap must be normalized by the caller.

## Stop-line sampling conditions

A stop is not inferred from one low-speed point or from widely separated samples. A crossing result also requires at least two samples inside the declared approach zone; skipped spatial coverage is indeterminate even when the time gap is short. A qualifying stopped interval requires:
- each consecutive low-speed sample gap no larger than the declared maximum;
- route progress remains before the line within the declared tolerance;
- sampled progress displacement remains compatible with the speed threshold and tolerance;
- the trace begins at or before the declared approach-zone start;
- total observed sampled dwell between qualifying low-speed samples reaches the required duration before line crossing;
- the final low-speed-sample-to-crossing gap is within the declared maximum but is not counted as stopped dwell.

A large gap or speed/route-progress inconsistency that prevents determining the stop sequence yields `Indeterminate`, not `Violation`. A trace that starts after the approach-zone boundary is also indeterminate because the earlier stop history is missing. A trace with fewer than two in-zone samples, or one that skips the whole approach segment between samples, cannot support a violation finding.

This is a discrete trajectory check, not proof of continuous physical standstill. The maximum sample gap and position tolerance are part of the rule profile and must be calibrated to the data source.

## Report integrity

The report retains:
- exact rule requirements and evidence records;
- exact trajectory samples;
- per-rule findings and violation sample indexes;
- counts for compliant, violation, indeterminate, not-encountered and pending episodes;
- a BLAKE3 report fingerprint.

The validator re-runs the rule evaluator from the retained inputs and reconstructs all findings and counts before checking the digest. BLAKE3 is an integrity fingerprint, not a signature, authentication, proof of lawful rule interpretation or proof that a trajectory came from an independent simulator.

## Benchmark plan

Reproduce TrafficSignBench on its published closed-loop protocol before claiming parity or superiority. Keep three layers separate:
1. **Rule recognition/evidence:** sign and active-zone detection, source disagreement, confidence/calibration and localization error.
2. **Rule-conditioned planning:** route intent, selected maneuver, feasibility and explicit rule-constraint satisfaction.
3. **Trajectory execution:** observed rule compliance, destination completion, collision/contact, progress and runtime.

Compare the published non-rule-aware planner baselines, rule-aware variants and privileged rule-access oracle under the same scenario manifests. Report both episode-weighted and scenario-family macro-averaged compliance. Do not treat the privileged oracle as a deployable perception/planning model.

Proposed stretch target after exact reproduction:
- at least 90% macro-averaged Sign-Compliant Destination on a frozen held-out profile set;
- at least 80% destination completion;
- no deterioration in the separately reported collision/contact metric versus the strongest reproduced baseline;
- 100% of unsupported/conflicted/unavailable rule cases surfaced as indeterminate rather than being silently omitted.

These are proposed targets, not current results. They must be revised if the reproduced dataset, task definitions or baseline results differ from the paper's public report.

## Next steps / ownership

1. Add image/map rule observations with calibrated source identity and conflict handling; keep this separate from the current offline evaluator.
2. Add right-of-way, crosswalk and signal-phase semantics only when their actor/conflict-zone observations are available.
3. Connect Symtropy #3387 execution receipts to rule trajectory reports.
4. Add a black-box harness for evaluating external planners without source modification.
5. Validate route progress, stop dwell and rule predicates against independent trajectory sources and a physical proving-ground protocol before stronger claims.
6. Add jurisdiction-specific legal profiles only with sourced rule semantics and versioned applicability.

## Claim ceiling

The present module validates supplied typed rule claims against supplied route-progress trajectory samples. It does not recognize signs, establish the actual governing law, prove the map is correct, model interactions with other road users, establish safety, or qualify a public-road vehicle.
