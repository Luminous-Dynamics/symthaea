# SEMI-INVERSE-VM-001 — evidence-qualified inverse VM shortlist v0.1

**Issue:** #13699  
**Implementation:** `symthaea-semiconductor::benchmark::inverse_vm_shortlist`  
**Status:** synthetic Rust implementation with authored unit tests; exact-head validation pending.

## Research comparator

Lin, C.-Y.; Barros, A.D.O.; Emon, S.H.; Tseng, T.-L., “Feasibility-Aware Inverse Virtual Metrology for Top-K Recipe Recommendation in Semiconductor Process Correction,” IEEE Transactions on Automation Science and Engineering, Volume 23, pp. 13445–13468, published 20 July 2026. DOI: 10.1109/TASE.2026.3715101.

The paper's public abstract reports a single-process EPI cohort with 2,140 historical correction events. On its time-based split:
- reported Success@5: 0.742;
- reported Feasible@5: 0.946;
- strongest reported overall baseline: Success@5 0.647, Feasible@5 0.901.

The described pipeline uses target-conditioned inverse candidate generation, historical-support estimation, forward verification and diversity-aware ranking. The authors note prospective plant validation and cross-process generalization remain future work.

Primary record: https://ieeexplore.ieee.org/document/11614528/  
DOI: https://doi.org/10.1109/TASE.2026.3715101

We have not obtained the authors' exact 2,140-event cohort, metric code or candidate-generation implementation in this workstream. The headline numbers are external comparison targets, not local results. Do not claim reproduction or superiority until data/split/metric parity is verified.

## Paradigm shift: from inverse prediction to admissible decision shortlist

A VM point predictor estimates what may happen. An inverse model proposes what could be changed. Neither one independently establishes whether the proposal is supported or acceptable.

The v0.1 screen implements:

`process snapshot -> candidate proposal -> recipe hard bounds -> historical support -> forward interval verification -> side constraints -> feasible-only top-K diversity selection`

Four things remain separate:
1. predicted process outcome;
2. inverse-generated candidate proposal;
3. evidence-supported and constraint-feasible offline shortlist;
4. engineer-authorized physical experiment or process action.

The module consumes proposed candidates. It does not generate a recipe, claim a causal correction effect, authorize an experiment or write to equipment.

## Typed input contract

The input binds:
- current process snapshot digest;
- versioned policy and constraint-source digest;
- allowed recipe knobs, units and hard numeric ranges;
- target measurand, unit, target value and acceptable interval;
- side-constraint measurands and allowed intervals;
- required forward-model artifact digest;
- historical support manifest and support-policy digests;
- minimum neighbour count and maximum normalized support distance;
- uncertainty-width limit, score weights, diversity weight, minimum recipe diversity and top-K;
- each candidate's recipe settings, historical neighbours/distance, forward-model identity, evidence state, predicted metric intervals, content-addressed forward-evaluation receipt and candidate evidence digest.

Candidate payload and policy digests are canonicalized. Forward-evaluation receipts additionally bind evaluator ID, model artifact and model-qualification report digest, exact process-snapshot digest, canonical recipe-vector digest, canonical output-metric digest, evidence state, evaluation time, availability time, clock-domain ID and clock-mapping digest. The decision's Unix timestamp is tied to the same declared clock mapping. A receipt available after the decision frontier is UNKNOWN; a context, recipe, model, output, status, digest or clock mismatch is REJECTED.

All these digests provide content integrity and replayable linkage only. The current report explicitly sets `forward_evaluation_receipt_authenticity_proven=false`: there is no trusted issuer signature or external fab authority registry in this version. A self-consistent receipt is not proof that the named evaluator actually ran.

## Admission semantics

A candidate is **FEASIBLE** only when all of the following hold:
- its process snapshot matches the requested context;
- every required recipe knob is present exactly once, has the expected unit and is inside its authoritative hard range;
- its historical support receipt matches the declared manifest and policy, has enough unique neighbours and lies inside the declared support envelope;
- its forward model matches the required model artifact and its evidence state is QUALIFIED;
- the required target interval is present, unit-correct, narrow enough and entirely contained in the requested target band;
- every side-constraint interval is present, unit-correct and entirely contained in the corresponding allowed range.

Hard violations and out-of-support candidates are REJECTED. Missing support, insufficient support, missing forward-evaluation receipt, late/unavailable forward receipt, unavailable forward evidence, missing forward metrics or incomplete authoritative constraint inventory are UNKNOWN. Neither UNKNOWN nor REJECTED candidates can enter top-K.

The target/side-constraint check is deliberately conservative: the entire declared forward interval must be inside the acceptable band. A future policy may support a different risk functional, but it must be separately versioned and benchmarked.

## Ranking and diversity

Feasible candidates receive a base score composed from normalized target error, interval width and normalized support distance. Greedy selection applies a diversity bonus based on minimum normalized recipe-space distance to already selected candidates.

The configured minimum diversity distance is a hard selection constraint. If fewer than K feasible candidates satisfy it, the shortlist is allowed to contain fewer than K entries. The system does not silently relax diversity, support or feasibility to fill the list.

Candidates are sorted and tie-broken deterministically by candidate ID. Input candidate order and policy constraint row order do not affect the report digest.

## Authored adversarial unit tests

The current Rust module includes 22 tests for:
1. deterministic known-answer ranking and full replay;
2. recipe hard-bound rejection;
3. historical-support envelope rejection despite an attractive forward prediction;
4. forward interval crossing the target band;
5. incomplete constraint inventory returning UNKNOWN and no shortlist;
6. missing support receipt returning UNKNOWN;
7. UNKNOWN forward evidence never being admitted;
8. wrong process-context digest rejection;
9. duplicate candidate ID and duplicate recipe knob rejection;
10. policy/candidate input-order-invariant report digest;
11. shortlist receipt tamper detection;
12. diversity spacing applied only after feasibility and permitted to produce a shorter-than-K shortlist;
16. missing, late, digest-tampered, snapshot/recipe/model/output-mismatched forward-evaluation receipts are not admitted;
17. a qualification report digest is required for a forward receipt marked QUALIFIED;
18. decision frontier and clock mapping are sealed in the output report;
13. malformed/blank historical-support neighbour IDs are rejected;
14. minimum recipe-diversity spacing is enforced as a hard top-K selection constraint;
15. independent exhaustive enumeration verifies the production greedy diversity order and per-rank selected scores;
16. missing forward-evaluation receipt returns UNKNOWN;
17. receipt after decision frontier returns UNKNOWN;
18. recipe and prediction-interval mismatch are rejected;
19. snapshot mismatch and receipt-digest tampering are rejected;
20. QUALIFIED status requires a model-qualification report digest;
21. clock-domain or mapping mismatch is rejected;
22. decision frontier and clock mapping are sealed into the report digest.

An additional independent exhaustive top-K oracle separately recomputes normalized recipe distances, enumerates all valid sequences under the minimum-diversity rule, and compares the lexicographically optimal greedy path and per-rank scores with the production result. Exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 reference workflow remain mandatory before marking these tests as passed.

## Scenario robustness audit

The companion module `inverse_vm_shortlist_stability` replays the same candidate inventory under explicit stress scenarios such as sensor dropout, reference drift, model-interval widening, support sparsification and delayed evidence.

Every scenario must keep the same:
- process snapshot and decision time;
- clock domain and mapping;
- target, hard recipe/side constraints, support policy and ranking policy;
- candidate IDs and exact recipe vectors.

A scenario may vary the forward prediction intervals, model evidence status/receipt availability, support neighbour set/distance or evidence arrival time, but each changed claim must be resealed with its own forward/support receipts.

The report gives per-candidate:
- count and fraction of scenarios remaining feasible;
- count and fraction selected into top-K;
- best, worst and median selected rank;
- rank spread and whether it meets the declared stability limit;
- per-scenario state and selected rank.

The fractions are **descriptive stress-scenario frequencies, not probabilities or confidence levels** unless the scenario generator is separately calibrated and justified. The evaluator never changes a recipe, relaxes constraints or infers probability from an unweighted scenario list.

## Benchmark programme

### Faithful paper reproduction

Obtain and verify the actual 2,140-event cohort or an author-authorized data path. Reconstruct the exact time split, candidate-generation process, target definition, recipe variables, constraint handling and metric semantics. Reproduce the paper's own baselines before claiming comparison.

Required metrics:
- Success@1/@3/@5;
- Feasible@1/@3/@5;
- unsupported-candidate rate;
- side-constraint preservation;
- candidate diversity;
- target deviation and paired bootstrap intervals.

### Evidence assurance overlays

Run the same candidate pool through:
- paper-faithful reference method;
- support-only gate;
- forward-interval-only gate;
- hard recipe constraints + support gate;
- full evidence-qualified shortlist.

Report feasible-candidate yield, unknown/abstention rate, empty shortlist, hard-constraint rejection, support rejection, forward interval rejection, diversity@K and reconstruction success.

### Robustness and transfer

Where data allow:
- strictly chronological holdout;
- leave-tool-out and leave-chamber-out;
- maintenance/regime change split;
- recipe-family holdout;
- support sparsification;
- forward-model interval undercoverage;
- stale/revoked calibration and model artifact;
- incomplete side-constraint source;
- hidden/unseen knob or unit mismatch.

A single EPI cohort cannot establish cross-fab or cross-process generalization. Do not combine normalized error metrics across physical targets.

## Proposed targets (hypotheses, not achievements)

After the paper protocol is faithfully reproduced:
- Success@5 above the reported 0.742, with paired confidence intervals;
- Feasible@5 above 0.946 on the same metric definition and time split;
- zero emitted candidates outside declared recipe/support hard bounds;
- zero emitted candidates with forward intervals crossing hard side constraints;
- 100% independent shortlist replay and tamper detection;
- no reduction in success/feasibility hidden by returning an empty shortlist;
- report the abstention/empty-shortlist rate alongside every top-K score.

The software zero-violation target is not a claim of physical process safety; it only applies to the versioned constraints and intervals supplied to the evaluator.

## Claim ceiling

This is a synthetic offline shortlist evaluator. It is not a reproduction of the 2026 paper, a real EPI recipe model, a causal effect estimate, a physical process window qualification, a prospective fab result, or an authorization to execute recipe changes. No recipe write, equipment command, APC/R2R action, wafer disposition or experiment approval is implemented.
