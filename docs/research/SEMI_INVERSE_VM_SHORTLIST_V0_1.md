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
- each candidate's recipe settings, historical neighbours/distance, forward-model identity, evidence state, predicted metric intervals and candidate evidence digest.

Candidate payload and policy digests are canonicalized. These are content-integrity values, not issuer-authenticity proofs.

## Admission semantics

A candidate is **FEASIBLE** only when all of the following hold:
- its process snapshot matches the requested context;
- every required recipe knob is present exactly once, has the expected unit and is inside its authoritative hard range;
- its historical support receipt matches the declared manifest and policy, has enough unique neighbours and lies inside the declared support envelope;
- its forward model matches the required model artifact and its evidence state is QUALIFIED;
- the required target interval is present, unit-correct, narrow enough and entirely contained in the requested target band;
- every side-constraint interval is present, unit-correct and entirely contained in the corresponding allowed range.

Hard violations and out-of-support candidates are REJECTED. Missing support, insufficient support, unavailable forward evidence, missing forward metrics or incomplete authoritative constraint inventory are UNKNOWN. Neither UNKNOWN nor REJECTED candidates can enter top-K.

The target/side-constraint check is deliberately conservative: the entire declared forward interval must be inside the acceptable band. A future policy may support a different risk functional, but it must be separately versioned and benchmarked.

## Ranking and diversity

Feasible candidates receive a base score composed from normalized target error, interval width and normalized support distance. Greedy selection applies a diversity bonus based on minimum normalized recipe-space distance to already selected candidates.

The configured minimum diversity distance is a hard selection constraint. If fewer than K feasible candidates satisfy it, the shortlist is allowed to contain fewer than K entries. The system does not silently relax diversity, support or feasibility to fill the list.

Candidates are sorted and tie-broken deterministically by candidate ID. Input candidate order and policy constraint row order do not affect the report digest.

## Authored adversarial unit tests

The current Rust module includes tests for:
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
12. diversity spacing applied only after feasibility and permitted to produce a shorter-than-K shortlist.

Exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 reference workflow remain mandatory before marking these tests as passed.

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
