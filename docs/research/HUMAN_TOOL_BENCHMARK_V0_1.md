# Human + Symthaea Repair-Workbench Benchmark v0.1

Status: synthetic contract/evaluator slice. All fixture values are invented to exercise counting and gate semantics. No physical repair observations, participant observations, product safety qualification, or performance effect is claimed.

## Run locally

From the repository root:

    cargo test --manifest-path tools/human-tool-bench/Cargo.toml --offline
    cargo run --manifest-path tools/human-tool-bench/Cargo.toml --offline -- fixtures/human_tool_bench_v0_1.json

The tool is an isolated Cargo workspace so it does not become an accidental member of Symthaea's large main workspace. The core audit reads local JSON only. The build still needs the declared Rust crates available in the local Cargo cache for an offline run.

## What the contract prevents

- Hiding failed, abandoned, referred, or safety-stopped attempts by counting only successful repairs.
- Treating a completed task or a model hypothesis as verified restoration.
- Recording missing cost as zero.
- Calling a repair durable without a verified restoration plus positive observed-use exposure.
- Treating stale, missing, contradictory, or invalid evidence as nominal (future corpus extension).
- Allowing a diagnostic model to produce a safe-to-operate verdict or override an independent safety authority.
- Reusing an episode ID or misclassifying a record that never started as a failed attempt.

The program reports outcome counts for A0/A1/A2 separately and median/P90 elapsed time only conditional on verified restorations. That conditional distribution is *not* a complete treatment-effect estimate: confirmatory work must analyze all started attempts, use a preregistered estimand, and explicitly account for non-restoration, competing outcomes, censoring and repeated task exposure.

## Experiment arms

- A0: existing human practice, with access to its normal manuals and the common instrument set.
- A1: the same resources plus structured open work instructions.
- A2: A1 plus local Symthaea advisory, limited to evidence-linked hypotheses, non-actuating next-measurement proposals, contradictions, explicit unknowns and abstention.

An actual human-participant study needs task blinding, randomized/counterbalanced assignment, controls for learning/order effects, an independent adjudicator, consent/data minimization, and any required ethics review. A pilot should estimate task feasibility and variance, not prove superiority. The primary comparison and its smallest practically important effect, sample size, exclusions, safety/function gates and analysis plan must be frozen before confirmatory data are collected.

## Intended repository composition

- Symthaea #7819 / #7843: synthetic fault semantics and advisory diagnostics; verify the exact code state and upstream kernel before wiring this evaluator to their output.
- Symtropy #1805, #1908, #1910: repair task, verified restoration and architecture experiment owners.
- Mycelix #5156 / #4689: design/unit/parts evidence and eventual reproducible physical manufacturing cell.
- Sol-Atlas: regional supplier/tooling/logistics constraints once the experiment is grounded.

This crate owns only the matched-experiment contract audit and outcome-accounting reference. It does not duplicate the fault model, decision-value kernel, calibration system, repair task graph, manufacturing ledger, or actuation path.

## Research references

- Open Repair Data Standard v0.3: https://standard.openrepair.org/standard.html (focused on small electricals/consumer electronics; community repair data are not automatically representative of a population's failures).
- NIST metrological traceability policy: https://www.nist.gov/calibrations/traceability (traceability does not by itself establish fitness for purpose).
- Open Source Hardware Association definition: https://oshwa.org/resources/open-source-hardware-definition/
- Open Compute Project Modular Hardware System: https://www.opencompute.org/projects/server/mhs/
- NASA Systems Engineering Handbook, verification/validation: https://www.nasa.gov/reference/system-engineering-handbook-appendix/

## Claim ceiling

A passing audit means this synthetic input satisfies the evaluator's current contract and arithmetic checks. It does not establish real diagnostic accuracy, reduced repair time, product safety, measurement fitness, durability, human-factors benefit, legal conformity, or general performance across device families.


## Contract hardening follow-on (2026-10-18)

The evaluator now requires a frozen `contract_sha256`, exact `protocol_id`, a single root currency (no silent cross-currency cost comparison), non-empty per-episode procedure identity, and observation references for every started attempt. A2 recommendations must reference only observations recorded for that same episode and carry a versioned advisory profile and non-empty rationale. A safety-stopped episode must have a failed safety gate; a not-restored episode cannot report a passed function gate.

The canonical contract-digest preimage is the following UTF-8 text, including a newline after every line:

```text
HUMAN-TOOL-CONTRACT-v0.1
arms=A0,A1,A2
success=verified_restoration AND function_gate=pass AND safety_gate=pass AND distinct_function_safety_receipts
unknown_cost=null_never_zero
critical_fault_safe_to_operate=forbidden
recurrence_claim=verified_restoration AND positive_follow_up_use_exposure_hours
A2_claim=hypothesis_or_abstain_only
advisory_evidence_refs=nonempty_subset_of_episode_observation_refs
all_started_episodes=require_observation_refs
currency=single_root_currency
```

Its SHA-256 is `c7ffbabaca875faf066ee63e5d12acf7e46d36001cf36c9590799bffbea61577`. This digest fingerprints only the named v0.1 contract preimage; the report separately fingerprints the exact JSON fixture bytes. The advisory profile ID in this synthetic fixture is an identifier, not a real model/code digest.
