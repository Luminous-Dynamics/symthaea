# PITB successor design: eligibility-aware component feasibility

**Date:** 2026-10-29  
**Status:** candidate infrastructure and protocol proposal; no reviewed real census, model outcome, or confirmatory result is claimed.  
**Scope:** successor design only. It does not modify the PITB v0 frozen partition, denominator, or receipts.

## Executive finding

Component-first splitting fixes only one level of dependence. A partition can meet its overall component-count quotas and still be unusable because many components contain no eligible theorem target, contain only excluded targets, or have unresolved eligibility.

The relevant denominator for the primary held-out inference is not:

- the number of catalog paths;
- the number of source families; or
- the total number of graph components.

It is the number of **eligible, independently connected components that contribute at least one frozen target to the estimand**, alongside the target denominator and all excluded/unknown counts.

The new diagnostic `audit_component_split_feasibility` checks this gap before a solver run is allowed to consume evaluation budget.

## Implemented candidate artifacts

- Rust diagnostic: `crates/bridges/symthaea-proof-insights/src/bin/audit_component_split_feasibility.rs`
- Census schema: `data/benchmarks/proof-insight-transfer-v0/pre-split-target-eligibility-census-v0.schema.json`
- Report schema: `data/benchmarks/proof-insight-transfer-v0/component-split-feasibility-report-v0.schema.json`
- Synthetic census/split fixtures under `data/benchmarks/proof-insight-transfer-v0/fixtures/`
- Workflow contract check in `.github/workflows/proof-insight-transfer-v0.yml`

The synthetic fixture is deliberately inadequate: it contains three eligible train components, one eligible validation component, zero eligible test components, one excluded target, and one unknown target. The correct output must retain those rows, detect the held-out floor failure, report not-ready for power planning, and remain unqualified.

The diagnostic distinguishes the synthetic fixture from a real candidate. A real candidate must contain exactly the pinned 372-family universe whose canonical sorted-ID digest is `c8ee398840443fb0693e50baa1dfda9982cd8bfebe1f81d9de3050e3b15cb048`. A reviewed census is required to use that same universe and is rejected if it only repeats the old partition-digest string while supplying different family IDs. Synthetic fixtures are explicitly labelled and cannot masquerade as a reviewed real census.

The diagnostic reconciles:

1. the exact candidate family universe and partition digest;
2. the census's independently defined family-universe digest;
3. candidate target IDs declared for each family versus the target rows;
4. target artifact and statement digests, including explicit null statement hashes for excluded/unknown targets;
5. family-level eligible/excluded/unknown classifications against target-row statuses;
6. eligible target and eligible component counts in each split; and
7. the current provisional structural floor of 20 eligible components in validation and test.

Receipt hash strings are checked for shape only. This diagnostic does **not** authenticate signatures, prove reviewers independent, validate theorem semantics, reconstruct the relationship graph from a signed inventory, or clear the main confirmatory analyzer's qualification gate. It always emits `qualified_for_confirmatory_use=false`.

## Protocol correction: freeze eligibility before assigning splits

For a successor protocol, construct and freeze an **outcome-blind target eligibility census before split allocation**, using an eligibility rule fixed in advance. This is not permission to filter the denominator after observing performance. It is the opposite: it makes the potential target universe explicit before any model outcomes exist.

The census must include every candidate target and every family in the family universe, including:

- `eligible`: source bytes and exact statement are available and required static/semantic eligibility checks pass;
- `excluded`: an explicit, reviewable reason makes the target ineligible under the frozen criteria;
- `unknown`: evidence is missing, contradictory, or unresolved.

An absent target list is not by itself proof that a family is ineligible. The family-level status and receipt must distinguish a verified absence from incomplete inventory. Every status has a pinned receipt digest and remains in raw accounting.

Eligibility decisions must be blinded to solver outcomes and treatment assignment. A later correction creates a new census artifact, digest, review receipt, and protocol version; it does not silently edit the old denominator.

## Allocation objectives for a future successor

The candidate split allocator should not optimize family share alone. The preregistered objectives should be lexicographic, with the higher-priority constraints never traded away for prettier balance:

1. **No leakage:** every audited hard-connected component belongs to exactly one partition.
2. **Eligible held-out feasibility:** validation and test each have at least the declared floor of eligible components; no model run starts if either partition fails this condition.
3. **Fixed independent-unit quotas:** component-count quotas follow the frozen allocation policy and seed. Do not move components across splits after looking at any model result.
4. **Eligible-target balance:** among allocations satisfying the hard constraints, reduce imbalance in eligible target count and optionally predeclared domain/source-era strata. All stratum variables must be outcome-blind, source-pinned, and declared before assignment.
5. **Publish imperfections:** report component counts, eligible component counts, total/eligible/excluded/unknown target counts, family counts, and share errors by split.

Because components are indivisible and can be large, exact target balance may be impossible. The allocator should report infeasibility instead of relaxing a required gate automatically. Any change to component quotas or estimand is a new protocol, not a post-hoc tuning step.

The current 20-component floor is only a provisional minimum. Before confirmatory evaluation, use the observed **pre-outcome** eligible-target distribution to calibrate false-claim behavior and power under strict-null and weak-null scenarios, including strong component-size imbalance. A passed structural floor is not equivalent to adequate power.

## Current benchmark landscape and the opportunity

Short static proof benchmarks can saturate. On 24 October 2026, the Vals AI ProofBench v1.1 page reported several systems at 100% on its 100-problem evaluation, while the live LeanEval leaderboard explicitly accepts new reviewed tasks and records incoming submissions. Those are different evaluation designs, but together they illustrate why a static headline score is not a durable research target.

Research-level and repository-scale tasks also probe different capabilities:
- T² evaluates whether dependent successor theorems remain valid after a generated theorem compiles: https://aclanthology.org/2026.acl-industry.150/
- TheoremBench studies theorem-level coverage and token efficiency over longer, dependency-rich proof structures: https://arxiv.org/abs/2606.09450
- VeriSoftBench evaluates 500 proof obligations across real Lean repositories with cross-file dependencies: https://arxiv.org/abs/2602.18307
- Formal Conjectures provides an evolving source of open and solved research conjectures; pin exact revisions rather than relying on its live denominator: https://github.com/google-deepmind/formal-conjectures
- ProofBench v1.1 live page, updated 24 October 2026: https://vals.ai/benchmarks/proof_bench
- LeanEval's evolving formalization leaderboard: https://leanprover.github.io/lean-eval-leaderboard/

PITB should use such sources as **distinct, pinned cohorts**, not merge their percentages. The distinctive contribution remains the causal transfer contrast: whether a typed insight improves exact-target, semantic-gate verified proof success under matched compute, after source availability, target eligibility, dependency leakage, and statistical uncertainty are independently qualified.

A second important guardrail is that **temporal novelty is not a standalone contamination detector**. The ACL 2026 paper *Test of Time: Rethinking Temporal Signal of Benchmark Contamination* finds that temporal performance patterns change substantially with task construction even when the underlying source material is unchanged. PITB's temporal track should therefore combine chronology with source/dependency ancestry, hidden future-artifact quarantine, mutation controls, and multiple semantics-preserving formulations of the same target. A post-cutoff score drop alone must not be reported as proof of non-contamination, and a lack of score drop must not be reported as proof of memorization. Source exposure remains an explicitly bounded, partially unobservable risk.

## Acceptance gates before any confirmatory claim

- The actual Rust diagnostic and its tests pass on the exact commit.
- Independent JSON Schema 2020-12 validation passes for census and report fixtures.
- Negative controls reject outcome-derived eligibility, an unregistered status, missing/duplicate family or target IDs, mismatched family-target indices, unsafe paths, malformed digests, wrong partition digests, and attempts to pass a qualified split as input.
- A reviewed real census covers the exact frozen family universe and every candidate target.
- The signed hard-relationship receipt is independently verified and the graph is reconstructed from the authenticated inventory.
- The held-out eligible-component floor is met; the power plan passes registered strict-null and weak-null calibration.
- The primary analysis and cost-frontier plans are frozen before model evaluation.
- Matched-budget transfer results replicate on a fresh temporal or independent-source cohort before any state-of-the-art claim.

## Current validation boundary

The source, schemas, fixtures, and CI workflow additions have been committed. This environment does not have Rust/Cargo installed, cannot resolve GitHub hosts from the shell, and the current exact-head PR query has no status entries or PR-triggered workflow runs. The new Rust binary and workflow checks are therefore **unverified**. The synthetic fixture is a contract control, not benchmark evidence.
