# PITB component-first split candidate — research note (2026-10-29)

**Status: exploratory implementation; not a qualified benchmark.** This note introduces a successor-protocol direction only. It does not revise, replace, or re-score the frozen PITB v0 split.

## Why start this work

PITB v0 first assigns 372 theorem families to train/validation/test and then builds an audited graph of hard cross-family relationships. The analyzer correctly refuses confirmatory qualification when an audited hard-connected component crosses split boundaries. That is safe, but it can leave the current split unable to qualify after the graph is discovered.

A successor should make the independent unit consistent across the entire protocol: **discover and audit the relationship components first, then allocate whole components to splits**. This prevents a known hard-connected component from straddling train, validation, and test by construction. Component count—not family count—remains the inferential cluster denominator.

## New exploratory CLI

`build_component_first_split_candidate` accepts:

```text
<pinned-v0-splits.json> <component-manifest.json> <candidate-output.json>
```

It verifies that the input split is exactly the pinned v0 family universe and partition; checks that the component manifest is bound to that partition; reconciles every family-to-component row against the component membership lists; rejects duplicate, missing, unknown, or contradictory memberships; and emits a deterministic candidate assignment with every component kept whole.

The allocator uses a two-stage design:

1. Sort components by a fixed SHA-256 rank and allocate them to exact floor-based 70/15/15 **component-count quotas**. This makes the independent resampling unit the primary split unit.
2. Repeatedly apply the best deterministic cross-split component swap that reduces the sum of squared relative errors from the target family counts. Every move swaps one component for another, so component quotas cannot change. The search stops at a deterministic one-swap local optimum; it does not claim a global optimum.

The family-balance objective uses no theorem outcomes. The output reports family and component counts/shares, target component quotas, and the maximum absolute share error for each denominator. It also reports whether both validation and test meet the current structural floor of 20 components. Under the exact floor-based cumulative 70/15/15 cut points, 131 total components is the smallest count that yields at least 20 in both held-out splits (131 gives quotas [91, 20, 20]; 130 gives [91, 19, 20]). This is a structural minimum only: it does not establish power or confirmatory qualification. Exact 70/15/15 family proportions are not guaranteed, especially when the graph contains giant components. These diagnostics make that trade-off visible rather than concealing it.

The output is always marked `candidate_only` and `qualified_for_confirmatory_use: false`. The command is a design instrument, not a new benchmark result.

### Synthetic allocation screen

`scripts/research/pitb_component_allocation_sensitivity_v0.py` compares the hash-ordered family-deficit heuristic to the component-quota-first swap optimizer on synthetic component-size profiles. It is design-only and uses no PITB relationship inventory or theorem outcomes. Reproduce with:

```bash
python scripts/research/pitb_component_allocation_sensitivity_v0.py --random-replicates 1000 --output /tmp/pitb-allocation-screen.json
```

An independent authoring-session implementation of the same algorithm was run over 1,000 randomized 372-family size profiles: maximum family-share error improved in 906 cases, tied in 94, and worsened in none; the maximum of family-share and component-share errors improved in 990 cases, tied in 6, and worsened in 4. Mean candidate-minus-legacy component-share error was -0.03044. These are **synthetic screening observations**, not a run of the checked-in script and not evidence about the real PITB graph. The checked-in script remains the reproducibility path and still needs its own execution.

## Critical trust-boundary blocker

The v0 relationship inventory and signed coverage receipt are bound to the **old partition digest**. Reassigning families changes that digest. The existing receipt therefore cannot simply be reused as proof that the new split is qualified.

A successor receipt contract must separately bind at least:

- the exact sorted family-universe digest, independent of split assignments;
- the complete relationship-inventory digest and its schema/version;
- the reviewer trust-store and review-methodology digests;
- authenticated reviewer attestations and edge-level reviewer identity/organization rules;
- an unambiguous rule for unresolved hard relations (conservatively connect them or block qualification).

The successor verifier must verify those bytes, rebuild the hard-component graph independently from the signed inventory, and prove that every family appears exactly once and every component belongs to exactly one split. It must not trust a candidate manifest's claimed component IDs or qualification bit.

## Required acceptance tests before any confirmatory use

1. **Universe integrity:** missing, extra, duplicate, or renamed family IDs reject; output family-universe digest is stable.
2. **Graph integrity:** duplicate component IDs, repeated family membership, missing component members, and family/component table disagreements reject.
3. **Split integrity:** no hard component crosses partitions; deterministic replay reproduces every assignment and the partition digest.
4. **Receipt integrity:** old split-bound v0 receipt cannot qualify the new assignment; invalid signature, untrusted reviewer, unsigned edge reviewer, same-organization reviewer pair, incomplete inventory, and unresolved cross-split hard relation all fail closed.
5. **Adversarial structure:** isolated nodes, a single giant component, highly imbalanced component sizes, chains, cycles, and transitive cross-family links are covered. Publish actual family/component counts for all splits.
6. **Leakage and estimand audit:** separately inspect soft/citation relations and temporally newer source exposure; keep this distinct from hard-component construction. Freeze target eligibility only after split construction, and never tune a split to improve evaluation scores.
7. **Inference:** simulate strict-null and weak-null scenarios under component-size imbalance and within-component correlation; choose a calibration-valid analysis and freeze it before model outcomes. The 20-component floor is a provisional floor, not a substitute for simulation-based power.

## Research and comparison rationale

Recent formal-benchmark audits demonstrate that kernel-accepted proofs do not guarantee correct task semantics or robust evaluation. T² evaluates dependent-successor behavior rather than relying on compilation alone. Research-level collections such as TCSAlgBench are refreshable and use matched model-call opportunities. PITB should borrow these lessons while keeping its central claim narrower and falsifiable: whether a typed proof insight changes held-out, exact-target, semantically checked solve probability under matched budgets.

References:
- Faults in Our Formal Benchmarking (ICML 2026): https://proceedings.mlr.press/v306/ammanamanchi26a.html
- T²: Benchmarking Testing in Automated Theorem Proving (ACL 2026 Industry): https://aclanthology.org/2026.acl-industry.150/
- TCSAlgBench (September 2026 preprint): https://arxiv.org/abs/2609.35606
- Goedel-Prover-V2 (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html

## Decision boundary

This work can improve **methodological validity**, not prove a solver improvement. It must never be reported as a PITB utility result, SOTA result, empirical type-I error estimate, or proof that a public theorem source was absent from model pretraining. Those claims require a separately frozen and executed successor protocol, independently verified receipts, exact-head tests, matched-budget model runs, and fresh temporal replication.
