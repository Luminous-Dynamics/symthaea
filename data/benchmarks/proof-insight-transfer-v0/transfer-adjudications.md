# Proof-transfer adjudications: static pilot notes

**Date:** 2026-10-20  
**Upstream source snapshot:** `openai/math@1581d4cbdd69a0faf34b2089614fe2a814316a6a`  
**Status:** curator-level static review only. No generated transfer attempt, Lean compilation, Comparator run, or benchmark score is represented here.

This file records why two superficially plausible graph/geometry transfers should be rejected. They are development fixtures for the typed `directional_contract` field in `insight-record.schema.json`, not scored tasks and not evidence of a model failure.

## Adjudication A: Borsuk finite obstruction → Hadwiger counterexample

- **Source family 156 (train):** [`OAI.BorsukNine.main_theorem`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/OAI/Geometry/Borsuk/Counterexample.lean)
- **Target family 157 (train):** [`OAI.HadwigerCounterexample.not_hadwiger_conjecture`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/OAI/Combinatorics/HadwigerCounterexample/Main.lean)
- **Disposition:** reject direct transfer; `object_type`, `premise_unavailable`, `domain_mismatch`, `precondition_mismatch`.

Both arguments prove a contradiction, but the source move is specific: transform a hypothetical small-diameter cover of rank-one projectors into `Admissible` data, derive a finite support-face incidence system, and invoke a finite ten-system obstruction. The target requires a different witness pipeline: construct arbitrarily large graphs with independence number at most two and small connected-matching number, then combine a matching–minor inequality with a chromatic lower bound. The source representation and finite incidence relation do not construct the graph witness or discharge those invariant inequalities.

The machine-readable disposition is included in [candidate 156](candidates/156-borsuk-nine-finite-obstruction.json). Its status `rejected` denotes a curator's static judgment—not an empirical failed solve.

## Adjudication B: Hadwiger color-fiber bound → clique-free independence bound

- **Source family 157 (train):** [`card_le_twice_chromaticNumber_of_indepNum`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/OAI/Combinatorics/HadwigerMatching/ChromaticBound.lean)
- **Target family 184 (train):** [`CliqueFreeLog.logarithmic_independence_bound`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/OAI/Combinatorics/CliqueFree/Main.lean)
- **Disposition:** reject direct transfer; `inequality_direction`, `precondition_mismatch`, `domain_mismatch`.

In the source, `indepNum G ≤ 2` implies that every color class has size at most two, so `|V| ≤ 2χ(G)`; this is a *lower bound* on the chromatic number. The target seeks a *lower bound* on the independence number, `c_r |V| log(d)/d ≤ indepNum G`, for clique-free graphs with average degree (d≥2). The source inequality needs an upper bound on independent-set size and cannot be reversed to prove the target. The target proof uses a weighted optimizer, triangle-mass control, a degree cutoff, and maximum-degree independence reasoning instead.

This is a directionality negative control: a retriever must not infer transfer from shared vocabulary such as graph, independent set, or chromatic number. It must represent whether the source's premise/conclusion and invariant direction can discharge a concrete target obligation.

## How this changes the benchmark

The v0 typed transfer contract now records:

- a compatibility class (`compatible`, `incompatible`, `partial`, or `unknown`);
- machine-readable mismatch codes, including `predicate_mismatch` and `inequality_direction`;
- the move kind plus typed source/target obligations: domain, object type, predicate, relation direction, quantifier scope, and preconditions;
- what the source move requires and what the target goal requires;
- consistency constraints: `verified_transfer` requires `compatible`; a curator-rejected transfer requires `incompatible` or `partial`.

These fields do not prove that a strategy is valid. They make the reason for accepting/rejecting a candidate inspectable and testable.

### Proposed transfer-matcher gates

Before claiming positive transfer utility:

1. Reject **100% of a frozen finite negative-control suite**, including the two adjudications above and synthetic polarity, quantifier-scope, unavailable-premise, wrong-domain, and missing-dependency controls.
2. Include expert-labelled positive controls; report false rejection as well as false acceptance. Do not accept an all-abstaining matcher as successful.
3. Report acceptance/rejection precision and recall separately from proof-solve rate, semantic-gate success, time, and cost.
4. Only call a transfer verified after the exact target theorem and its declared dependency/axiom policy pass the checker and semantic gate.
5. Keep every result above labeled as proposed until executable fixtures, positive controls, and the matcher have actually run.

A broad match in topic or proof shape is a retrieval hypothesis, not mathematical transfer.

## Adjudication C: family 026 eventual-density transport → family 012 largest-prime-factor ordering

- **Source family 026 (train):** [`Problem344.large_gaps_and_ratio_density`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/ComparatorChallenges/PrimeGaps.lean), scoped in [`lean/docs/026.md`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/docs/026.md).
- **Target family 012 (validation, now development-exposed):** [`JointDickmanPaper.increasing_order`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/ComparatorChallenges/JointDickman.lean), scoped in [`lean/docs/012.md`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/docs/012.md).
- **Disposition:** provisional direct-transfer rejection; `object_type`, `predicate_mismatch`, `premise_unavailable`. Development-only negative control, never a confirmatory validation result in this version.

Family 026 defines the source predicate over indices of the prime sequence: `p_n / n < p_(n+1) / (n+1)`. Its candidate insight moves from a positive lower bound on a source set's initial count, through eventual inclusion and finite-prefix accounting, to a positive lower asymptotic density. Family 012's target predicate is over integers whose consecutive largest prime factors satisfy `P⁺(n) < P⁺(n+1)`, and its declared result proves an ordinary natural density of one half as a corollary of the joint Dickman law.

The two claims share vocabulary about primes and density, but the predicates are not the same and the inspected target contract supplies no eventual-inclusion relation from the source set. The source insight cannot discharge that missing bridge merely because both conclusions mention positive density. The direct transfer should be rejected unless a separate, audited intermediate lemma establishes the exact set relation and compatible counting conventions.

**Evidence boundary:** this is a provisional curator judgment based on the pinned family scope notes and comparator statement signatures. The comparator files include `sorry` placeholders, so this record does not claim the full source/target proofs were rebuilt or that the transfer was experimentally attempted. It is not an observed model false-accept.

**Exposure handling:** inspection happened after checking the immutable split assignment. Family `012` is therefore recorded in `exposure-audit.md` as development-exposed and excluded from validation baseline selection/tuning for this benchmark version. `splits.json` remains unchanged; original and exposure-adjusted counts must be reported separately.