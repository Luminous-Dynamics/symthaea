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
- machine-readable mismatch codes, including `inequality_direction`;
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
