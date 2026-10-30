# PITB support-antichain benchmark extension (2026-10-30)

**Status:** research protocol and synthetic analyzer only. This extension is not yet wired to the Lean pilot receipts, has no real ablation ledger, and has no model results. No Lean build, CI pass, transfer gain, or SOTA win is claimed.

## Paradigm shift: support is a family of routes

PITB now distinguishes three layers:

1. **Available context:** declarations present in the elaborated state.
2. **Actual proof-term support:** declarations syntactically referenced by one checked proof term, plus local type/value dependencies.
3. **Minimal sufficient support:** the smallest contexts under which a pinned prover policy and budget still solve the exact target, passing both the trusted kernel and the independent semantic gate.

Layer 2 is an observation about one proof construction. It is not a minimality or necessity certificate. Layer 3 must come from counterfactual ablation and reproof.

Even layer 3 is not generally one set. If a theorem is provable through either \`h\` or \`hEq\`, its minimal support is the antichain \`[{h}, {hEq}]\`. Their intersection is empty: neither premise is unavoidable. Their union says both can appear in some minimal route. A flat list confuses route-specific support with universal necessity.

For a fixed target, environment, toolchain, fixed context, solver policy and budget, let \`D\` be the registered mutable declarations and \`success(S)\` mean the exact target passes both formal gates with precisely \`S\` available. The finite minimal-support antichain is:

\`M = { S ⊆ D : success(S) ∧ ∀T ⊂ S, ¬success(T) }\`.

The intersection of all members of \`M\` is support common to every observed minimal route. It is an operational property of the frozen experiment, not a theorem-level logical independence proof.

## Executable analyzer

\`scripts/research/pitb_support_antichain_v0.py\` implements a fail-closed, exact powerset analyzer. Each row binds the target statement, component, Lean environment, toolchain, non-mutated context, resource budget, prover policy, local-context digest, and state/run/checker/semantic receipt digests. A group must contain every one of the \(2^{|D|}\) subsets exactly once. It rejects:

- missing/duplicate subsets or declarations outside the mutable universe;
- mixed target, environment, toolchain, fixed context, budget, or solver policy;
- duplicate context digests across distinct subsets;
- unresolved outcomes or "success" without both kernel and semantic-gate success;
- a full-context baseline that is not a verified exact-target success.

The exhaustive cap defaults to 12 mutable declarations and has a hard maximum of 16. Larger context lattices require a separately labelled approximate or designed-sampling track; sampled rows must never be reported as an exhaustive antichain.

An optional prediction ledger scores (i) declarations claimed necessary across every minimal route and (ii) a proposed sufficient route. Metrics include necessity precision/recall/F1, route sufficiency, exact minimal-route recovery, prediction coverage, selective route risk and Brier score. Abstentions remain in coverage accounting. The analyzer checks receipt digest syntax and row binding only; external workflows must open and authenticate the referenced receipt bytes before accepting benchmark evidence.

## Calibration and qualification

The current Lean example suggests an eight-cell follow-on calibration over \`{h, hEq, hIrrelevant}\`, holding the exact goal, non-mutated context, environment, policy and budget fixed. The hypothesis is that \`{h}\` and \`{hEq}\` each form a minimal successful route, while \`hIrrelevant\` occurs in neither. This is not yet an observation: all eight contexts must actually elaborate and run, with kernel and semantic receipts, before that antichain is admitted.

After this pilot, expand to independently reviewed examples with conjunctive premises, alternative lemma chains, definitional and propositional reformulations, polarity/quantifier near-misses, and domain-changing distractors. Targets enter scored evaluation only after two independent reviewers agree on the theorem correspondence and mutable support universe.

Proposed future gates, not achieved values:

| Dimension | Qualification target |
|---|---:|
| Lattice/receipt integrity | 100% exact subset accounting; zero identity mismatches |
| Necessary-support prediction | Declaration F1 ≥ 0.90 on held-out audited components |
| Route sufficiency | ≥ 0.95 at ≥ 0.80 non-abstaining prediction coverage |
| Minimal route recovery | ≥ 0.80 exact antichain-member rate |
| Calibration | Route-sufficiency Brier ≤ 0.10, with reliability intervals once sample size permits |
| Counterfactual action | Zero use of a move after registered support removal; ≥ 0.95 stability on benign edits |
| End-to-end utility | Positive paired exact-target semantic-gate gain over the strongest validation-selected baseline at matched budgets |

These point gates are qualification targets for a future, powered cohort. Do not assert rates from the synthetic self-test or tiny Lean calibration. Inferential units remain audited cross-family components, not individual ablation rows.

## Industry comparison and competitive path

The nearest public frontiers solve adjacent tasks, not this exact exhaustive support-antichain problem:

- Goedel-Prover-V2 reports 90.4% MiniF2F pass@32 for its 32B self-correction setup. This is a strong proof-synthesis reference, not a support-minimality comparison. [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html).
- LeanGraphSearch reports MathlibQR Recall@100 of 94.7% versus 82.7% for LeanSearch v2 on its reported 810-query fair subset, and improved route coverage/downstream proof-loop results in its own setup. It should be a serious pinned retrieval baseline in PITB Arm I, not treated as a direct antichain competitor. [Paper](https://arxiv.org/abs/2610.22339) · [results](https://pluslabnlp.github.io/LeanGraphSearch/).
- T² evaluates successor-theorem behaviour on 2,206 repository-derived Lean tasks and reports 38.9% best testing accuracy, showing why compilation and semantic correctness must remain separate. [ACL 2026 Industry](https://aclanthology.org/2026.acl-industry.150/).

The fair local comparison is a paired rerun on identical targets, Lean/Mathlib revisions, exact context lattices, checker, semantic gate, model/tool budgets and resource caps: no retrieval; BM25; LeanSearch v2; LeanGraphSearch; typed dependency-graph insight; a transparent proof-term heuristic; Symthaea HDC/LTC support prediction; and a human-adjudicated support oracle as an upper-bound diagnostic. Select the strongest practical baseline on validation, freeze it before test, and report the full success/cost frontier.

A defensible superiority claim requires a positive matched-task effect with component-clustered uncertainty, the antichain and counterfactual gates, complete relationship coverage, exact reproducibility, and independent temporal or cross-repository replication. Higher pass@k on an unrelated corpus is contextual only.

## Reproduce synthetic controls

\`\`\`sh
python3 scripts/research/pitb_support_antichain_v0.py --self-test
python3 -m py_compile scripts/research/pitb_support_antichain_v0.py
\`\`\`

The synthetic controls exercise alternative routes, conjunctive support, prediction scoring, incomplete-lattice rejection, budget identity, and proof-gate enforcement. They validate analyzer mechanics only; they are not Lean evidence, model results, or proof-transfer observations.
