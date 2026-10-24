# Active Probe Design v1 — preliminary exact results

**Date:** 2026-10-22  
**Status:** Reproducible small synthetic benchmark; not a claim about real-world causal discovery or SOTA.

## Exact run

- Source commit: `687f74d2253850eb0e2045d07f8404cf07fb771c`
- Source Git blob SHA: `fd316114018565e4f5398f9b5732d0ecac8f8ebd`
- Source SHA-256: `e835933859e0fdfbebef8fa9b8fc061d5744e50fc198297e83a600378f4aec99`
- Test file Git blob SHA: `ec82cb202f935beaf9256ea60e4429414f877dd3`
- Eight committed `unittest` tests passed locally; embedded `--self-test` also passed, after copying both source files and confirming their Git blob SHAs against GitHub.
- Command: `python validation/active_probe_design.py --instances 100 --budget 3 --source-sha 687f74d2253850eb0e2045d07f8404cf07fb771c --output active-probe-results.json`
- Full 134,313-byte output SHA-256: `47ed95d74beff046c544a283fc6a159253b2fc34c1a4c36025661eb9e09fe737`. The raw output is not committed; it is deterministically regenerable by the command above.

Each of 100 deterministic instances samples 5–10 distinct two-input Boolean mechanisms from the complete 16-function truth-table family, uses integer hypothesis prior weights, and assigns four possible input-setting probes independently generated cost and damage values. The evaluator exactly enumerates deterministic outcomes and adaptive posterior branches under a three-unit cost budget. It compares fixed order, cost-first, greedy expected information gain, greedy information per cost-plus-damage, and a dynamic-programming oracle.

## Mean results over 100 generated instances

Lower posterior entropy is better; higher exact-identification probability is better. Cost and damage are separate objectives, not folded into the reported entropy.

| Strategy | Residual entropy (bits) ↓ | Exact identification probability ↑ | Expected cost ↓ | Expected damage ↓ |
|---|---:|---:|---:|---:|
| Fixed order | 1.250103 | 0.144262 | 2.760215 | 4.908091 |
| Cost first | 1.048580 | 0.170464 | 2.271837 | 5.364257 |
| Greedy information gain | 1.059519 | 0.188375 | 2.638837 | 5.024690 |
| Greedy information / (cost + 0.25 × damage) | 0.968120 | 0.206984 | 2.437333 | 5.194426 |
| Exact entropy-first oracle | 0.933603 | 0.223787 | 2.650379 | 5.616029 |

## Interpretation and limits

1. The exact oracle lowers residual uncertainty and raises exact identification relative to the tested heuristics, as expected by construction. It does **not** dominate all criteria: its lexicographic objective prioritizes entropy, then damage, then cost, so it incurs more expected damage than some heuristics.
2. Greedy information-per-cost-plus-damage is close to the oracle in entropy and uses lower expected intervention cost in this synthetic configuration. This is a Pareto trade-off, not proof of universal superiority.
3. Costs and damage are synthetic integer labels. The mechanisms are static two-input truth tables; there is no temporal feedback, noisy observation process, latent confounding, learned representation, or real physical disturbance.
4. This does not replicate the Active Causal Discovery Benchmark (ACDB), TCD-Arena, CausalVerse, PEID, PID/SURD, or IIT. Those are relevant external comparisons for subsequent stages. No actual industry leaderboard comparison has yet been performed.
5. Next gate: extend to multi-node SCMs, noisy observations, intervention side effects with defined semantics, and mechanism-family-held-out splits; then compare to established active-causal-discovery policies on the same observation/intervention contract.

This receipt is intended to make the benchmark's preliminary result inspectable without mistaking a toy oracle comparison for a field-level breakthrough.
