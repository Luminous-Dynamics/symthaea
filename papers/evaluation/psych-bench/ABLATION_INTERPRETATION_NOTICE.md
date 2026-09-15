# Psych-Bench Ablation Interpretation Notice

Status: publication claim-ceiling correction for the checked-in legacy ablation results.

This notice applies to the existing `ablation_domains.csv` data and the current Figure 8 / ablation discussion in the Psych-Bench paper. It does not change any historical numeric result.

## What the legacy presets actually do

The historical `AblationPreset::to_config()` implementation does not isolate subsystem removal from nuisance degradation. The presets combine mechanism changes with encoding-noise changes:

- `FullConsciousness`: no subsystem disablement, `encoding_noise = 0.0`.
- `CfcOnly`: disables FEP and social processing, and sets `encoding_noise = 0.35`.
- `NoFep`: disables FEP and sets `encoding_noise = 0.25`.
- `NoSocial`: disables social processing and sets `encoding_noise = 0.15`.
- `ReducedWm`: reduces working-memory capacity to 3 and sets `encoding_noise = 0.30`.
- `HdcOnly`: disables FEP and social processing, reduces working-memory capacity to 3, and sets `encoding_noise = 0.50`.

Therefore the checked-in historical contrasts are **legacy combined-model preset comparisons**. They are not clean mechanism-only interventions.

## Permitted interpretation

The historical data may support descriptive statements such as:

- a named legacy preset produced a particular observed domain score;
- two legacy presets differed by a stated amount in the checked-in run;
- the combined preset family shows more or less sensitivity in some domains than others.

Any effect size computed from these rows describes the **combined preset contrast** represented by that row.

## Not established by the historical table

The legacy table does not independently establish:

- the causal effect of disabling FEP alone;
- the causal effect of disabling social processing alone;
- the causal effect of reducing working-memory capacity alone;
- that an observed domain delta was caused by the named mechanism rather than the matched nuisance change or their interaction;
- a mechanism-specific effect size, statistical significance, or publication-grade causal estimate.

Phrases such as “FEP effect”, “subsystem removal caused”, “selectively devastates”, or equivalent causal language must not be treated as supported by the legacy combined-model table alone.

## Qualified replacement pipeline

The repository now contains a stricter additive pipeline for future deconfounded evidence:

1. **#3223** — typed ablation intervention contract, exact-head PASS at `728fa8ed…`, run `34919638492`. It separates `Baseline`, `MechanismOnly`, `NuisanceOnly`, `CombinedModel`, and `MultiMechanismCombined` conditions while preserving historical presets exactly.
2. **#3240** — provenance-preserving long-form ablation exporter, exact-head PASS at `3f71c9c0…`, run `34920373618`.
3. **#3258** — fail-closed paper projection selecting only baseline plus clean single-mechanism arms, exact-head PASS at `85c359b5…`, run `34959084822`.
4. **#3273** — explicit A/B/C/D contrast-set identity, exact-head PASS at `46ae9d18…`, run `34960598222`.
5. **#3380** — raw factorial arithmetic receipt is still under qualification at the time this notice was written and must not be treated as qualified until its exact-head workflow reaches terminal PASS.

These qualified software contracts make clean mechanism/nuisance comparisons representable and auditable. They do **not** mean that fresh mechanism-only paper data have already been generated.

## Publication migration gate

Before Figure 8 is replaced with clean mechanism-effect language, the paper workflow must:

1. execute the typed intervention pipeline on a declared benchmark suite and code subject;
2. generate fresh provenance-bearing intervention rows;
3. verify the mechanism-only projection and any estimator used;
4. preserve exact execution/output provenance;
5. update the manuscript and arXiv copy to cite the new evidence rather than silently reinterpreting the historical `ablation_domains.csv` values.

Until that happens, the current Figure 8 should be read and described only as a **legacy combined-model preset comparison**.
