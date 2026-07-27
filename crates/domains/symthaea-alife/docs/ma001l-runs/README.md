# MA-001L run logs — preserved raw output

Per `ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md`'s preregistration and the
`../ma001-runs/`/`../ma001r-runs/` non-erasure precedent.

- **`ma001l-run-2026-07-26.txt`** — all 7 configs (Hebbian-only, TD-only, Neither, Delta-rule ×2
  bias settings, Delta-rule+TD ×2 bias settings), all 7 gates (plan §6) computed per config via the
  structured `Ma001lGateResults` type before any prose verdict, plus the summary table. Run via
  `cargo run -p symthaea-alife --example ma001l_run --release`. Build noise excluded — this is the
  program's own stdout only.

**Result** (see the plan doc's §11 for the full write-up): **Delta-rule passes all 7 gates**, both
with and without bias-learning — held-out error 0.138/0.131 (best of any config, well below the
0.35 unconditional baseline), direction-correct and well-separated counterfactual predictions
(0.847 vs 0.423), a genuine reversal (flips and holds), and action-specificity ratios ~5-8× above
the required 2× bar. Gate G (shuffled genuinely fails) passed narrowly (9.3% vs. the 10%
tolerance) — disclosed as a real but not wide margin, not glossed over.

**TD-only and Delta-rule+TD both collapse catastrophically** — held-out error pinned at exactly
0.5500 (the unconditional-baseline mean), counterfactual predictions hard-zero on both contexts,
and a 90% drift in the *unrelated* Forage/Rest self-transition coefficients. This independently
confirms and sharpens MA-001R's own "TD-only degenerates to a uniform matrix" finding — reproduced
here with zero live `Organism` involvement, isolating it as a genuine instability in
`TemporalDifferenceLearner::update_model`'s own dynamics under this kind of sustained, perfectly
alternating exposure, not an artifact of the earlier Organism-based protocol. Composing the delta
rule with TD does not help — TD's own instability dominates and destroys whatever the delta rule
alone would have learned.

Hebbian-only fails Gate D (reversal never occurs) and Gate F (action-specificity — its own
coupling is *not* action-specific, exactly as Gate 0(b) predicted: the trigger condition
`state.mean[i] > 0.5` fires identically regardless of which action tag a tuple carries, so its
renormalization artifact shows up on Forage's own matrix too, not just Transfer's).
