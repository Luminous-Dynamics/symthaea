# MA-001R run logs — preserved raw output

Per `ALIFE_MA001R_SOCIAL_PHYSICAL_COUPLING_PLAN_2026-07-26.md`'s preregistration and the
`../genesis-runs/`/`../ma001-runs/` non-erasure precedent: the probe's raw output, kept as run,
not just described in prose.

- **`ma001r-run-2026-07-26.txt`** — Gate 0 summary, all 4 learning-pathway ablation arms
  (Hebbian-only / TD-only / Both / Neither), both essential controls (equal-outcome,
  shuffled-context), the reversal condition, and the program's own printed interpretation. Seed 1,
  `Ma001rConfig::default()` (training_ticks 2000, held_out_ticks 200, reversal_ticks 2000). Run via
  `cargo run -p symthaea-alife --example ma001r_run --release`. Build noise (cargo compile output)
  excluded — this is the program's own stdout only, starting at its first `println!`.

**Result** (see the plan doc's §12 for the full write-up): **Full null**, not "Learning-rule
limitation" as the program's own naive printed heuristic claimed — do not trust that line at face
value; the plan doc's §12 explains why and re-derives the correct rung directly from §8's literal
criteria. Headline evidence: the shuffled-context control's Δpredicted (0.4499) is *larger* than
the properly-bound "Both" arm's own value (0.2747), the opposite of what a genuine
context-specific coupling would require; the reversal condition never inverted direction over
2,000 further ticks; TD-only's own coefficients collapsed to an uninformative uniform value
(1/6 in every cell); and TD-only's raw movement didn't even exceed the equal-outcome control
built specifically to measure pure measurement-pipeline noise.
