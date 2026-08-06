# UAL-P2 Phase 6 — Preregistered Re-evaluation: Real Result

**Status: frozen result record.** This is the Phase 6 (P2 re-evaluation) work unit named in
`MASTER_ROADMAP.md`'s UAL/HDC-audit rows, scoped and approved via
`/home/tstoltz/.claude/plans/ethereal-wandering-dewdrop.md` ("UAL-P2 Re-evaluation (Phase 6):
Preregistered Power Analysis + Practical Floor"). This is the last open item in the entire
UAL + HDC-binding-algebra audit arc. Recorded verbatim, exactly as produced by the
preregistered confirmatory test, with no post-hoc adjustment.

## Source commit

- (this commit) — `p2_second_order.rs`: `VALUE_TRANSFER_PRACTICAL_FLOOR`,
  `PHASE6_CONFIRMATORY_N`, `phase6_confirmatory_seed_is_disjoint_from_pilot_seed`,
  `phase6_confirmatory_value_transfer_result`.

## What was preregistered, before any new data collection

- **Practical-significance floor**: `VALUE_TRANSFER_PRACTICAL_FLOOR = 0.60` — not an arbitrary
  new number; reuses this file's own existing positive-control bar for the structurally
  analogous reward-probability value (`positive_control_observed = mean_a_value > 0.6`).
  `Demonstrated` requires the choice-rate (B vs. neutral D) 95% bootstrap CI lower bound to
  clear this floor, not merely exclude bare chance (0.5) — closing exactly the "statistical
  significance alone is not sufficient" gap the original review flagged for this phase.
- **Confirmatory sample size**: `PHASE6_CONFIRMATORY_N = 2400`, from a one-sided power
  analysis (α=0.05, power=80%) using the existing frozen n=40 pilot estimate (p̂=0.625,
  `SYMTHAEA_UAL_FROZEN_EVIDENCE_2026-07-27.md`) purely as the *planning* estimate of effect
  size:
  ```text
  n = ((z_0.05*sqrt(0.60*0.40) + z_0.20*sqrt(0.625*0.375)) / (0.625-0.60))^2 ≈ 2356
  ```
  rounded up to 2400.
- **Calibration/holdout partition**: the existing n=40 pilot (config seed 42) served as the
  calibration estimate; the confirmatory run used a disjoint seed (`42 ^ 0xFEED_C0DE`),
  verified by `phase6_confirmatory_seed_is_disjoint_from_pilot_seed` to produce zero overlap
  in the underlying `trial_seed` stream with the pilot.

## Real empirical result

```text
[blocked] relational-identity(A over C) rate=1.0000 [1.0000,1.0000]; value-transfer
(choose B over D) rate=0.6579 [0.6383,0.6763] under Blocked; mean(sim_A - sim_D)=0.7432;
mean A_value=0.6999

[interleaved] relational-identity(A over C) rate=1.0000 [1.0000,1.0000]; value-transfer
(choose B over D) rate=0.6579 [0.6383,0.6763] under Interleaved; mean(sim_A - sim_D)=0.7432;
mean A_value=0.6999

both schedules independently reached Demonstrated
```

**Functional outcome: `Demonstrated`** — the first `Demonstrated` verdict anywhere in the UAL
track, reached via a properly-powered, preregistered test. Both Blocked and Interleaved
schedules independently qualified (not one arm carrying the other via
`combine_schedule_reports`).

- Relational-identity retrieval: unchanged from the pilot, essentially at ceiling (1.0000).
- Value-transfer (choice rate): point estimate **0.6579**, 95% CI **[0.6383, 0.6763]** — the
  CI lower bound clears the 0.60 floor with a comfortable margin (0.0383 above the floor at
  the *pessimistic* end of the CI, not a borderline pass).
- This closely replicates the n=40 pilot's point estimate (0.625 → 0.6579, same direction and
  magnitude) with a CI roughly **14x narrower** (0.30 wide at n=40 vs. 0.038 wide at n=2400) —
  exactly the outcome the power analysis was designed to produce if the real effect matched
  the pilot's planning estimate, which it did.
- The mandatory schedule caveat (unchanged from the original design, since P2's accumulator
  was already proven algebraically schedule-invariant for this mechanism) is still attached:
  "replicated across schedules" here means the two orderings are algebraically equivalent for
  this learner, not that a meaningful manipulation was varied and survived.

## Interpretation

- This is genuine evidence that the benchmark-local candidate HDC learner exhibits
  second-order value conditioning (B, never directly rewarded, acquires functional value via
  its bind-and-accumulate pairing with A, and that value measurably changes a forced-choice
  decision) — not merely relational-identity retrieval, and not merely a statistically
  significant-but-trivial departure from chance.
- Per the UAL design doc's disclosure requirement, this licenses at most: **"second-order
  value conditioning demonstrated for this benchmark-local candidate mechanism"** — it does
  **not** license "UAL demonstrated" (P1/P4a remain `NotDemonstrated`/`Inconclusive`
  respectively) and makes **no claim about live Symthaea** (`SystemUnderTest::
  BenchmarkLocalHdcLearner`, not `LiveSymthaea`).
- The result is not an artifact of "large n makes any noise significant": the decision rule
  tests against a fixed 0.60 floor, not the null of pure chance (0.5), so an arbitrarily large
  n cannot manufacture a pass against that floor from a true near-chance process. The CI's
  pessimistic lower bound (0.6383) still clears the floor with room to spare.

## What was NOT done (and why)

- The 0.60 floor and N=2400 were fixed in the approved plan *before* this test was run and
  were not touched afterward, regardless of which way the result landed.
- No threshold was adjusted, no additional runs were added, and no seed was changed after
  seeing the `Demonstrated` result — the same discipline applied when the P4a-v2 redesign's
  Interleaved arm landed just outside its own tolerance and was accepted as `Inconclusive`
  rather than chased.

## What "frozen" means here

Any future work that touches this probe's mechanism, floor, or sample size should diff its
new results against exactly the numbers above, run at the same configuration
(`dimension=512`, confirmatory seed `42 ^ 0xFEED_C0DE`, n=2400). A change in these numbers is
expected and useful evidence about what changed — but only if this record stays untouched as
the "before" side of that comparison.
