# Genesis exploratory run logs — preserved raw outputs

Per `ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md`'s deliberate practice of not erasing a
retracted result: both runs are kept, not just the corrected one.

- **`pre-audit-contaminated-2026-07-26.txt`** — the original 20-condition run (`examples/
  genesis_explore.rs` before the Genesis v0.1 audit). Produced the "population turnover lower
  under `FixedPartners`" observation, reported as a candidate phenomenon and **retracted the same
  day** once an external review found six causal-plumbing confounds sufficient to explain it
  without any social mechanism (see the plan doc's "Genesis v0.1 — Causal Plumbing Audit"
  section).
- **`post-audit-corrected-2026-07-26.txt`** — the 30-condition rerun after all six gates were
  fixed (tagged `genesis-v0.1-closed`, commit `49b188e094`). The retracted finding does not
  reappear: `Clonal`-arm births are statistically indistinguishable between pairing modes (301.4
  vs. 301.0 mean), confirming the original pattern really was the confounds, not a real effect.

Sequence preserved on purpose: apparent effect → causal audit → confounds found → corrected rerun
→ effect vanished. That's the record of how this substrate earned its current credibility, not
just a snapshot of the current state.
