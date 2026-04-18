# Phase 6 Session 1 — HDC fingerprint cluster separation

**Status:** complete. 2026-04-18.
**Go/no-go decision:** **go** (weak positive signal, session 2 empirically justified).

## Setup

Fingerprinted the 31 Lake-verified goals from the seed-42 ingest
baseline (22 Lake-accepted, 9 Lake-rejected). Each goal's raw Lean
source → token-bag HDC encoding with positional binding → 16,384D
`BinaryHV` signature. Then measured pairwise cosine similarity
within each Lake outcome class and between classes.

## Result

| Partition | Mean cosine | Pairs |
|-----------|-------------|-------|
| within-accept | +0.247818 | 231 |
| within-reject | +0.228814 | 36 |
| between | +0.234030 | 198 |

**Effect size (within-accept − between): +0.013788** (+1.4%).

## Interpretation

- **Signal is present but weak.** The within-accept cluster is
  slightly tighter than baseline. This means that when the cascade
  closes a goal, the HDC signature carries *some* information about
  "why" — goals that close share more token structure with each other
  than with goals that don't.
- **Null rejected (barely).** The effect size exceeds the pre-declared
  null threshold of 0.005. Per the Phase 6 scoping doc, this is the
  go/no-go condition.
- **Small-N caveats.** 9 rejected goals is not many. 36 within-reject
  pairs gives a high-variance baseline. The within-reject mean
  (+0.229) is actually *lower* than between (+0.234), which is
  counterintuitive — likely noise in the small denominator.

## What this licenses

Session 2 (4 cascade variants tournament) is empirically justified.
Given the signal is positive but small, Session 2's matrix should
reveal whether cascade choice actually correlates with signature
neighborhood; if that correlation is zero, Session 1's +0.014 was
noise and we should publish the null.

## What this does NOT claim

- Symthaea "understands" the math goals. It doesn't. The encoder is
  a crude token-bag with positional binding — not the cognitive loop.
- The current encoder is good enough. It might not be; the cognitive
  loop's `wisdom_hv` (from actual HDC→CfC→Φ cycling) might produce
  better clusters. That's a Session-1-refinement candidate if Session
  2 fails.
- The cluster separation is task-specific. "Lake accepts this goal
  given the current cascade" is an artifact of our cascade choices,
  not of intrinsic goal difficulty. A different cascade would produce
  different accept/reject labels and maybe different clusters.

## Reproduction

```bash
cargo run -p symthaea-lean-bridge --example phase6_fingerprint
```

Reads `docs/phase3-results/ingest_baseline_seed42_n50.csv` for Lake
outcomes. Reads raw `.lean` sources from the corpus.

Output saved to `docs/phase6-results/session1_fingerprint_pairs.csv`
(pairwise cosine similarities with kind labels).
