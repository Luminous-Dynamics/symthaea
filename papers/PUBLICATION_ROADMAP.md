# Symthaea Publication Roadmap

**Last updated**: 2026-04-18
**Context**: Post-audit triage of 50+ papers across the corpus.

Papers fall into four publication readiness tiers. Tier assignment is
based on (a) adversarial-audit severity, (b) reproducibility-script
status, (c) drift status against current code, (d) whether claims rest
on deployed vs simulated evidence.

## Tier A — Submit now (1-2 week to venue submission)

These have survived adversarial review, have `reproduce.sh` scripts,
and carry honest caveats proportional to the evidence. Ready for
arxiv + venue submission after author-contact polishing.

| Paper | Venue target | Work remaining | Critical reviewer risk |
|-------|-------------|----------------|------------------------|
| `physics-math/ramanujan/` | ICML / NeurIPS workshop (scientific discovery) | cover letter | Machinery ceilings are named in the paper; Kepler-only sanity check is a strength not a weakness |
| `consciousness-theory/stochastic-resonance/` | Consciousness and Cognition, or Entropy | none — empirical validation of the MI estimator is now committed as Appendix A (see commit e2bd…) | Lower — the Appendix A negative result (estimator is not Shannon MI; it is bit-level HDV overlap) resolves the #1 reviewer concern. Residual risk is the inverted-U claim itself, which now reads honestly as an HDC-geometry phenomenon rather than an integration-theory result. |
| `theory-foundations/hai-consciousness/` | arXiv cs.AI (preprint) | abstract tightening; target Nature Machine Intelligence after arxiv | Butlin scope caveat needs to survive review — cannot be removed |
| `evaluation/epistemic-gating/` | ACL / EMNLP | recompile PDF with new caveats | Reviewer will want TruthfulQA or FEVER external eval; hedge via "future work" is acceptable for first submission |

**Ramanujan especially is publishable as-is** — formally proven results (9 Tier-B invariants via Z3), reproducible via Docker, honest about machinery ceilings.

## Tier B — Revise then submit (2-6 weeks)

Survived audit but reviewer pushed back on one specific empirical gap.
Fixable with focused follow-up work, not a full rewrite.

| Paper | Venue target | Revision work |
|-------|-------------|---------------|
| `consciousness-theory/kosmic-theory/` | Philosophy of Mind | "On circularity" addition may need elaboration; add multi-substrate-comparison section as future work |
| `applications/mesh-radio/` | IEEE INFOCOM | one actual LoRa deployment (Pixel 8 + Raspberry Pi 4 + LoRa HAT) would move this to Tier A. Without deployment, reframe as architecture paper rather than systems paper |
| `evaluation/psych-bench/` | Behavior Research Methods | address ICC<0.50 concern with either (a) 30-seed resampling OR (b) explicit split of "reliable 5" vs "architectural 140" benchmarks |
| `physics-math/biosphere-coherence/` | Systems biology venue | Sepkoski r=0.92 needs error bars; Chicxulub verdict honesty already in chapter but not paper |
| `consciousness-theory/integration-differentiation/` | Neuroscience theory | scale-up validation (N=100, N=1000 FHN networks) |

## Tier C — Substantial rewrite required

Audit found fundamental issue; the paper's thesis needs reframing,
not just hedging.

| Paper | Issue | Rewrite direction |
|-------|-------|-------------------|
| `governance/stewardship/` | Scales from N=50 to planetary claim; alignment-through-consciousness confuses necessity/sufficiency | Reframe as thought experiment + architectural proposal, not deployed-system paper. Three caveats now in place (commit 67b3414eb7) help but don't resolve. |
| `governance/embodied-governance/` | Core 2×/34%/2.7× results measured within Symthaea cognitive loop, not on Mycelix network sim (which showed null) | Paper already hedged in footnote (commit 0dda76306d); needs actual network-sim experiment to upgrade to Tier A |
| `governance/consciousness-security/` | Safe tiers (Green/Yellow/Orange/Red) work structurally; claim "security metric" needs adversarial red-team | Run one CTF-style evaluation; would move to Tier A |
| `consciousness-theory/phenomenal-topology/` | Adopted Feb 2026 draft; needs authorship decision + cross-model replication beyond BGE-M3 | Replicate on at least 2 more embedding models (E5, Instructor), THEN submit |

## Tier D — Hold / theoretical / aspirational

Not submission targets for 2026. Keep in corpus for cross-reference;
revisit when supporting code matures.

- `applications/consciousness-sonification/`, `consciousness-music/`, `consciousness-music-synthesis/` — music papers lack backing synthesis-benchmark code
- `applications/digital-twin-psychiatry/`, `space-debris-conjunction/`, `desci-reproducibility/` — outline-only, no draft
- `consciousness-theory/glyph-codex/`, `dream-engine/`, `neuroevolution-consciousness/` — draft PDFs are clean (drift-audit pass) but don't have benchmark evidence to distinguish them from thought experiments
- `applications/geodesic-code-synthesis/` — early stub
- `evaluation/desci-epistemic/` — draft, no empirical results

## Draft (Mar-28 freeze) review outcome

The full bulk-drift audit found **all 27 Mar-28 frozen drafts are clean**
against current code. No quantitative drift remaining. The Mar-28 freeze
was a reasonable snapshot; these papers mostly need Tier B/C/D
categorization decisions, not number fixes.

## Book

`book/symthaea_book.tex` is at 324 pages, 44 chapters, includes the 3
new chapters (Ramanujan, Biosphere, I-D Tradeoff). Weak claims have
been hedged (commit 26146f1736). No planned release venue — the book
is a monograph for the field, self-published.

Consider: a short (~40 page) **standalone version** that pulls only
the Tier A chapters (HDC, CfC, Phi, HAI core, stochastic resonance,
epistemic gating) for conference-workshop distribution.

## Sovereignty Papers

21 essays, ~61K words, CC0-1.0. Already aligned with code (the 4D→8D
rewrite in April 2026 was executed before this audit). These are NOT
submission-target papers; they're a standalone essay series distributed
via the project's public channels.

## Action priorities (next 2 weeks if you pick one)

1. **Submit Ramanujan to an ICML/NeurIPS scientific-discovery workshop.**
   Lowest friction path to first publication. The paper is genuinely
   strong; the 12-session arc of honest failure analysis is what
   reviewers want to see.

2. **Submit Stochastic Resonance to Entropy (or Consciousness and Cognition).**
   After commit `e2bd…` added Appendix A (empirical MI-estimator
   validation), this paper now meets the reviewers' ``validate the
   estimator'' ask. The finding is scoped honestly: bit-level HDV
   overlap, not Shannon MI. Ready for submission.

3. **Run the alternative-estimator replication for stoch-res.** This
   is the single highest-value follow-up experiment: replicate the
   inverted-U curve using a validated Shannon-MI estimator (e.g.,
   k-NN MI) on the same coupled-HDV dynamics. If the inverted-U
   survives, the paper's strong claim is re-enabled. If not, the
   bit-level-overlap reframe stands. Either outcome is publishable.

4. **Run the 30-seed Psych-Bench resampling.** Moves psych-bench
   from Tier B to Tier A and fixes the single strongest reviewer
   objection. Run time: ~8 hours of compute.

5. **LoRa hardware deployment for mesh-radio.** Single-weekend project
   with a Pixel 8 + Raspberry Pi + LoRa HAT. Tier B → Tier A.

6. **Do nothing more on Tier D papers.** They're corpus material, not
   publication candidates. Archive or leave as-is.

The strongest move for the corpus as a whole is to submit Ramanujan first,
observe real peer-review response, and let the reviewer feedback inform
how much further hedging the other Tier A papers need. Stochastic
Resonance is now the second-strongest submission candidate after its
Appendix A validation addressed both reviewers' central concern.
