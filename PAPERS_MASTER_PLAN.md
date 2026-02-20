# The Master Plan: From 0 Submissions to Published Research

> Created: 2026-02-17 | Updated: 2026-02-17 (post-audit HONEST RECKONING)
> Author: Tristan Stoltz / Luminous Dynamics
> Portfolio: 52+ papers, 0 submitted
> Philosophy: Quality over quantity. Honest claims. Ship what's ready.
>
> **AUDIT RESULTS**: Phase 0 deep verification (Feb 17) revealed only 2 of 3 planned papers are submittable.
> P1 (HAI): VERIFIED. P4 (K-Index): FIXED (9 defects). HK-1: CRITICAL INTEGRITY FAILURE (removed from Phase 0).
> P19 (K_Topo): METHODOLOGICALLY UNRELIABLE (7 compounding failures). Moved to Phase 3 contingent on reproducibility.

---

## Guiding Principles

1. **Ship what's ready.** Three papers can go out this week. Every week of delay is a week someone else might publish similar results.
2. **Honest claims only.** The Phi metric issue is real. We reframe honestly rather than publish inaccurate claims. Our reputation matters more than any single paper.
3. **Build momentum.** Start with journals (rolling deadlines, less pressure) to build a publication track record. Target conferences only when we're confident.
4. **Strategic sequencing.** Some papers establish credibility for later, bolder papers. HK-2 before HK-2B. P1 before P8.
5. **arXiv everything.** All major venues allow preprints. Post to arXiv the same day we submit to journals. This timestamps our claims.
6. **One thing at a time.** Solo researcher = serial execution on writing tasks. But submissions can overlap (papers are in review simultaneously).

---

## Phase 0: Foundation Week (Days 1-7)

*Goal: Get 2 verified papers out the door and fix critical infrastructure.*

> **POST-AUDIT UPDATE**: HK-1 removed from Phase 0. Deep audit (Feb 17) found critical integrity failures:
> claims 43 civilizations but only 4 validated, R²=0.991 not derivable, all figures from synthetic data.
> Phase 0 is now P1 + P4 only.

### Day 1: Submit P1 (Symthaea HAI) to PLoS Comp Bio

**Why first**: VERIFIED READY. Novel (no competing work). Honest claims. Our strongest paper.

- [x] Read hai_paper.tex end-to-end. Check for stale numbers:
  - LOC count: 341K verified via tokei (paper updated from 343K)
  - Test count: 2,738 main + 3,473 core verified (paper updated)
  - Feature count: 49 verified (paper updated from 48)
  - Workspace members: 24 verified (paper updated from 22)
  - Sub-crates: 22 verified (paper updated from 15)
- [x] Verify all 6 figures render correctly in PDF (all 6 present)
- [x] Cover letter updated (341K LOC, 6,750+ tests, 24 crates)
- [x] Compiled final PDF with pdflatex+bibtex — 19pp, zero undefined refs
- [ ] Post to arXiv (cs.AI + q-bio.NC cross-list)
- [ ] Submit to PLoS Computational Biology via their editorial manager
- [ ] Record submission date, confirmation number, and manuscript ID in PAPERS.md

**Quality gate**: PASSED. All figures render. All numbers verified against codebase. Clean compile.

### Day 2: Submit P4 (K-Index Unified Framework) to Neural Computation

**Why second**: 9 blocking defects found and FIXED on Feb 17. Now clean.

- [x] Read manuscript.tex end-to-end — found 9 blocking defects:
  1. No figures embedded (added 6 \includegraphics environments)
  2. Fabricated bib entry removed (replaced with real VanRullen 2021)
  3. Placeholder arXiv ID removed (replaced with real Butlin 2023)
  4. Only 3/21 citations wired (now 18 citations wired)
  5. Test count inflated 46→36 (corrected throughout)
  6. No individual author (added "Tristan Stoltz")
  7. GitHub repo placeholder fixed
  8. Symthaea conflict of interest disclosed
  9. No Related Work section (citations now cover this)
- [x] Cover letter: EIC "Dr. Sejnowski", author, test counts, disclosure — all fixed
- [x] Supplementary: Table S3 test counts corrected, author/date updated
- [x] All 3 documents compiled clean
- [ ] Verify 6 figure files are publication quality (300 DPI)
- [ ] Submit to Neural Computation
- [ ] Post to arXiv (q-bio.NC)
- [ ] Record in PAPERS.md

**Quality gate**: PASSED (compilation). Figures need DPI check before submission.

### ~~Day 3-4: Submit HK-1 (Historical K-Index) to Nature~~ REMOVED

> **INTEGRITY FAILURE** (Feb 17 audit):
> - Claims 43 civilizations but only 4 in codebase
> - R²=0.991 not derivable from any validation output (actual errors 52-66%)
> - All 4 figures generated from SYNTHETIC DATA (seed=42)
> - 48 supplementary figures don't exist
> - "DRAFT VERSION 1.3" still in manuscript
>
> **Required before resubmission**: Build real analysis pipeline for all 43 civilizations,
> generate figures from actual data, verify R² against real outputs. Estimated: 4-8 weeks.
> Moved to Phase 3.

### Day 5-7: Critical Infrastructure

**Phi Metric Audit (BLOCKS P7 and P12)**:
- [ ] Read `phi-lab/src/hdc/phi_real.rs` line by line — document exactly what it computes
- [ ] Read `phi-lab/src/hdc/tiered_phi.rs` — document what the true IIT implementation does
- [ ] Check if P7 (Master Equation) uses `phi_real.rs` or `tiered_phi.rs` or something else
- [ ] Write `METRIC_VERIFICATION.md` at repo root documenting:
  - Which metric each paper actually uses
  - Whether claims match computation
  - Recommended reframing for affected papers
- [ ] Decision: Can P7 be reframed to use honest terminology? Or does it need fundamental rework?

**arXiv Account Setup** (if not already done):
- [ ] Ensure arXiv account exists and is endorsed for cs.AI, q-bio.NC, physics.soc-ph categories
- [ ] Test with a submission (P1 is ideal first arXiv post)

**Submission Tracker Setup**:
- [ ] Create a simple tracking table in PAPERS.md with columns: Paper, Venue, Submitted, Status, Manuscript ID

---

## Phase 1: Quick Wins (Week 2-3)

*Goal: Submit 5 more papers. Build momentum. All are ready or nearly ready.*

### Week 2: Security Paper + Kosmic Quick Wins

**P2 (Zero-TrustML) → Target: ACM CCS 2026 Cycle 2 (April 29 deadline)**

ACM CCS is 10 weeks away — much better fit than missed USENIX deadline.

- [ ] Full proofread of all 8 section .tex files
- [ ] Anonymize: remove author names, affiliation, self-citations, acknowledgments
- [ ] Verify `stats.tex` generates correctly (or hardcode values)
- [ ] Check against ACM CCS formatting requirements (ACM template)
- [ ] Ensure all 3 figures are embedded (not external references)
- [ ] Run figure generation scripts to verify reproducibility
- [ ] Compile final anonymous PDF
- [ ] Check page count (ACM CCS allows 12 pages + unlimited references)
- [ ] Submit by April 15 (2 weeks buffer before April 29 deadline)
- [ ] Post non-anonymous version to arXiv simultaneously

**Quality gate**: Anonymous. Compiles cleanly. Figures reproduce. Under page limit.

**P3 (Coherence-Guided Control) → PLoS Comp Bio**

- [ ] Locate all figures referenced via `../logs/` paths
- [ ] Copy figures to `kosmic-lab/papers/paper2/figures/` as PDF + PNG
- [ ] Update all `\includegraphics` paths in manuscript.tex
- [ ] Recompile PDF and verify all figures render
- [ ] Quick proofread (this paper has strong stats — just verify formatting)
- [ ] Submit to PLoS Comp Bio
- [ ] Post to arXiv
- [ ] Record in PAPERS.md

**P10 (Topology of Collective) → Frontiers in Computational Neuroscience**
**P11 (Developmental Pathway) → Neural Networks**

- [ ] For each: verify bibliography is customized (not shared 6.4KB generic bib)
- [ ] For each: check that all `\cite{}` commands resolve
- [ ] For each: quick proofread (both are compact, 4-5 pages)
- [ ] Submit P10 to Frontiers
- [ ] Submit P11 to Neural Networks
- [ ] Both to arXiv

### Week 3: More Kosmic Papers

**P9 (Coherence Corridors) → PLoS ONE**

- [ ] Wire `\cite{}` commands throughout manuscript (currently 0 citations despite bib file)
- [ ] Fix author field (currently "[Authors TBD]")
- [ ] Add 5-10 relevant citations from existing bibliography
- [ ] Quick proofread
- [ ] Submit to PLoS ONE (lower barrier, establishes corridor methodology)

**End of Phase 1 Milestone**: 8 papers submitted across 6+ venues. arXiv preprints establishing priority.

---

## Phase 2: The K-Index Campaign (Week 3-5)

*Goal: Launch the Historical K-Index series — but only after HK-1 integrity issues are resolved.*

> **POST-AUDIT UPDATE**: HK-1 was NOT submitted. The entire HK series depends on building a real
> analysis pipeline first. HK-2 may still be independently submittable if its data is verified.

### Week 3-4: Verify HK-2 Independence from HK-1

**Before submitting any HK paper**: Verify that HK-2's data/figures are from real analysis (not synthetic).

- [ ] Deep audit HK-2 (same methodology as HK-1 audit)
- [ ] Verify 39-civilization dataset against source data (Seshat, HYDE, World Bank)
- [ ] Spot-check 5 collapse threshold predictions against historical dates
- [ ] Verify leave-one-out cross-validation results (theta = 0.375 +/- 0.004)
- [ ] **CRITICAL**: Verify figures are from real analysis, not synthetic generation
- [ ] If verified: Submit to PNAS + arXiv (physics.soc-ph + nlin.AO)
- [ ] If integrity issues found: Add to rework queue with HK-1

**Quality gate**: All data points traceable to source datasets. No synthetic figures.

### Week 4-5: Begin HK-1 Rework (if capacity allows)

- [ ] Build real analysis pipeline for all 43 civilizations
- [ ] Generate figures from actual data (not seed=42 synthetics)
- [ ] Verify R² claim against real outputs
- [ ] Remove "DRAFT VERSION 1.3" from manuscript
- [ ] Honest reassessment: Is R²=0.991 achievable with real data?

---

## Phase 3: High-Impact Writing Sprint (Week 4-8)

*Goal: Write the manuscripts for our highest-impact unwritten findings.*

### P19 (K_Topo: Operational Closure in LLMs) — NEEDS COMPLETE REDO

> **POST-AUDIT UPDATE (Feb 17)**: The "140x divergence" is METHODOLOGICALLY UNRELIABLE.
> Stress test found 7 compounding failures:
> 1. Different K_Topo formulas used for GPT-4o vs Claude
> 2. Different conversation lengths (40 vs 20 turns)
> 3. Asymmetric error exclusion
> 4. Metric fails own validation (drift > recursive)
> 5. Human baseline inflated by crash exclusion
> 6. No stored GPT-4o results file
> 7. Paper formula != code formula
>
> **Verdict**: Not fabrication, but deeply unreliable. The finding may be real but cannot be
> published based on current data. Needs complete methodological redo.

**Week 4: Methodological Redesign**
- [ ] Design consistent methodology: SAME formula, SAME conversation length, SAME error handling for ALL models
- [ ] Write methodology spec document BEFORE running any experiments
- [ ] Get methodology peer-reviewed (even informally) before generating data

**Week 5: Clean Reproducibility Run**
- [ ] Run K_Topo on GPT-4o with new consistent methodology
- [ ] Run on Claude Sonnet with IDENTICAL methodology
- [ ] Run on 2-3 additional models (Llama 3, Gemini, Mistral)
- [ ] Run on different prompts/tasks — robustness check
- [ ] Store ALL raw results with timestamps and full logging
- [ ] If divergence REPRODUCES with clean methodology: proceed to manuscript
- [ ] If divergence DISAPPEARS: honest retraction of draft claims, shelve paper

**Week 6-7: Manuscript Writing (only if finding reproduces)**
- [ ] Title: "Operational Closure is Architecture-Specific: Evidence from Frontier Language Models"
- [ ] Explicitly document the methodological failures of the original analysis
- [ ] Present clean results only
- [ ] Target: arXiv preprint first, then Nature Machine Intelligence

**Quality gate**: Consistent methodology across all models. Finding reproduces. All raw data stored and verifiable.

### P18 (Learned HDC for Genomics) — Week 6-7

**Why now**: Benchmarks complete (94.5% accuracy), code in production (Rust), no competing work. Just needs a manuscript.

- [ ] Structure: Background (HDC for bioinformatics), Method (contrastive pre-training + fine-tuning pipeline), Results (comparison table vs DNABERT, baseline HDC), Discussion (edge computing implications)
- [ ] Generate 3-4 figures: accuracy comparison bar chart, training curves, speed/accuracy tradeoff Pareto front, architecture diagram
- [ ] Benchmark table: all 5 datasets (MNIST, E. coli, splice sites, TATA-box, etc.)
- [ ] Write ~3,500-4,000 words
- [ ] Target: Bioinformatics or BMC Genomics

### P14 (Kosmic K-Vector 7D) — Week 7-8

- [ ] Run Phase 1 experiments from EXPERIMENTAL_DESIGN.md
- [ ] Verify "Commons Paradox" result (r=-0.44, p<0.001) reproduces
- [ ] Polish manuscript (already 90% framework)
- [ ] Generate figures from `generate_figures.py`
- [ ] Target: Neuroscience of Consciousness

---

## Phase 4: The Honest Reframe (Week 6-8, parallel with Phase 3)

*Goal: Address the Phi metric issue head-on.*

### P12 (Temporal Topology) → Reframe + Submit

- [ ] Use `letter_reframed.md` as the basis for honest reframing
- [ ] Systematically replace all "Phi" / "IIT" / "integrated information" with "lambda2" / "algebraic connectivity" / "spectral topology"
- [ ] Rewrite abstract: focus on "spectral connectivity predicts effective network topology in continuous-time neural architectures"
- [ ] Update all figure captions
- [ ] Remove Tononi/IIT references. Add spectral graph theory references (Chung, Spielman)
- [ ] The core finding (99.2% optimal 3D small-world, lambda2 → 0.5 asymptotically) is STILL NOVEL and publishable
- [ ] Target: Network Neuroscience or Journal of Complex Networks
- [ ] Post to arXiv (cs.NE + math.SP)

### P7 (Master Equation) → Depends on Audit

**If audit confirms lambda2 (likely)**:
- [ ] Reframe C(t) as composite spectral-cognitive metric (not IIT)
- [ ] This significantly changes the paper's framing but not its core contribution
- [ ] Target: PLoS Comp Bio (instead of Nature Neuroscience)
- [ ] Complete the 8 figures and 60 references
- [ ] Submit after reframing

**If audit reveals actual IIT Phi is used**:
- [ ] Proceed with Nature Neuroscience as planned
- [ ] Complete figures and references
- [ ] Submit

---

## Phase 5: The Second Wave (Week 8-12)

*Goal: Submit the remaining high-value papers.*

### Week 8-9: Conference Targeting

**P8 (HAI NeurIPS variant)**:
- If NeurIPS 2026 deadline is ~May 15 (based on 2025 pattern):
  - [ ] Sync results from P1 (hai_paper.tex) into hai_neurips2026.tex
  - [ ] Condense to NeurIPS 8-page format
  - [ ] Add all 6 figures
  - [ ] Submit to NeurIPS by deadline
  - NOTE: NeurIPS prohibits concurrent archival submissions. P1 must be published or withdrawn from PLoS before NeurIPS submission. **If P1 is still in review at PLoS, DO NOT submit P8 to NeurIPS.** Wait for PLoS decision.

**P2 (Zero-TrustML) → ACM CCS (if not submitted in Week 2)**:
- Deadline: April 29
- Follow Phase 1 instructions

**P16 (PoGQ) → ICML 2026 or standalone submission**:
- If ICML 2026 deadline is ~April: very tight, likely defer
- Alternative: Submit to IEEE TDSC or ACM TOPS as journal paper (rolling deadline)
- [ ] Complete Section 3 (threat model + protocol + security proofs)
- [ ] Write Sections 1, 2, 4, 5, 6 from outline
- [ ] Generate benchmark figures from Grand Slam results

### Week 9-10: Historical K-Index Continuation

**HK-2B (Golden Threshold) → Physical Review Letters**
- [ ] Submit ONLY after HK-2 is submitted (strategic sequencing to avoid "numerology" perception)
- [ ] Verify all 9 derivations are mathematically rigorous
- [ ] Monte Carlo validation: re-run 10^5 samples
- [ ] PRL format: ~3,500 words max
- [ ] Include honest discussion of why phi^-2 might be coincidence vs deep structure

**HK-9 (Coordination Contagion) and HK-10 (Micro-K Framework)**:
- Both have substantial manuscripts but zero figures
- [ ] Generate figures for HK-9 (network diffusion maps, super-spreader identification)
- [ ] Generate figures for HK-10 (organizational K-scores, city comparisons)
- [ ] Target: Nature Human Behaviour (HK-9), Organization Science (HK-10)

### Week 10-12: Completion + Polish

**P6 (Phenomenal Signature) → arXiv preprint**:
- [ ] Complete results section with statistical tables
- [ ] Generate topology heatmap figures
- [ ] Write discussion and conclusion
- [ ] Post to arXiv as preprint (lower bar, establishes priority)

**P5 (Mycelix MLSys) → Journal submission**:
- MLSys 2026 conference deadline has passed. Retarget to journal:
- [ ] Expand sections 4-7
- [ ] Generate 4-6 figures
- [ ] Target: IEEE TPDS or Journal of Machine Learning Research (Systems track)

**P15 (Unified Indices)**:
- [ ] Integrate correction (r=+0.61 bug → r=-0.21 actual)
- [ ] Add forward reference to P14 (K-Vector 7D)
- [ ] Recompile and submit to PLoS ONE

---

## Phase 6: Long-Term Pipeline (Month 4-6)

### Historical K-Index Continuation
- HK-3 (Modern Fragility) → Science or Nature (after HK-1 response)
- HK-2D (Capacity Gap) → PNAS
- HK-11 (Modernization Paradox) → Complexity
- HK-12 (Fermi Paradox) → Astrobiology

### Phi-Lab Satellites
- Only pursue after P7 is published (all 15 depend on it)
- Priority: Paper 02 (AI Consciousness) → Nature Machine Intelligence
- Priority: Paper 12 (Computational Implementation) → PLoS Comp Bio

### Whitepapers
- P17 (MATL) → IEEE TDSC (after P16/PoGQ accepted)
- PoGQ whitepaper → ACM TOPS if conference deadlines missed

### Co-Author Strategy
- Contact Alexander Wendt (Ohio State) about HK papers — his quantum social science framework aligns perfectly
- Contact Peter Turchin (Complexity Science Hub) — cliodynamics expertise validates our approach
- Emails are already drafted in `.archive.../submissions/strategy/CO_AUTHOR_OUTREACH_EMAILS.md`
- Timing: reach out AFTER first acceptance (strengthens the ask)

### Grants
- NSF SBIR: SAM.gov registration was the blocker. Complete registration, then resubmit in next cycle.
- Protocol Labs: Finish draft, submit when ready. Not urgent.

---

## Milestone Summary (Revised Post-Audit)

| Week | Papers Submitted (cumulative) | Key Event |
|------|-------------------------------|-----------|
| 1 | 2 (P1, P4) | First submissions ever (verified packages) |
| 2 | 4-5 (+ P2 prep, P3, P10) | arXiv preprints live |
| 3 | 6-7 (+ P11, P9) | Quick wins complete |
| 4 | 7-8 (+ HK-2 if verified) | K-Index campaign starts (contingent on HK-2 audit) |
| 5 | 7-8 | P19 methodology redesign. HK-1 rework begins. |
| 6 | 8-9 (+ P12 reframed) | Spectral topology reframe |
| 7 | 9-10 (+ P18) | Genomics HDC paper |
| 8 | 10-12 (+ P14, P15, P19 if clean) | K-Vector + K_Topo (only if methodology passes) |
| 10 | 12-14 (+ HK-2B, P8 or P16) | Conference submissions |
| 12 | 14-16 (+ HK-9, HK-10, P6) | Second wave complete |

**Conservative target: 12 papers submitted in 12 weeks** (revised from 15 — honest assessment)
**Optimistic target: 16 papers submitted in 12 weeks** (revised from 19)

---

## Risk Registry (Updated Post-Audit)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **HK series has more integrity issues** | **60%** | **Critical** | Deep audit EVERY HK paper before submission. Verify data against source datasets. |
| **P19 K_Topo finding doesn't reproduce with clean methodology** | **50%** | High | Redesign methodology first. Only write manuscript if finding holds. |
| **Other "ready" papers have hidden defects (like P4's 9)** | **40%** | High | Verify P2, P3 with same rigor as P1/P4 before submission. |
| Phi metric affects P7 more deeply than expected | 60% | Medium | Reframe as spectral-cognitive metric, retarget venue |
| Solo authorship reduces acceptance | 40% | Medium | Build track record first, add co-authors for HK series |
| Reviewer pushback on K-Index extraordinary claims | 70% | Medium | Honest limitations, robust methodology, co-author credibility |
| Burnout from simultaneous papers in review | 50% | High | Batch submissions, then rest. Reviewer timelines create natural breaks. |
| ACM CCS deadline too tight for P2 | 20% | Low | Submit to IEEE TDSC instead (rolling) |
| arXiv endorsement delay | 15% | Low | Submit endorsement request in Week 1 |

---

## The Non-Negotiables

1. **Never claim IIT Phi when the code computes lambda2.** Reframe honestly or don't publish.
2. **Deep audit EVERY paper before submission.** The HK-1 and P4 audits proved that "ready" packages can have critical defects. No exceptions.
3. **Never submit synthetic data as real.** If figures come from `seed=42` generation scripts, they are illustrations, not results.
4. **Verify P19 with clean methodology before writing.** The original data is unreliable. Start fresh.
5. **Don't submit P8 to NeurIPS if P1 is still under review at PLoS.** Dual submission policies exist for a reason.
6. **Spot-check data in every paper before submission.** At least 3 claims per paper verified against source data.
7. **Update PAPERS.md submission tracker after every submission.** Future us will thank present us.

---

## What Success Looks Like

**3 months**: 15+ papers submitted. 3-5 arXiv preprints generating citations. First reviewer feedback arriving.

**6 months**: 3-5 papers accepted. Track record established. Co-author conversations started for HK series.

**12 months**: 8-12 papers published across Nature/PNAS/PLoS/Neural Computation/conferences. K-Index series gaining traction. Symthaea HAI establishing the HDC+FEP field. K_Topo result (if it holds) making waves in AI consciousness.

**The goal isn't 52 published papers. It's 15-20 excellent, honest, impactful ones.**

---

*The best time to submit was 7 weeks ago. The second best time is today.*
