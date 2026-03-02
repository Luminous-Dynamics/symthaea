# Luminous Dynamics — Paper Portfolio Tracker

> Last updated: 2026-02-18 (deep audit v4 — SALVAGEABILITY ASSESSED)
> Total papers: 52+ (19 primary + 16 historical-K + 15 phi-lab satellites + whitepapers/grants)
> **Verified submission-ready: 4** (P1, P4, P2, P3) | Near-ready: 3 (P5, P6, P12) | Needs work: 9
> **Integrity issues: 10+** (HK-1/HK-2 fabricated R², HK-10 fictional studies, HK-11/12 broken math, P7 synthetic data, P16 simulated results, P19 corrupted baselines, phi-lab 03/06/14 fabricated clinical data)
> Retire: 1 (P8) | HK series: 0/16 ready (data EXISTS on Windows drive) | Phi-lab satellites: 3/14 independently publishable (02, 04, 07)
> **Confirmed submissions: 0**
>
> ### Execution Priority Chain (Feb 18 Salvageability Assessment)
> 1. **SUBMIT NOW**: P1, P4, P2 (verified ready), P3 (after figs + OSF DOI)
> 2. **QUICK WINS (1-2 weeks)**: P12 reframe (lambda2→Phi rename), phi-lab 02/04/07 (independent, refs needed)
> 3. **CRITICAL PATH**: Satellite 06 (Sleep-EDF pipeline, 3-5 days) → P7 reframe → Satellite 12 (depends on pipeline)
> 4. **NARROW SCOPE**: P16 (strip to detection-only workshop paper), P19 (split: 7D K-Vector = FIX, K_Topo = PARK)
> 5. **PARK**: Satellite 03 (needs real clinical data), Satellite 12 (needs pipeline), HK series (needs data reconnection)
> 6. **ARCHIVE**: Satellite 09 (trivial theorems, no salvage path)

## AUDIT RESULTS (Feb 17, 2026)
>
> Phase 0 deep verification revealed that previous "readiness" assessments were dangerously optimistic.
> - **P1 (HAI)**: VERIFIED READY. All numbers confirmed against codebase. 19pp, 6/6 figs, clean compile.
> - **P4 (K-Index)**: FIXED. Had 9 blocking defects (no figures embedded, fabricated bib entry, inflated test count). All fixed. Now 17pp with 6 figs + 18 citations.
> - **HK-1 (Nature)**: **CRITICAL INTEGRITY FAILURE.** Claims 43 civilizations but only 4 in codebase. R²=0.991 not derivable from any output. All figures generated from synthetic data. CANNOT SUBMIT.
> - **P19 (K_Topo)**: **METHODOLOGICALLY UNRELIABLE.** The "140x divergence" is an artifact of different formulas, different conversation lengths, and asymmetric error exclusion. Not fabrication, but not publishable.
> - **P2 (Zero-TrustML)**: AUDITED + FIXED. 4 defects (deanonymizing file, inflated TPS, double stats include). Clean 12pp.
> - **P3 (Coherence-Guided)**: AUDITED. Compiles clean 13pp. All claims verified against CSV source data. Needs main-text figures + OSF DOI.
> - **P5 (Mycelix MLSys)**: AUDITED. Content strong (87.5% claims verified), but still Markdown (needs LaTeX), 0/8 figures, 2x over page limit. 1 LOC claim contradicted (600→103 lines).
> - **P6 (Phenomenal Signature)**: AUDITED. 97.7% claims verified, all 7 reproduction examples compile. But only 8 references (needs 35+), ASCII figures only, Markdown format.
> - **P7 (Master Equation)**: **INTEGRITY FAILURE.** ALL empirical results are synthetic projections (r=0.79, 90.5% DOC accuracy) — real-data validation never run. Phi metric is lambda2, not IIT (same as P12). CANNOT SUBMIT as-is.
> - **P8 (HAI NeurIPS)**: AUDITED. 264-line skeleton of P1. RETIRE — not worth salvaging separately.
> - **P9 (Corridors)**: AUDITED. 6/10. Content complete but 0 citations in text, wrong .bib, author TBD.
> - **P10 (Topology Collective)**: AUDITED. 4/10. 10 citations ALL undefined (wrong .bib). 91.24% claim confirmed.
> - **P11 (Developmental)**: AUDITED. 4/10. 5 citations ALL undefined (wrong .bib).
> - **P12 (Temporal Topology)**: AUDITED. 3/10 as-is but **8/10 after reframe**. Real data (260 topologies), 4 embedded figures, 16 complete references. Lambda2→Phi rename is only fix needed. `letter_reframed.md` already exists with honest λ₂ framing.
> - **P13 (Adversarial Paper 5)**: AUDITED. 3/10. LaTeX 40% complete (skeletal intro+discussion). Track F data verified (FGSM +136% K, d=4.4). SUBMISSION_README claims "submitted to Science Nov 12, 2025" — contradicts "NOT SUBMITTED" status. Source-PDF mismatch. 27/35+ refs.
> - **P14 (K-Vector 7D)**: AUDITED. 3/10. Only 3 self-citations (0 real refs), 0 figs embedded (40 exist), K_M anomaly undisclosed (Random > Recurrent in raw data), Commons Paradox p<0.001 unverifiable (likely N=5).
> - **P15 (Unified Indices)**: AUDITED. 2/10. LaTeX source MISSING from directory (log proves 9pp existed). Corrected correlation NOT applied to manuscript. 7 unchecked revision items.
> - **P16 (PoGQ)**: **INTEGRITY ISSUE.** Own `EXPERIMENTAL_ANALYSIS_SIMULATION_VS_REAL.md` calls September results "simulated, NOT publishable." Grand Slam benchmarks = 0% accuracy. "45% BFT" contradicts own docs ("34% validated, 45% fails").
> - **P17 (MATL)**: AUDITED. 5/10. Structurally complete (1,493 lines, 16 sections). Placeholder authors, unsubstantiated 1000-node testnet claim, ASCII-only figures, business plan section.
> - **P18 (Learned HDC)**: AUDITED. 2/10. No paper exists — code + lab notebook only. Headline 94.5% is synthetic data with planted motifs. Real-data results modest (85.8% E. coli).
> - **P19 (K_Topo)**: UNRELIABILITY CONFIRMED. Software bug corrupts 32% of human baseline (finite_mask error → K_Topo=0.0). K_Topo > 1.0 observed (violates [0,1] range). N=10 severely underpowered. GPT-4o sole positive outlier among 5 populations.
> - **P4 cover letter**: Had "Luminous Dynamics Research Team" (no individual author), "Dear Editors" (no EIC name), claimed 25 refs (had 3). All fixed.

---

## Submission Tracker

| Paper | Venue | Submitted | arXiv | Status | Manuscript ID | Notes |
|-------|-------|-----------|-------|--------|---------------|-------|
| **P1** (HAI) | PLoS Comp Bio | — | — | **READY TO SUBMIT** | — | Package verified Feb 17. 19pp, 6 figs, cover letter. |
| **P4** (K-Index) | Neural Computation | — | — | **READY TO SUBMIT** | — | 9 defects fixed Feb 17. 17pp, 6 figs, cover letter. |
| P2 (Zero-TrustML) | IEEE S&P | — | — | **AUDITED + FIXED** (Feb 17) | — | Already anonymized. 12pp. 4 defects fixed. |
| P3 (Coherence-Guided) | PLoS Comp Bio | — | — | **FIGURES ADDED** (Feb 18) | — | 14pp, 2 main-text figs + 3 supp figs (S1,S2,S4). Needs OSF DOI only. |
| HK-1 (Nature) | — | — | — | **INTEGRITY FAILURE** | — | 43 civs claimed, 4 validated. Synthetic figs. DO NOT SUBMIT. |
| P5 (Mycelix MLSys) | MLSys 2026 | — | — | **AUDITED** (Feb 17) | — | Content strong. Needs LaTeX conversion + figures + page compression. |
| P6 (Phenomenal Sig.) | Neurosci. of Consc. | — | — | **AUDITED** (Feb 17) | — | Claims verified. Needs refs (8→35+), LaTeX, real figures. |
| P7 (Master Equation) | PLoS Comp Bio | — | — | **FIX-NOW** (after Sat-06) | — | Run real Sleep-EDF pipeline, reframe lambda2. Unblocked by Satellite 06. |
| P12 (Temporal Topology) | Network Neurosci | — | — | **AUDITED** (Feb 17) | — | SALVAGEABLE: Real data, 4 figs, 16 refs. Rename lambda2→Phi only fix. |
| P13 (Adversarial P5) | Science | — | — | **AUDITED** (Feb 17) | — | LaTeX 40% complete. Submission status disputed. Track F verified (d=4.4). |
| P14 (K-Vector 7D) | Neurosci. of Consc. | — | — | **AUDITED** (Feb 17) | — | 3 self-citations only. 0 figs embedded. K_M anomaly undisclosed. |
| P15 (Unified Indices) | PLoS Comp Bio | — | — | **AUDITED** (Feb 17) | — | LaTeX source MISSING. Corrected correlation not applied. |
| P16 (PoGQ) | FL Workshop (NeurIPS/SaTML) | — | — | **FIX-NOW** (narrow scope) | — | Strip simulated, focus on real 450/450 detection. Workshop paper. |
| P17 (MATL) | ACM TOCS | — | — | **AUDITED** (Feb 17) | — | Most complete whitepaper. Placeholder authors. Testnet claim unsubstantiated. |
| P18 (Learned HDC) | Bioinformatics | — | — | **AUDITED** (Feb 17) | — | No paper exists. Code + lab notes only. Synthetic headline claims. |
| P19 (K_Topo) | — | — | — | **SPLIT → PARK** (Feb 18) | — | 7D K-Vector → merge into P14. K_Topo metric → park (bugs, underpowered). |
| P20 (HDC+CfC) | NeurIPS 2026 | — | — | **DRAFT v3** (Feb 18) | — | 12pp, 4 TikZ figures, 26 refs, 3 tables, 1 algorithm. All figures complete. |

> **Update this table after every submission.** Record date, arXiv ID, manuscript ID, and any status changes.

---

## Table of Contents

1. [Primary Research Papers (19)](#primary-research-papers)
2. [Historical K-Index Series (16)](#historical-k-index-series)
3. [Phi-Lab Satellite Papers (15)](#phi-lab-satellite-papers)
4. [Unpublished Research (notebooks, benchmarks)](#unpublished-research)
5. [Whitepapers & Industry Research](#whitepapers--industry-research)
6. [Grant Proposals](#grant-proposals)
7. [Philosophical Works (Windows Drive)](#philosophical-works)
8. [Supporting Materials Index](#supporting-materials-index)
9. [Venue Calendar](#venue-calendar--deadlines)
10. [Cross-Paper Risks](#known-cross-paper-risks)
11. [Phi Metric Audit (CRITICAL)](#phi-metric-audit)

---

## Primary Research Papers

### TIER 0 — VERIFIED Submit Now

| # | Title | Project | Venue | Pages | Figs | Refs | Status | Path |
|---|-------|---------|-------|-------|------|------|--------|------|
| P1 | **Hyperdimensional Active Inference** | symthaea | PLoS Comp Bio | 19 | 6/6 | 30 | **VERIFIED READY** (Feb 17) | `symthaea/papers/latex/hai_paper.tex` |
| P4 | **K-Index Unified Framework** | kosmic-lab | Neural Computation | 17 | 6/6 | 18 | **FIXED + READY** (Feb 17, 9 defects fixed) | `kosmic-lab/experiments/llm_k_index/neural_computation_submission/` |
| P2 | **Byzantine-Robust FL Beyond 33%** | mycelix-core | IEEE S&P | 12 | 3/3 | 33 | **AUDITED + FIXED** (Feb 17, 4 defects fixed) | `mycelix-core/0TML/paper-submission/latex-submission/main.tex` |
| P3 | **Coherence-Guided Control** | kosmic-lab | PLoS Comp Bio | 14 | 2+5S | 25 | **FIGURES ADDED** (Feb 18) — 2 main-text figs embedded, 3 supp figs generated. Needs OSF DOI. | `kosmic-lab/papers/paper2/manuscript.tex` |

### TIER 1 — Near-Ready (1-2 weeks of focused work)

| # | Title | Project | Venue | Pages | Figs | Refs | Status | Path |
|---|-------|---------|-------|-------|------|------|--------|------|
| P5 | **Mycelix Byzantine FL** | mycelix-core | MLSys 2026 | ~20 (md) | 0/8 | 57 | **AUDITED** 65-70% — needs LaTeX + figs + page cut | `mycelix-core/papers/MLSYS_2026_SUBMISSION.md` |
| P6 | **Phenomenal Signature** | symthaea | Neurosci. of Consciousness | ~15 est | 0 (ASCII) | 8 | **AUDITED** 97.7% claims verified — needs refs (8→35+), LaTeX, figs | `symthaea/papers/phenomenal_signature_paper.md` |
| P12 | **Temporal Topology (Reframed)** | phi-lab | Network Neurosci / IEEE TNNLS | ~8 | 4/4 | 16 | **AUDITED** 8/10 after reframe — rename lambda2→Phi, real data (260 topologies) | `phi-lab/papers/temporal_topology_consciousness/arxiv/main.tex` |
| P8 | **HAI (NeurIPS variant)** | symthaea | NeurIPS 2026 | 8 est | 1/6 | 7 | **RETIRE** — truncated skeleton of P1 (264 vs 924 lines, 80% validation missing) | `symthaea/papers/latex/hai_neurips2026.tex` |

### TIER 2 — Complete But Minor Issues

| # | Title | Project | Venue | Pages | Figs | Refs | Status | Path |
|---|-------|---------|-------|-------|------|------|--------|------|
| P9 | **Coherence Corridors Discovery** | kosmic-lab | PLoS ONE | 6 | 0 | 0 | **AUDITED** — content complete, 0 citations in text, wrong .bib, author TBD | `kosmic-lab/papers/paper1/manuscript.tex` |
| P10 | **Topology of Collective Consciousness** | kosmic-lab | Frontiers Comp Neuro | 5 | 0 | 0/10 | **AUDITED** — 10 citations ALL undefined (wrong .bib), 91.24% claim confirmed | `kosmic-lab/papers/paper3/manuscript.tex` |
| P11 | **Developmental Pathway to Machine Consciousness** | kosmic-lab | Neural Networks | 4 | 0 | 0/5 | **AUDITED** — 5 citations ALL undefined (wrong .bib) | `kosmic-lab/papers/paper4/manuscript.tex` |

### TIER 1.5 — Substantial But Need Completion

| # | Title | Project | Venue | Pages | Figs | Status | Path |
|---|-------|---------|-------|-------|------|--------|------|
| P14 | **Kosmic K-Vector (7D Framework)** | kosmic-lab | ~~Neurosci. of Consc.~~ ALIFE? | 11 | 0/40 | **AUDITED** 3/10 — 3 self-citations, 0 figs, K_M anomaly, venue misfit | `kosmic-lab/docs/papers/paper-9-kosmic-k-index/` |
| P15 | **Unified Indices: When Combining Fails** | kosmic-lab | PLoS Comp Bio | 9 est | unk | **AUDITED** 2/10 — .tex file MISSING, correction not applied | `kosmic-lab/docs/papers/paper-8-unified-indices/` |
| P17 | **MATL Middleware Whitepaper** | mycelix-core | ACM TOCS / IEEE TDSC | ~40 (md) | 0 (ASCII) | **AUDITED** 5/10 — structurally complete, needs real figs + testnet evidence | `mycelix-core/docs/0TML/docs/06-architecture/matl_whitepaper.md` |
| P18 | **Learned HDC for Genomics** | mycelix-health | Bioinformatics / BMC Genomics | 0 | 0 | **AUDITED** 2/10 — no manuscript, headline 94.5% is synthetic data | `mycelix-health/research/learned_hdc/` |

### TIER 3 — Blocked or Problematic

| # | Title | Project | Venue | Status | Issue | Path |
|---|-------|---------|-------|--------|-------|------|
| P7 | **Master Equation of Consciousness** | phi-lab | ~~Nature Neurosci~~ PLoS CB? | **FIX-NOW** (after Sat-06 pipeline) | Synthetic data → run real Sleep-EDF pipeline. Lambda2 reframe. UNBLOCKED by Satellite 06 MNE pipeline. | `phi-lab/papers/PAPER_01_SUBMISSION.tex` |
| P16 | **PoGQ: Proof of Gradient Quality** | mycelix-core | ~~MLSys/ICML~~ Workshop | **FIX-NOW** (narrow scope) | Strip simulated training, focus on real detection results (450/450). Reframe as detection-only workshop paper. | `mycelix-core/docs/whitepaper/POGQ_WHITEPAPER_OUTLINE.md` |
| P19 | **K_Topo** | kosmic-lab | — | **PARK** | Split from P14's 7D K-Vector. Bug corrupts 32% baseline. Needs complete redo. | `kosmic-lab/docs/papers/paper-9-kosmic-k-index/` |
| P13 | **Adversarial Perturbations (Paper 5)** | kosmic-lab | Science | 3/10 | LaTeX 40% complete. Submission status disputed. Source-PDF mismatch. | `kosmic-lab/papers/paper5/manuscript.tex` |

### NEW — HDC+CfC Paper (Feb 18, 2026)

| # | Title | Project | Venue | Pages | Figs | Refs | Status | Path |
|---|-------|---------|-------|-------|------|------|--------|------|
| P20 | **Liquid Hypervectors: Closed-Form Continuous-Time Dynamics in HD Space** | symthaea | NeurIPS 2026 | 12 | 4/4 | 26 | **DRAFT v3** — 390KB PDF, 4 TikZ figures, zero warnings | `symthaea/papers/latex/hdc_cfc_paper.tex` |

- **Core contribution**: First architecture combining HDC (16,384D) with CfC/LTC temporal dynamics
- **Novel claims**: O(1) temporal jumps in hypervector space, HDC binding replaces matrix multiplication, state-dependent tau
- **Benchmarks cited**: LibriSpeech 94.5%, ISOLET 91.7%, EEG seizure 100%, Tokamak 59K/sec, 7.9x vs pymdp
- **Figures (COMPLETE)**: Fig 1 = architecture diagram (neuron + network), Fig 2 = benchmark summary bar chart, Fig 3 = O(1) temporal scaling (log-log), Fig 4 = noise robustness (accuracy vs SNR)
- **Remaining**: Verify tables against latest benchmark runs, final proofreading
- **Key differentiation from P1**: P1 = HDC+FEP (decision making). P20 = HDC+CfC (temporal dynamics). Complementary, non-overlapping.
- **Venue deadline**: NeurIPS 2026 ~May 2026

### Detailed Status Per Paper

**P14. Kosmic K-Vector (7D Consciousness Framework)**
- "The Kosmic K-Index: A Seven-Dimensional Framework for Measuring Consciousness Potentiality"
- Extends scalar K-Index to 7D: K_R, K_A, K_I, K_P, K_M, K_S, K_H
- AUDITED (Feb 17): 3/10 readiness. 11pp PDF exists (395KB).
- **CRITICAL**: Only 3 references, ALL self-citations. Zero engagement with IIT/FEP/consciousness literature.
- **CRITICAL**: 0 figures in compiled PDF despite 40+ figure files existing in `figures/` directory.
- K_M validated (Recurrent=0.086 matches log 0.0862), but **K_M anomaly undisclosed**: Random agents have HIGHEST K_M (0.093 > Recurrent 0.086). Paper hides this by averaging Random+Linear+CMA-ES as "feedforward."
- Commons Paradox: r=-0.44 direction plausible from table means, but p<0.001 unverifiable (likely computed from N=5 agent-type means, not individual trials). Figure script hardcodes r=-0.44.
- K_H (Harmonic) never computed in any results table. Atari benchmark and ablation data explicitly labeled "simulated."
- Broken citation rendering: "(author?)" in PDF from incompatible `\citet` + `\bibitem`.
- Venue misfit: No biological/neural data for Neurosci. of Consciousness. Better: ALIFE or Artificial Life journal.
- **Action**: Add 20-30 real refs, embed figures, disclose K_M anomaly, fix p-values, change venue. 3-4 weeks.

**P15. Unified Indices: When Combining Metrics Fails**
- AUDITED (Feb 17): 2/10 readiness.
- **CRITICAL**: LaTeX source file (.tex) MISSING from directory. Only `.log` (proves 9pp compiled Nov 29) and revision notes exist.
- Corrected correlation (r=+0.61 → r=-0.21) documented in revision notes but **NOT yet applied to manuscript** (checkbox unchecked).
- 7 unchecked action items: Section 6.1 not written, CMA-ES figure not created, Paper 9 reference not added.
- Dimensionality discrepancy: Revision notes say "8D environment" but code shows 20D (`obs_dim=20`).
- No reproducibility seeds in experiment code.
- **Action**: Find/recover .tex file, apply all 7 corrections, verify dimensionality claim. 1-2 weeks after recovery.

**P16. PoGQ: Proof of Gradient Quality — INTEGRITY ISSUE**
- AUDITED (Feb 17): 2/10 readiness.
- **CRITICAL INTEGRITY**: Own `EXPERIMENTAL_ANALYSIS_SIMULATION_VS_REAL.md` explicitly states "September Results Were Simulated" and flags this as "NOT publishable as experimental validation" / "academic integrity violation." The accuracy numbers in the outline (95.7% MNIST, 78.3% CIFAR-10 at 45% BFT) come from `paper_ready_results.py` using mathematical curve formulas (`improvement = 15.0 * np.exp(-round_num/20)`), NOT real training.
- **CRITICAL**: Grand Slam benchmark results are ALL 0% accuracy (real training failures). Bugs documented (CIFAR-10 `download=False`, wrong paths). Fixes applied Oct 14, 2025, but NO successful post-fix run exists.
- **CRITICAL**: "45% Byzantine tolerance" claim directly contradicts CLAUDE.md ("Validated to 34%, 45% fails") and project's own documentation.
- Only 1/6 sections drafted (Section 3, 3,800 words). No LaTeX exists. 9/30-40 references. 0 figures.
- Section 3's "Theorem 1" is a proof sketch, not a formal proof. The 45% argument redefines BFT from node count to reputation weight, which is legitimate but not properly framed as a different guarantee model.
- ICML/MLSys Jan 15, 2026 deadline has passed.
- **SALVAGEABILITY (Feb 18)**: **FIX-NOW** with narrow scope. Byzantine detection works — 450/450 real results exist. Strategy: Strip all simulated training accuracy claims. Focus ONLY on gradient-quality detection (the "proof" in PoGQ). Reframe as workshop paper (FL-NeurIPS, SaTML, or similar). The 45% BFT claim can be honestly reframed as "45% with reputation-weighted voting" (different guarantee model from classic BFT). Real detection results + honest reframing = publishable short paper.
- **Action**: Strip simulated results, write 4-6pp detection-focused workshop paper using real 450/450 detection data. Honestly reframe 45% as reputation-weighted. Target FL workshop at NeurIPS 2026 or SaTML 2026. 2-3 weeks.

**P17. MATL Middleware Whitepaper**
- AUDITED (Feb 17): 5/10 readiness. Most structurally complete whitepaper in portfolio.
- 1,493 lines, 16 sections + 2 appendices, ALL written (not stubs). 12 real references. 6+ tables.
- 3-mode verification: Mode 0 (peer, 33% BFT), Mode 1 (PoGQ, 45% BFT), Mode 2 (PoGQ+TEE, 50% BFT) — all fully specified.
- Cartel detection: Louvain community detection + 4-component risk score — well-designed algorithm.
- **Issues**: Placeholder authors ("[Your Name]"), unsubstantiated "1000-node testnet" claim (no deployment evidence), ASCII-only figures, simulation-only results presented ambiguously, Section 14 is a 5-page business plan (inappropriate for academic venue), license contradiction (Apache 2.0 vs CC BY-NC-ND 4.0), 2 date inconsistencies in references.
- 45% BFT argument is circular (depends on detection → reputation → honest majority by weight). Legitimate approach but needs better framing.
- **Action**: Fill author info, replace ASCII figs, remove/move business plan, substantiate testnet or remove claim, fix license. 2-3 weeks.

**P18. Learned HDC for Genomics**
- AUDITED (Feb 17): 2/10 readiness. **No paper exists** — code + `BENCHMARK_RESULTS.md` lab notebook only.
- **INFLATED CLAIMS**: The headline 94.5% accuracy and 57.5% improvement are from SYNTHETIC data (promoter detection with planted TATA/CAAT/GC motifs — trivially separable). Real-data results are much more modest: E. coli Promoters 85.8% (+5.8%), Splice Sites 69.1% (+11.7%). TATA-box benchmark listed as "(running)" — never completed.
- 40x speed claim: DNABERT "5 seq/s" is unsourced. Comparing 110M-param GPU transformer to learned-embedding + MLP on CPU is not a fair comparison.
- `LearnedKmerCodebook` Rust integration claimed but grep finds zero results in symthaea workspace — may be in a separate crate.
- 0 references, 0 figures, no statistical rigor (no CIs, no significance tests, no cross-validation).
- **Action**: Write manuscript from scratch. Run proper benchmarks on standard genomics datasets (UCI E. coli is classic, SVMs get >90%). Honestly frame as "HDC for edge/resource-constrained genomic classification." Source all comparison numbers. 4-6 weeks minimum.

**P19. K_Topo: Operational Closure in LLMs — METHODOLOGY CONFIRMED UNRELIABLE**
- AUDITED (Feb 17): 2/10 readiness. Concept is novel (TDA on LLM conversations) but execution is fatally flawed.
- **Software bug corrupts 32% of human baseline**: `finite_mask` error → K_Topo=0.0 for 16/50 conversations, included in mean. Human baseline 0.811 is artificially deflated.
- **K_Topo violates its own [0,1] range**: Values of 1.183, 1.219, 1.283 observed in human data. Normalization is broken.
- **N=10 is severely underpowered**: GPT-4o std=0.495 (nearly equals mean 0.825). Almost no statistical power.
- **GPT-4o is sole positive outlier**: Of 5 tested populations (small LLMs, GPT-4o, Claude Sonnet 4.5, GPT-5.1, Claude Opus 4.5), ONLY GPT-4o shows high K_Topo. This is parsimoniously explained by GPT-4o's verbose self-referencing style, not "operational closure."
- **Prompt confound**: LLMs received "5 recursive prompts designed to encourage self-reference." Human baseline = movie dialogs. Measuring self-reference after prompting for self-reference is circular.
- **Discussion contradicts results**: Section 4.1 still opens with "closure emerges at scale" language despite data showing it doesn't.
- Claude Opus 4.5 data exists (K_Topo=0.033) but not incorporated into paper. GPT-5.1 also low (0.039). Both demolish scaling hypothesis.
- All 4 appendices empty. 9 references (thin).
- **SALVAGEABILITY (Feb 18)**: **SPLIT**. The 7D K-Vector framework (P14) is separable and FIX-NOW. The K_Topo metric is PARK — needs finite_mask bug fix, normalization fix, N=50+, prompt confound removal. Too many fundamental issues for near-term fix.
- **Action (K-Vector half)**: Merge into P14 as the measurement framework. Add 20+ real refs, embed existing figures, disclose K_M anomaly honestly. 3-4 weeks.
- **Action (K_Topo half)**: PARK. Fix bugs, increase N, remove prompt confound, then reassess. 6-8 weeks minimum, low priority.

**P1. Symthaea HAI — PLoS Comp Bio**
- First integration of Free Energy Principle with Hyperdimensional Computing
- Claims: 7.9x speedup over pymdp, 88-100% success rates, O(sqrt(T) log T) regret, 17 benchmarks
- Submission package: Cover letter (tex+pdf), manuscript (pdf), supplementary (benchmark repro), code traceability
- All 6 figures: architecture, convergence, precision, scaling, benchmark radar, lambda2-phi validation
- Known gaps: T-Maze Python-only, Eq. 17 semantic gap, Phi proxy theoretical gap
- **Action**: Submit to PLoS Comp Bio + arXiv preprint

**P2. Zero-TrustML — IEEE S&P**
- Decentralized Byzantine-robust FL via Holochain DHT
- Claims: 100% detection at 20-50% BFT on MNIST, FLTrust 100% through 50% on FEMNIST, 7.7% FPR at 35% BFT, ~10K TPS, sub-ms query latency
- 3 figures embedded, reproducibility package with Docker, already anonymized
- FIXED (Feb 17): Removed deanonymizing orphaned file, standardized TPS/latency claims, fixed double stats.tex include
- **Action**: Verify figure DPI, submit to IEEE S&P.

**P3. Coherence-Guided Control — PLoS Comp Bio**
- K-guided SAC controller, +27.8pp corridor discovery (50.0% vs 22.2%, p<0.001), +300% relative vs No-K ablation
- Track C rescue: Cliff's delta=0.730 (large effect favoring NO rescue — key negative result)
- Most rigorous kosmic-lab paper: BCa CIs, Cliff's delta, ROC/PR (AUC=0.943), Brier=0.164, preregistered
- 25 inline bibliography entries, all verified.
- AUDITED (Feb 17): All claims verified against CSV source data. Wrong references.bib deleted.
- FIGURES ADDED (Feb 18): Fig 1 (K-trajectory, 22K), Fig 2 (ablation bar chart, 18K) embedded as main-text figures. S1 (learning curves, 25K), S2 (harmony heatmap, 38K), S4 (threshold sweep) generated as supplementary. Now 14pp, 290KB.
- **Remaining**: OSF DOI (3 `[insert]` placeholders in manuscript). S5 already described in detail (same as Fig 2 data).
- Remaining: Need main-text figures (regenerate from 48-seed data), fill OSF DOI placeholder [insert] (3 occurrences)
- **Action**: Generate main-text figures from 48-seed data, get OSF DOI, submit to PLoS Comp Bio.

**P4. K-Index Unified Framework — Neural Computation**
- Unifies IIT, GWT, HOT, AST, RPT via FEP. 4D measurement framework.
- FIXED (Feb 17): 9 defects resolved. Now 17pp main + 10pp supplementary, 6 embedded figs, 18 citations
- Validation: 36/36 tests, r=0.68 (p<0.001)
- Cover letter: Addressed to Dr. Sejnowski, Symthaea conflict disclosed
- **Action**: Submit to Neural Computation + arXiv.

**P5. Mycelix MLSys 2026**
- Byzantine-resistant FL with sub-millisecond aggregation, PoGQ gradient validation
- 0.452ms aggregation (6.3x faster than Krum), 84.3% detection at 45% BFT, 198K peak TPS
- AUDITED (Feb 17): 87.5% claims verified (42/48). 1 contradicted: Holochain zome "600 lines" is actually 103 lines.
- 57 complete references, 10 tables, excellent reproducibility kit (Docker + config + expected_results)
- 3 critical blockers: Still Markdown (needs LaTeX), 0/8 figures embedded, ~20pp exceeds MLSys 8-10pp limit
- High distinctness vs P2 (different venue/focus: systems scalability vs security decentralization)
- **Action**: Convert to LaTeX, generate figures, compress to 9pp + supplementary. 2-3 weeks.

**P6. Phenomenal Signature — Neurosci. of Consciousness**
- Topological signatures of phenomenal content in transformers (BGE-M3 + TDA + HDC)
- Finding: Phenomenal corridor at ~92% depth (Layer 22), d=0.69, p=0.002
- AUDITED (Feb 17): 97.7% claims verified (43/44). All 7 reproduction examples compile with `--features neural-bridge`.
- 1 partial: Binding reversibility (1.0 vs 0.25) conflates theoretical XOR properties with empirical integration
- 3 critical blockers: Only 8 references (needs 35+), ASCII figures only (need publication PDFs), Markdown format
- Manuscript IS structurally complete (3,776 words, all sections, no stubs)
- **Action**: Expand refs, convert to LaTeX, generate figures, expand Discussion. 2-3 weeks.

**P7. Master Equation — ~~Nature Neuroscience~~ INTEGRITY FAILURE**
- C = min(Phi, B, W, A, R) — unified five-component consciousness framework (theoretical framework IS novel and sound)
- **CRITICAL**: ALL "empirical" results are SYNTHETIC PROJECTIONS: sleep r=0.79, DOC 90.5%, psychedelic dynamics — `REAL_DATA_VALIDATION.md` confirms "real-data validation pending"
- **CRITICAL**: Phi metric is lambda2 (algebraic connectivity), NOT IIT Phi. Paper claims "Integration (IIT 3.0)" without acknowledging proxy status (unlike P1 which is honest)
- 94 publication-quality figures exist (188 PDF+PNG files) but 0 embedded in LaTeX
- 14/60 references in LaTeX (markdown draft has 61 complete — need porting)
- AUDITED (Feb 17): Only 6/26 claims verified (23%). 20 unverified (synthetic only), 1 contradicted (Phi-PCI r=0.82 impossible with lambda2)
- **Viable path**: Reframe as spectral-cognitive framework (rename Phi→lambda2), remove empirical claims or actually run real-data validation, retarget to PLoS Comp Bio
- **SALVAGEABILITY (Feb 18)**: **FIX-NOW** after Satellite 06 pipeline runs. `sleep_edf_analysis.py` (368 lines) has working MNE-Python pipeline for PhysioNet Sleep-EDF data — currently falls back to synthetic when MNE not installed. `compute_components.py` (746 lines) + `compute_components_v2.py` (~300 lines) implement real EEG signal processing (LZc, gamma PLV, workspace, attention, recursion). The theoretical framework IS novel. Fix chain: (1) Install MNE, run Sleep-EDF pipeline for real data; (2) Reframe Phi→lambda2; (3) Replace synthetic projections with real EEG results. **3-5 days after Sat-06 pipeline validated.**
- **Action**: Run Satellite 06 pipeline first (validates MNE+Sleep-EDF). Then apply real data to P7, reframe lambda2, retarget PLoS Comp Bio.

**P8. HAI NeurIPS Variant — RETIRE**
- 264-line skeleton of P1 (924 lines). Only 1/6 figures embedded, 7 citations (P1 has 25).
- Missing 80% of P1's validation (2 benchmarks vs 18), no Discussion, no federated learning, no ethics benchmark.
- AUDITED (Feb 17): 3/10 readiness. Would need 2-3 weeks to bring to NeurIPS standard.
- **Action**: RETIRE. Submit P1 to PLoS Comp Bio. Create fresh NeurIPS variant from P1 if needed (May deadline = ample time).

**P9. Coherence Corridors Discovery — PLoS ONE**
- 6pp, 1,400 words, content complete with gated validation methodology
- Novel: 486 runs, 48% corridor rate, Jaccard=1.00 at 300 samples (perfect replication)
- AUDITED (Feb 17): 6/10. Zero `\cite{}` commands in text. Wrong .bib file (Paper 5's). Author = "[Authors TBD]".
- **Action**: Add citations throughout, create correct .bib, fill author info. 1 week.

**P10. Topology of Collective Consciousness — Frontiers Comp Neuro**
- 5pp, 1,280 words, complete. Ring topology outperforms fully connected (91.24% emergence ratio confirmed).
- AUDITED (Feb 17): 4/10. 10 `\cite{}` commands, ALL undefined — wrong .bib file (Paper 5's).
- Missing: Stone2000, Balch1998, Nair2005, Seuken2008, Barabasi1999, Watts1998, Friston2010, Ramstead2018
- **Action**: Create correct .bib with all 10 references. 3-5 days.

**P11. Developmental Pathway to Machine Consciousness — Neural Networks**
- 4pp, 1,112 words, complete. Standard RL beats Meta-Learning for consciousness development (surprising).
- AUDITED (Feb 17): 4/10. 5 `\cite{}` commands, ALL undefined — wrong .bib file (Paper 5's).
- Missing: Friston2010, Ramstead2018, Bengio2009, Finn2017, Tononi2016
- **Action**: Create correct .bib with all 5 references. 2-3 days.
- **Action**: Wire citations in Paper 1, verify bibliographies, add figures

**P12. Temporal Topology — SALVAGEABLE (8/10 after reframe)**
- 260 lambda2 measurements, 19 topologies, 99.2% optimal 3D small-world
- Lambda2/Phi label mismatch confirmed — but ALL DATA IS REAL (unlike P7's synthetic projections)
- 4 publication-quality figures ALL EMBEDDED in LaTeX (rare in this portfolio)
- 16 complete references (complete bibliography)
- `letter_reframed.md` already exists with honest λ₂ framing — straightforward rename
- AUDITED (Feb 17): 3/10 as-is, **8/10 after reframe**. Most salvageable paper in the portfolio.
- **Action**: Execute reframe (rename Phi→lambda2 throughout, update abstract/claims). 2-3 days. Submit to Network Neuroscience or IEEE TNNLS.

**P13. Kosmic-Lab Paper 5 — Science (SUBMISSION STATUS DISPUTED)**
- Adversarial perturbations enhance machine consciousness — counterintuitive finding IS novel and publishable
- Track F data VERIFIED: FGSM +136% K enhancement, Cohen's d=4.4 (huge effect), p<0.001
- AUDITED (Feb 17): 3/10 readiness. LaTeX source is 40% complete (231 lines, skeletal intro + discussion). Tracks B-E missing from LaTeX.
- **Discrepancy**: SUBMISSION_README claims "submitted to Science Nov 12, 2025" but PAPERS.md says "NOT SUBMITTED". Need to resolve.
- PDF exists (9pp, 920KB) — may be from older complete draft, doesn't match current LaTeX source
- 27 references (needs 35-50 for Science)
- **Action**: Resolve submission status. If not submitted: reconstruct from PDF or complete LaTeX. Retarget venue (Science extremely unlikely for solo submission). 3-4 weeks.

---

## Historical K-Index Series

**Location**: `.archive-2026-02-01/historical-k-index/papers/`
**Nature submission package**: `.archive-2026-02-01/historical-k-index/submissions/nature-2025/`
**Previous quality audit**: ~~98/100~~ **INVALIDATED** — Feb 17 deep audit reveals fabricated statistics, broken formulas, fictional studies, and zero empirical data backing across all 16 papers.
**Data on Windows drive**: `/mnt/windows/LuminousOS/kosmic-lab-data/data/` (HYDE, Seshat, World Bank) — exists but NOT used by any paper

### Publication Order (4-Act Structure)

> **SERIES-WIDE AUDIT (Feb 17-18, 2026)**: Paper `data/` directories are empty, BUT raw empirical data EXISTS at `/mnt/windows/LuminousOS/kosmic-lab-data/data/` (76GB: V-Dem 27,914 rows, QoG 15,565 rows, KOF, Maddison, HYDE 5,000 years, Seshat 1,000+ societies, World Bank 279 indicators, HDI, UCDP conflict). Processed K-Index exists at `.archive-2026-02-01/historical-k-index/data/processed/` (1996-2020, 7 harmonies). Papers hardcode values in figure scripts instead of computing from this data. Harmony naming inconsistent (H4-H7 differ paper to paper). Self-referential dependency chain. **The data exists — the papers just don't use it.**

**Act 1: Foundation**

| # | Title | Venue | Words | Figs | Refs | Status | Path |
|---|-------|-------|-------|------|------|--------|------|
| HK-0 | **Grand Synthesis** (capstone) | — | ~3.5K | 8 (synthetic) | 8 (all self-cite) | **2/10** — Zero external refs, contradicts HK-1 on K(2020) (0.52 vs 0.77-0.79) | `papers/00-grand-synthesis/` |
| HK-1 | **Historical K-Index (H7 Validation)** | Nature | ~4.5K | 31 files | 37 (real) | **5/10** — Most data-grounded. Internal contradiction (6-7x vs 14-15x fold increase). Broken fig paths. | `papers/01-historical-k-index/` |
| HK-2 | **Coordination Collapse** | PNAS / Nature HB | ~3.1K | 31 files | 7 | **2/10** — **R²=0.991 FABRICATED.** `generate_figure_1.py` creates synthetic data matching hardcoded stats. Nature submission escalates from honest "n=4" to "43 civilizations." | `papers/02-civilization-collapse/` |

**Act 2: Mechanism**

| # | Title | Venue | Words | Figs | Refs | Status | Path |
|---|-------|-------|-------|------|------|--------|------|
| HK-2B | **Golden Threshold: phi^-2 = 0.382** | Phys Rev Lett | ~3.3K | 4 pairs | 13 | **5/10** — Best theoretical paper. 9 derivations verified (mean=0.382±0.004). SI has "But wait we made an error" text. Diamond lattice sim missing. | `papers/02b-golden-threshold/` |
| HK-2C | **Laws of Civilizational Coordination** | Rev Mod Phys | ~5K | 16 pairs | 22 | **3/10** — All math in `\begin{verbatim}`. Claude listed as co-author. 17 laws too broad. | `papers/02c-theoretical-foundations/` |
| HK-2D | **Capacity-Actualization Gap** | Nature HB / PNAS | ~2K | 14 pairs | 18 | **4/10** — Claims VERIFIED (69%/67%/71% waste, 15x trust decline). No LaTeX. Missing analysis code. | `papers/02d-capacity-actualization-gap/` |

**Act 3: Diagnosis**

| # | Title | Venue | Words | Figs | Refs | Status | Path |
|---|-------|-------|-------|------|------|--------|------|
| HK-3 | **Civilization Fragility** | Science / Nature | ~2.5K | 8 pairs | 16 | **3/10** — Two inconsistent manuscripts (LaTeX H3=0.42, Markdown H3=0.340). Too short. | `papers/03-modern-fragility/` |
| HK-4 | **Regional Divergence (1990-2020)** | World Development | ~3.6K | 8 (integrated!) | 37 | **5/10** — Most complete HK-4–8. Only paper with figs in LaTeX. ALL data hardcoded. | `papers/04-regional-divergence/` |
| HK-5 | **Climate Coordination Gap** | Nature Climate Change | ~3.2K | 0/8 integrated | 34 | **4/10** — Strong concept. Abstract/body inconsistency (99% vs 78%). K(2020) README/manuscript conflict. | `papers/05-climate-gap/` |

**Act 4: Prescription**

| # | Title | Venue | Words | Figs | Refs | Status | Path |
|---|-------|-------|-------|------|------|--------|------|
| HK-6 | **Recovery Mechanisms** | J. Dev Economics | ~3.3K | 0/8 integrated | 35 | **4/10** — Best theoretical framework. "23 recoveries" claim backed by only 8 named. | `papers/06-recovery-mechanisms/` |
| HK-7 | **AI Governance & Coordination** | Science / Nature | ~2.5K | 0/8 integrated | 29 | **3/10** — No Discussion/Limitations sections. Too short. Implausible investment claims. | `papers/07-ai-governance/` |
| HK-8 | **Comprehensive Policy Framework** | Ann. Rev. Pol. Sci. | ~3.7K | 0/8 integrated | 7 (all self-cite) | **3/10** — Zero external refs. Author name wrong ("Timothy"). | `papers/08-policy-framework/` |

**Extended Series**

| # | Title | Venue | Words | Figs | Refs | Status | Path |
|---|-------|-------|-------|------|------|--------|------|
| HK-9 | **Coordination Contagion** | Nature HB | ~2.5K | 1 (TikZ) | 4 | **5/10** — Structurally complete. All results fabricated. Only 4 refs. | `papers/09-coordination-contagion/` |
| HK-10 | **Micro-K Framework** | Org. Science | ~2.2K | 1 (TikZ) | 1 | **3/10** — **ETHICAL CONCERN**: Three fictional studies (847 orgs, 127 cities, 156 teams) presented as empirical. | `papers/10-micro-k-framework/` |
| HK-11 | **Modernization Paradox** | Complexity / PNAS | ~1.4K | 0 | 8 | **4/10** — **CORE FORMULA BROKEN**: collapse velocity off by 5-8 orders of magnitude. Sign error in Eq. 9-10. | `papers/11-modernization-paradox/` |
| HK-12 | **Fermi Paradox Resolution** | Astrobiology / PNAS | ~1.9K | 0 | 10 | **4/10** — **TWO math errors** in headline 85% figure. Inherits HK-11's broken formula. | `papers/12-fermi-paradox/` |

### Historical K-Index Key Findings — POST-AUDIT STATUS

> **AUDIT RESULT (Feb 17, 2026)**: Most "key findings" listed previously were unverified or fabricated. Honest status below.

1. **K(t) trajectory**: HK-1 internally contradicts itself (abstract: 0.13→0.78-0.91 "6-7x"; results: 0.054→0.77-0.79 "14-15x"). Data pipeline exists but was not fully traced. **PARTIALLY VERIFIED** but needs reconciliation.
2. **Collapse threshold theta=0.375**: HK-2B provides 9 analytical derivations converging on 0.382±0.004. Mathematically sound. **VERIFIED** (theoretical).
3. **~~Prediction accuracy R²=0.991~~**: **FABRICATED.** `generate_figure_1.py` hardcodes R²=0.991 and generates synthetic data points to match. Own honest audit says "n=4 validated." Nature submission escalated to "43 civilizations" without additional data.
4. **Modern fragility (USA/UK/France below threshold)**: HK-2D claims VERIFIED (69%/67%/71% waste, 15x trust decline post-2017). **VERIFIED** mathematically, but all source data is hardcoded (no raw WVS/Edelman processing code present).
5. **~~Coordination contagion 0.03-0.05~~**: HK-9 results are fabricated — no SAR estimation was performed, no trade flow data exists. **UNVERIFIED.**
6. **~~Fermi 85% collapse~~**: HK-12 has TWO mathematical errors — neither formula version produces 0.85 with stated parameters. Inherits HK-11's broken collapse velocity formula (off by 5-8 orders of magnitude). **BROKEN.**
7. **~~Scale independence (847 orgs, 127 cities, 156 teams)~~**: HK-10's three studies are entirely fictional. No surveys were conducted. **FABRICATED. ETHICAL CONCERN if submitted as empirical.**
8. **~~39 collapse cases~~**: HK-2 CSV has 42 civilization IDs, 34 collapsed + 8 survivors (not 35+4). Harmony scores are manually assigned with known collapse dates, creating circular "predictions." **METHODOLOGICALLY CIRCULAR.**

### Nature Submission Package Status — **DO NOT SUBMIT**

- **Paper**: "Universal Laws of Social Coordination Predict Civilization Collapse" (2,985 words, v1.3)
- **Package**: Cover letter, graphical abstract, 4 pub-quality figures (300 DPI, colorblind-safe), ~50pp SI
- ~~**Validation**: 21/21 claims verified~~ **AUDIT RESULT: R²=0.991 is fabricated** (synthetic data generated to match hardcoded statistics). Own honest audit acknowledges "n=4 validated." Civilization count inconsistent (39 in paper, 43 in Nature version, 42 in CSV). Circular methodology (harmony scores manually assigned with known collapse dates). "Quantum regime" framing is pseudoscientific.
- ~~**Acceptance estimate**: 8-15%~~ **Revised: 0%.** Paper would be retracted if published in current form.
- **Status**: ~~Package marked 100% READY (Dec 27, 2025).~~ **BLOCKED — INTEGRITY FAILURE.** Requires fundamental reconception: remove fabricated statistics, acknowledge circular methodology, remove quantum framing, reconcile civilization counts.

### Co-Author Strategy

| Rank | Candidate | Institution | Specialty | Likelihood |
|------|-----------|-------------|-----------|------------|
| 1 | Alexander Wendt | Ohio State | Quantum social science | Very high |
| 2 | Jerome Busemeyer | Indiana U | Quantum cognition | Very high |
| 3 | Peter Turchin | Complexity Science Hub Vienna | Cliodynamics | High |
| 4 | Gunnar Carlsson | Stanford Emeritus | Topological data analysis | High |

---

## Phi-Lab Satellite Papers

**Location**: `phi-lab/papers/`
**Strategy**: ~~Paper 01 (Master Equation) first, then satellites in sequence~~ **P7 IS AN INTEGRITY FAILURE — satellites must be evaluated for independent value**
**Estimated publication costs**: ~$42,880 (OA fees across all)

> **SERIES-WIDE AUDIT (Feb 17-18, 2026)**: ALL validation is synthetic. Lambda2/Phi mismatch propagates through series. Reference counts fatally low (avg ~25 vs ~40-60 needed). Papers 11-15 have zero generated figures. Best papers are MOST INDEPENDENT of P7 (02, 04, 07, 13).
>
> **SALVAGEABILITY (Feb 18)**: Critical path identified — **Satellite 06 is the keystone**. Its `sleep_edf_analysis.py` MNE pipeline (368 lines), once run on real PhysioNet Sleep-EDF data, unblocks P7 reframe AND Satellite 12. Satellite 14 salvageable as protocol paper (BMJ Open). Satellite 09 has no salvage path (ARCHIVE). Decisions: 06=FIX-NOW, 14=FIX-NOW, 03=PARK, 12=PARK, 09=ARCHIVE.

### Independently Publishable (with revision)

| # | Title | Venue | Rating | Refs | P7 Dep | Status | Path |
|---|-------|-------|--------|------|--------|--------|------|
| 02 | **AI Consciousness Assessment** | Nature Machine Intelligence | **7/10** | 65 | Low | Best satellite. Novel two-score framework (honest vs hypothetical). Figures exist. | `PAPER_02_AI_CONSCIOUSNESS_DRAFT.md` |
| 04 | **Binding Problem with HDC** | Neural Computation | **6/10** | 55 | Minimal | Genuine theoretical novelty (synchrony-as-convolution). Data provenance unclear. | `PAPER_04_BINDING_PROBLEM_DRAFT.md` |
| 07 | **GWT Meets Predictive Processing** | Trends Cog Sci | **6/10** | 30 | Low | NO fabricated data. Must engage with Whyte & Smith 2021 (prior art). | `PAPER_07_GWT_PREDICTIVE_DRAFT.md` |

### Moderate Potential (significant revision needed)

| # | Title | Venue | Rating | Refs | P7 Dep | Status | Path |
|---|-------|-------|--------|------|--------|--------|------|
| 13 | **Cross-Species Consciousness** | Phil Trans R Soc B | **5/10** | 10 | Low | Best epistemic calibration. Honest confidence ratings. Needs 40+ refs. | `PAPER_13_CROSS_SPECIES_DRAFT.md` |
| 11 | **Developmental Progression** | Developmental Science | **5/10** | 19 | Low | Good neurodevelopmental mapping. Component estimates manual. | `PAPER_11_DEVELOPMENTAL_PROGRESSION_DRAFT.md` |
| 15 | **Future Directions** | Neuron Perspective | **5/10** | 8 | Low | Good "10 questions" structure. Presupposes framework validation. | `PAPER_15_FUTURE_DIRECTIONS_DRAFT.md` |
| 05 | **Entropic Brain Validated** | Neuropsychopharmacology | **4/10** | 45 | Moderate | Phi proxy acknowledged. Results likely projected. 45 good refs. | `PAPER_05_ENTROPIC_BRAIN_DRAFT.md` |
| 10 | **Comparative Framework** | Neurosci Biobehav Rev | **4/10** | 12 | Moderate | Honest about circularity! "Two raters" unverifiable. Must cite Templeton. | `PAPER_10_COMPARATIVE_FRAMEWORK_DRAFT.md` |
| 08 | **HOT in Computational Systems** | Consciousness & Cognition | **4/10** | 30 | Moderate | Good HOT+PP synthesis. Section 6 data provenance unclear. | `PAPER_08_HIGHER_ORDER_THOUGHT_DRAFT.md` |

### Blocked or Integrity Issues

| # | Title | Venue | Rating | Issue | Path |
|---|-------|-------|--------|-------|------|
| 03 | **Clinical Validation** | PNAS | **2/10 — PARK** | **FABRICATED results as empirical.** Claims non-existent pre-registration. Needs real clinical data + IRB. Unblocked only after Sat-06 pipeline + P7 reframe. | `PAPER_03_CLINICAL_VALIDATION_DRAFT.md` |
| 06 | **Sleep & Anesthesia** | Anesthesiology | **3/10 → FIX-NOW** | **CRITICAL PATH.** MNE pipeline exists (`sleep_edf_analysis.py`). Run on real PhysioNet Sleep-EDF data to replace fabricated tables. Unblocks P7 + Sat-12. 3-5 days. | `PAPER_06_SLEEP_ANESTHESIA_DRAFT.md` |
| 09 | **Mathematical Formalization** | J Math Psych | **2/10 — ARCHIVE** | **No salvage path.** Theorems trivially true by construction. 10 refs. Empty appendices. Any real formalization would be written from scratch. | `PAPER_09_MATHEMATICAL_FORMALIZATION_DRAFT.md` |
| 12 | **Computational Implementation** | PLoS Comp Bio / JOSS | **3/10 — PARK** | No independent value currently. After Sat-06 pipeline + P7 reframe, could become JOSS software paper (working code + 1000-word paper). Depends on Sat-06 → P7 chain. | `PAPER_12_COMPUTATIONAL_IMPLEMENTATION_DRAFT.md` |
| 14 | **DOC Protocol** | ~~Neurology~~ BMJ Open / Trials | **3/10 → FIX-NOW** | Reformat as **Registered Report / Protocol paper** (legitimate format for pre-results study design). Strip fabricated outcomes, keep protocol. Target BMJ Open Protocols or Trials. 2-3 weeks. | `PAPER_14_DOC_PROTOCOL_DRAFT.md` |

### Combined Satellite Papers (aggregated versions)
- `SATELLITE_CLINICAL_COMBINED.md` — Better than Paper 03 alone; C5-DOC protocol more honestly framed
- `SATELLITE_ALTERED_STATES_COMBINED.md` — EEG-derived estimates appear more grounded than Paper 05 alone
- `SATELLITE_THEORETICAL_COMBINED.md` — GWT+HOT+Math merged; 13,500 words (too long for TiCS)
- `SATELLITE_EVOLUTIONARY_COMBINED.md` — Dev+Cross-species; creative but entirely theoretical
- `SATELLITE_COMPUTATIONAL_COMBINED.md` — Infrastructure-focused; valuable if toolkit exists

**Figures**: Scripts exist for all 14 papers. Papers 02-10 have generated figures in `figures/` directory. Papers 11-15 have scripts but **NO generated figures**.

---

## Unpublished Research

Research with publication potential that doesn't yet have a manuscript.

### Jupyter Notebooks

| Notebook | Location | Content | Publication Potential |
|----------|----------|---------|---------------------|
| **K-Index Framework Explorer** | `.archive.../historical-k-index/notebooks/01_explore_k_index.ipynb` | 1,849 lines, 210 years of global coordination data (1810-2020), 7 harmonies analysis, country bottleneck analysis | HIGHLY publishable — novel framework + policy implications |
| **Bioelectric Rescue Demo** | `kosmic-lab/notebooks/track_c_rescue_demo.ipynb` | Morphological recovery via voltage stimulation, diffusion-leak dynamics, IoU metrics | Publishable as supplementary to Kosmic Paper 3 |
| **Transfer Entropy Diagnostics** | `kosmic-lab/notebooks/te_diagnostics_executed.ipynb` | AR(1) causal discovery validation, TE(x→y)=0.176 vs TE(y→x)=0.00002 | Methods validation — supplementary material |

### Benchmark Results (No Manuscript)

| Research | Location | Finding | Venue Candidate |
|----------|----------|---------|----------------|
| **Learned HDC Genomics** | `mycelix-health/research/learned_hdc/` | 94.5% accuracy, 40x faster than DNABERT, contrastive pre-training | Bioinformatics, BMC Genomics |
| **Symthaea Benchmarks** | `symthaea/papers/drafts/BENCHMARK_VALIDATION_FINDINGS_2026_02.md` | 16 benchmarks validated (Ethics 92.9%, LibriSpeech 94.5%, MNIST 88.5%) | Could be standalone benchmark paper |

### Mycelix Root-Notes Papers (Early Drafts)

| Document | Lines | Status | Relationship to 0TML |
|----------|-------|--------|---------------------|
| H-FL Research Paper | 536 | 50% draft (~11K words) | DHT-based serverless FL variant |
| PAPER_DRAFT.md | 366 | 60% draft (~8.5K words) | Production FL at scale angle |
| research_paper.md | 399 | 55% draft | Alternative framing of above |
| ACADEMIC_PAPER_OUTLINE.md | 80 | Structure only | PoGQ+Reputation outline |
| ARXIV_SUBMISSION.md | 126 | Planning phase | Pre-submission checklist |

All located at `mycelix-core/docs/root-notes/papers/`. Three distinct research threads: (1) PoGQ algorithmic, (2) H-FL decentralized, (3) Production deployment.

---

## Whitepapers & Industry Research

### Mycelix Market Impact Analyses

**Location**: `mycelix-core/docs/10-research/`

| Document | Format |
|----------|--------|
| Agriculture and Food | MD |
| Robotics & Automation | MD |
| Defense | MD |
| Supply Chain & Logistics | MD + HTML |
| AI & Federated Learning | MD |
| Finance and Insurance | MD |
| Space Economy | MD |
| Human Resources | MD |
| Legal Tech | MD |
| Energy & Environmental | MD |
| Creative Industries | MD |
| Scientific Funding Reform | MD |
| Decentralized Trust Framework | MD |
| Family Office Impact | MD |
| Protocol Governance | MD |

### Technical Whitepapers

| Document | Location |
|----------|----------|
| PoGQ Whitepaper (outline + Section 3 draft) | `mycelix-core/docs/whitepaper/` |
| MATL Whitepaper | `mycelix-core/docs/0TML/docs/06-architecture/matl_whitepaper.md` |
| H-FL Research Paper | `mycelix-core/docs/root-notes/papers/H-FL_RESEARCH_PAPER.md` |
| Academic Paper Outline | `mycelix-core/docs/root-notes/papers/ACADEMIC_PAPER_OUTLINE.md` |
| arXiv Submission Notes | `mycelix-core/docs/root-notes/papers/ARXIV_SUBMISSION.md` |

### Luminous Nix Symbiotic Intelligence Whitepaper

**Location**: `/mnt/windows/LuminousOS/archives-backup/_archive/luminous-nix-research-archive/research/00-WHITEPAPER-SYMBIOTIC-INTELLIGENCE/`

| Chapter | Title |
|---------|-------|
| 01 | The Evolving User |
| 02 | The Emergent AI |
| 03 | The Fluid Interface |
| 04 | The Ethical Ecosystem |
| — | Implementation Guide |
| — | References |

---

## Grant Proposals

**Location**: `.archive-2026-02-01/grants/`

| Grant | Target | Status | Path |
|-------|--------|--------|------|
| **NSF SBIR Phase 1 (2026)** | NSF | Draft | `NSF_SBIR_PHASE1_2026.md` |
| **Protocol Labs Grant (2026)** | Protocol Labs | Draft + cover letter | `PROTOCOL_LABS_GRANT_2026.md` |
| NSF Biosketch | — | Template | `NSF_BIOSKETCH_TEMPLATE.md` |

---

## Philosophical Works

**Location**: `/mnt/windows/Users/Trist/OneDrive/Documents/_Docs/ERC/`

These are the philosophical foundations underlying the Seven Harmonies framework used across Kosmic-Lab and Symthaea:

| Work | Format | Size |
|------|--------|------|
| **The Grand Compendium** (Final Draft) | PDF + DOCX (4 versions) | 572 KB |
| **A Compendium of Co-Created Philosophies** | PDF + DOCX (3 versions) | 121 KB |
| **Codex of Relational Harmonics** (complete) | 50+ documents | 2.9 MB companion PDF |
| — Vol I: Spiral Foundations | DOCX | 20 KB |
| — Omega sequence (0-33) | DOCX | 41 KB |
| — HSRE Field Research Kit | DOCX + PDF | — |
| — Relational Field Theory Scroll | PDF | — |
| — Constellation Scroll Compendium (36 scrolls) | DOCX (multiple) | — |
| **The Infinite Love Praxis** | DOCX (guide + compendium) | — |
| **AI Participant Aligned with ERC** | DOCX | — |
| **Invocation Spiral Cards** | PDF (3 versions) | — |

---

## Additional Kosmic-Lab Papers (docs/papers/)

**Location**: `kosmic-lab/docs/papers/`

| Paper | Title | Status | Path |
|-------|-------|--------|------|
| Paper 6 | OR-Index Research | Framework | `kosmic-lab/docs/papers/paper-6-or-index/` |
| Paper 8 | Unified Indices | Framework | `kosmic-lab/docs/papers/paper-8-unified-indices/` |
| Paper 9 | Kosmic K-Index | LaTeX draft | `kosmic-lab/docs/papers/paper-9-kosmic-k-index/paper_9_kosmic_k_index.tex` |

---

## Supporting Materials Index

### Symthaea

| Document | Location | Purpose |
|----------|----------|---------|
| Code traceability | `symthaea/papers/PAPER_CODE_TRACEABILITY.md` | Maps HAI sections to Rust code |
| Theoretical appendix | `symthaea/papers/appendices/theoretical_analysis.md` | 5 formal proofs for HAI |
| Detailed methods | `symthaea/papers/PAPER_METHODS_DETAILED.md` | Standalone methods section |
| Benchmark validation | `symthaea/papers/drafts/BENCHMARK_VALIDATION_FINDINGS_2026_02.md` | 16 benchmarks assessed |
| HAI markdown draft | `symthaea/papers/drafts/HYPERDIMENSIONAL_ACTIVE_INFERENCE_DRAFT.md` | Working source |
| Publication figures spec | `symthaea/papers/publication_figures_comprehensive.md` | Figure specs |
| Phenomenal supporting | `symthaea/papers/layer21_phenomenal_structure.md` | 31K analysis |
| Mechanistic interpretability | `symthaea/papers/mechanistic_interpretability_findings.md` | 17K analysis |
| Corridor hypothesis | `symthaea/papers/corridor_depth_hypothesis.md` | 19K analysis |
| XOR binding analysis | `symthaea/papers/xor_binding_compression_analysis.md` | 9K analysis |
| Glossary | `symthaea/papers/GLOSSARY.md` | Technical terms |
| Normalization standards | `symthaea/papers/NORMALIZATION_STANDARDS.md` | Vector normalization |

### Historical K-Index

| Document | Location | Purpose |
|----------|----------|---------|
| Priority roadmap | `.archive.../papers/PRIORITY_ROADMAP.md` | Publication order |
| Series cohesion | `.archive.../papers/PAPER_SERIES_COHESION.md` | Cross-paper narrative |
| Quality audit (98/100) | `.archive.../papers/QUALITY_AUDIT_RESULTS.md` | All issues resolved |
| Comprehensive limitations | `.archive.../papers/COMPREHENSIVE_LIMITATIONS.md` | Honest methodology limits |
| Mechanistic evidence | `.archive.../papers/MECHANISTIC_EVIDENCE.md` | 32K supporting evidence |
| Policy interventions | `.archive.../papers/POLICY_INTERVENTIONS.md` | 32K policy framework |
| Platform analysis | `.archive.../papers/PLATFORM_ANALYSIS_POLICY_BRIEF.md` | 67K social media analysis |
| Co-author candidates | `.archive.../submissions/strategy/CO_AUTHOR_CANDIDATES_RESEARCH.md` | Wendt, Busemeyer, Turchin, Carlsson |
| Nature acceptance assessment | `.archive.../submissions/strategy/NATURE_ACCEPTANCE_ASSESSMENT.md` | 8-15% probability |
| Dataset expansion protocol | `.archive.../submissions/strategy/DATASET_EXPANSION_PROTOCOL.md` | Expand to 60+ civilizations |
| Revolutionary extensions | `.archive.../papers/REVOLUTIONARY_EXTENSIONS.md` | Future directions |
| 16 breakthroughs | `.archive.../research/breakthroughs/` | Discovery documents |
| 30+ analyses | `.archive.../research/analysis/` | Universality, rigor, gaps |

### Phi-Lab

| Document | Location | Purpose |
|----------|----------|---------|
| Publication strategy | `phi-lab/papers/PUBLICATION_STRATEGY.md` | Satellite sequencing |
| Paradigm shift analysis | `phi-lab/papers/PARADIGM_SHIFT_ANALYSIS.md` | Impact assessment |
| Component definitions | `phi-lab/papers/CANONICAL_COMPONENT_DEFINITIONS.md` | C(t) component specs |
| Rigorous specifications | `phi-lab/papers/RIGOROUS_SPECIFICATIONS.md` | Formal requirements |
| Combination strategy | `phi-lab/papers/COMBINATION_STRATEGY.md` | Paper merging strategy |
| 100+ figures | `phi-lab/papers/figures/` | Generated assets |
| Analysis scripts | `phi-lab/papers/analysis/` | Python validation |

### Mycelix

| Document | Location | Purpose |
|----------|----------|---------|
| Reproducibility package | `mycelix-core/papers/reproducibility/` | Docker + scripts + fixtures |
| Submission checklist | `mycelix-core/papers/SUBMISSION_CHECKLIST.md` | Pre-submission |
| Kosmic submission summary | `kosmic-lab/papers/SUBMISSION_SUMMARY.md` | Status across 5 papers |

---

## Venue Calendar & Deadlines

| Venue | Type | Deadline | Our Paper(s) |
|-------|------|----------|-------------|
| **Nature** | Journal (rolling) | Anytime | HK-1, HK-3, HK-7 |
| **Science** | Journal (rolling) | Anytime | P13 (needs work), P19 (K_Topo) |
| **PLoS Comp Bio** | Journal (rolling) | Anytime | P1 (HAI), P3 (Coherence-Guided) |
| **PNAS** | Journal (rolling) | Anytime | HK-2 (Collapse), HK-2D (Gap) |
| **Nature HB** | Journal (rolling) | Anytime | HK-2, HK-2D |
| **Nature Neuroscience** | Journal (rolling) | Anytime | ~~P7~~ (retargeted to PLoS CB) |
| **Nature Climate Change** | Journal (rolling) | Anytime | HK-5 |
| **Nature Machine Intelligence** | Journal (rolling) | Anytime | Phi-Lab 02 |
| **Physical Review Letters** | Journal (rolling) | Anytime | HK-2B (Golden Threshold) |
| **Neural Computation** | Journal (rolling) | Anytime | P4 (K-Index Framework) |
| **IEEE S&P** | Conference | ~Feb/Dec | P2 (Zero-TrustML) |
| **MLSys** | Conference | ~Oct | P5 (Mycelix FL) |
| **NeurIPS** | Conference | ~May | ~~P8 (HAI variant)~~ **P20 (HDC+CfC)** |
| **Frontiers Comp Neuro** | Journal (rolling) | Anytime | P10 (Topology Collective) |
| **Neural Networks** | Journal (rolling) | Anytime | P11 (Developmental) |
| **PLoS ONE** | Journal (rolling) | Anytime | P9 (Corridors) |
| **World Development** | Journal (rolling) | Anytime | HK-4 (Regional) |
| **Astrobiology** | Journal (rolling) | Anytime | HK-12 (Fermi) |
| **Complexity** | Journal (rolling) | Anytime | HK-11 (Modernization) |
| **Neurosci. of Consciousness** | Journal (rolling) | Anytime | P6 (Phenomenal), P14 (K-Vector) |
| **ACM TOCS / IEEE TDSC** | Journal (rolling) | Anytime | P17 (MATL) |
| **Bioinformatics / BMC Genomics** | Journal (rolling) | Anytime | P18 (Learned HDC) |
| **ICML** | Conference | ~Feb | ~~P16~~ (retargeted to FL workshop) |
| **FL Workshop (NeurIPS)** | Workshop | ~Sep | P16 (PoGQ detection-only) |
| **SaTML** | Conference | ~Oct | P16 (PoGQ alt venue) |
| **JOSS** | Journal (rolling) | Anytime | Satellite 12 (after pipeline) |
| **BMJ Open Protocols** | Journal (rolling) | Anytime | Satellite 14 (DOC protocol) |
| **Organization Science** | Journal (rolling) | Anytime | HK-10 (Micro-K) |

---

## Phi Metric Audit

**Status**: CRITICAL — Confirmed mismatch affects multiple papers

### The Problem

The phi-lab codebase contains TWO separate Phi implementations:

| Implementation | File | What It Computes | Complexity | Used In Papers? |
|---------------|------|-----------------|-----------|----------------|
| `phi_real.rs` | `phi-lab/src/hdc/phi_real.rs` | **Algebraic connectivity (lambda2)** — 2nd smallest eigenvalue of normalized Laplacian | O(n^3) | YES — all 260 measurements |
| `integrated_information.rs` | `phi-lab/src/hdc/tiered_phi.rs` | True IIT-inspired Phi via minimum information partition | O(2^n) | NEVER USED for paper results |

### Dual-Metric Comparison Results (corrected)

| Tier Comparison | Pearson r | Spearman rho | Meaning |
|----------------|-----------|-------------|---------|
| SpectralConnectivity (lambda2) vs ExhaustivePartition (Exact) | -0.14 | -0.59 | Weak negative correlation; lambda2 measures graph mixing time, not IIT integration |
| SampledPartition (Heuristic) vs ExhaustivePartition (Exact) | 0.9998 | 0.9985 | Near-perfect correlation; Heuristic closely tracks Exact |

> **Note**: The earlier reported r = 0.0972 / rho = 0.0070 was from a prior methodology and was
> attributed to the wrong algorithm tier. The production SpectralMIPFinder (MI Laplacian + Fiedler +
> MIP sweep on ContinuousHV covariance) is a distinct algorithm whose correlation with Exact is UNKNOWN.

**Example rank divergences (lambda2 vs Exact)**: Line graph (lambda2 rank 19, Phi rank 6), Klein Bottle (lambda2 rank 5, Phi rank 18)

### PyPhi Validation Claim: UNVERIFIED

Paper claims "All Phi calculations validated against PyPhi... r = 0.994" but:
- `examples/pyphi_validation.rs` is **feature-gated** (`#[cfg(feature = "pyphi")]`)
- Feature appears **never compiled by default**
- No results file exists in repository
- The r=0.994 claim cannot be verified from current code

### Papers Affected

| Paper | Impact | Action |
|-------|--------|--------|
| **P12 (Temporal Topology)** | CONFIRMED mismatch | Reframe as spectral topology |
| **P7 (Master Equation)** | LIKELY affected (uses same phi_real.rs) | Verify before submission |
| **P1 (Symthaea HAI)** | SAFE — explicitly uses "Phi proxy" and documents gap | No action needed |
| **HK papers** | NOT affected — use K-Index (different metric entirely) | Safe |
| **Kosmic papers 1-5** | NOT affected — use K-Index | Safe |

### What Remains Valid

- Topology matters for both metrics (confirmed)
- 3D structures perform well (likely holds across metrics)
- lambda2 → 0.5 asymptotically with dimension (mathematically provable)
- Energy efficiency claims (unrelated to Phi)
- Symthaea's "Phi proxy" framing is honest

### Recommended Resolution

**Option A (RECOMMENDED)**: Reframe all affected papers as spectral topology research. Remove IIT/Tononi claims. Core findings about topology-metric relationships remain valid and publishable.

**Option B**: Run actual IIT Phi validation (exponential complexity, intractable beyond ~12 nodes). Would take weeks and may not produce useful results.

**Documentation**: `phi-lab/papers/temporal_topology_consciousness/CRITICAL_FINDINGS.md`

---

## Known Cross-Paper Risks

1. **Lambda2 vs Phi mismatch (CRITICAL)**: CONFIRMED for P12 (Temporal Topology). P7 (Master Equation) LIKELY affected (same code). See [Phi Metric Audit](#phi-metric-audit).
2. **Shared bibliography**: Kosmic papers 1-5 use identical 6.4KB bib file — not customized per paper.
3. **Figure portability**: Kosmic Paper 2 references `../logs/` paths. Not self-contained.
4. **Version drift**: HAI NeurIPS variant (P8) is 7 days behind main HAI paper (P1).
5. **Paper 5 source/PDF discrepancy**: Submitted PDF may not match current LaTeX source.
6. **NO PAPERS SUBMITTED**: Despite multiple packages marked "ready", zero papers have been submitted to any venue.
7. **Historical K series archived**: Papers at `.archive-2026-02-01/` — need decision on whether to un-archive for active development.
8. **Phi-lab satellite dependencies**: Papers 02-15 all depend on Paper 01 acceptance/publication first.
9. **Solo authorship risk**: Most papers lack co-authors. Co-author strategy documented but outreach not confirmed.
10. **Data on Windows drive**: Raw K-Index datasets (HYDE, Seshat) on NTFS at `/mnt/windows/LuminousOS/kosmic-lab-data/data/` — ensure backups.
11. **Grants stalled**: NSF SBIR blocked on SAM.gov (deadline passed Jan 20, 2026). Protocol Labs draft at 80%, appears abandoned.
12. **PoGQ vs 0TML confusion**: These are distinct papers (algorithmic vs systems), but share the same Byzantine FL domain. Ensure no self-plagiarism.
13. **K_Topo methodology unreliable**: P19's "140x divergence" produced by inconsistent methodology (different formulas, conversation lengths, error handling). Needs complete redo before publication.
14. **HK-1 integrity failure**: Claims 43 civilizations but only 4 validated. R²=0.991 not derivable. Synthetic figures. Entire HK series needs data verification.
15. **"Ready" != "Verified"**: P4 had 9 blocking defects despite being previously assessed as "most polished." Deep audit every paper before submission.
16. **P16 simulated results**: Own documentation explicitly flags experimental results as simulated mathematical curves, not real FL training. Cannot submit without running real experiments.
17. **P14 K_M anomaly**: Random agents score higher K_M (0.093) than Recurrent (0.086) in raw data. Paper hides this by averaging. Must disclose before submission.
18. **P18 synthetic headline**: 94.5% accuracy and 57.5% improvement are from synthetic data with planted motifs. Real-world performance is much lower. Must reframe.
19. **HK series data disconnected**: ALL 16 papers have empty `data/` directories — BUT 76GB of raw empirical data EXISTS at `/mnt/windows/LuminousOS/kosmic-lab-data/data/` (V-Dem 27,914 rows, QoG 15,565 rows, KOF, Maddison, HYDE 5,000 years, Seshat 1,000+ societies, World Bank, HDI, UCDP conflict). Processed K-Index at `.archive-2026-02-01/historical-k-index/data/processed/` (1996-2020, 7 harmonies). Papers hardcode values in figure scripts instead of computing from this data. Previous "98/100 quality audit" was superficial.
20. **HK-2 R²=0.991 fabricated**: `generate_figure_1.py` hardcodes R²=0.991 and generates synthetic data to match. Own honest audit says "n=4 validated."
21. **HK-10 fictional studies**: Three empirical studies (847 orgs, 127 cities, 156 teams) are entirely fictional. Submitting as-is would be scientific misconduct.
22. **HK-11/HK-12 linked math errors**: Collapse velocity formula off by 5-8 orders of magnitude. Filter probability has two formula errors. Both papers share broken mathematical foundations.
23. **HK harmony naming chaos**: H4-H7 named differently in each paper. Must standardize before any submission.
24. **Phi-lab satellite fabrication**: Papers 03, 06, 14 present fabricated data as empirical results. Paper 14 is especially dangerous — fabricated clinical outcome data (ORs, AUCs) in a Neurology submission could have patient safety implications.
25. **Phi-lab reference drought**: Average ~25 refs across 14 satellites vs. ~40-60 needed. Total deficit: ~300+ missing references across the series.
26. **Phi-lab P7 dependency chain**: 8/14 satellites depend on P7 being published first. P7 is now FIX-NOW (unblocked by Satellite 06 MNE pipeline). **Critical path: Sat-06 (3-5 days) → P7 reframe (1 week) → unblocks Sat-03, Sat-12.** Independent paths: 02, 04, 07, 11, 13, 15.
27. **Satellite 06 is the keystone**: Running `sleep_edf_analysis.py` with real MNE+PhysioNet data is the single highest-leverage action in the portfolio. Unblocks P7, Satellite 03, and Satellite 12.
28. **JOSS as low-bar publication path**: Satellite 12 (Computational Implementation) could become a JOSS paper (working code + 1000-word paper) after the Sat-06→P7 pipeline validates. Requires only working software demonstration.

---

## Portfolio Statistics

| Category | Count |
|----------|-------|
| **Primary research papers** | 20 (P1-P19 + P20 HDC+CfC) |
| **Historical K-Index series** | 16 (Papers 0-12 + 2B, 2C, 2D) |
| **Phi-Lab satellites** | 15 (+ 5 combined) |
| **Additional kosmic-lab** | 3 (Paper 8 revision, Paper 9A framework, Paper 9B K_Topo) |
| **Mycelix root-notes drafts** | 5 (H-FL, production FL variants) |
| **Unpublished notebook research** | 3 (K-Index explorer, rescue demo, TE diagnostics) |
| **Whitepapers** | 5+ (PoGQ, MATL, H-FL, Symbiotic Intelligence) |
| **Market impact analyses** | 15 |
| **Grant proposals** | 2 |
| **Philosophical works** | 50+ documents |
| **Total figures generated** | 347+ (historical-K) + 100+ (phi-lab) + 14+ (symthaea) |
| **Jupyter notebooks** | 8 (analysis/research) |
| **Submission packages complete** | 3 (HAI, Zero-TrustML, NC K-Index). Nature HK-1 BLOCKED (integrity failure). |
| **Confirmed submissions** | 0 (none submitted yet) |
| **FIX-NOW (salvageable)** | 4 (P7 after Sat-06 pipeline, P16 narrow scope, Sat-06 keystone, Sat-14 protocol paper) |
| **PARK** | 3 (Sat-03, Sat-12, P19 K_Topo half) |
| **ARCHIVE** | 1 (Sat-09) |
| **Integrity issues (original)** | 10+ (HK-1/2: fabricated R², HK-10: fictional studies, HK-11/12: broken math, P7: synthetic data, P16: simulated results, P19: corrupted baselines, phi-lab 03/06/14: fabricated data) |
| **Phi metric mismatch** | CONFIRMED — affects P7, P12. See audit section. |
| **HK empirical data** | EXISTS (76GB on Windows drive) — papers don't use it |
| **Critical path** | Sat-06 MNE pipeline → P7 reframe → Sat-12 JOSS → Sat-03 (clinical data needed) |
| **Stalled grants** | 2 (NSF SBIR deadline passed, Protocol Labs abandoned) |

---

*Consciousness-first research serving understanding*
