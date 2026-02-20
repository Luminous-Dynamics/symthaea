# Paper Priority Ranking — Definitive

> Created: 2026-02-17 | Updated: 2026-02-17 (post-audit HONEST RECKONING)
> Status: 0 papers submitted. 52+ in portfolio. This ranking determines what to submit first.
>
> **AUDIT IMPACT**: HK-1 downgraded from 19→5 (integrity failure). P19 downgraded from 15→8 (unreliable methodology).

## Ranking Criteria

Each paper scored on 4 axes (1-5 each, max 20):

- **Readiness** (1-5): How close to submittable right now?
- **Impact** (1-5): How significant is the contribution if published?
- **Honesty** (1-5): How confident are we in the claims? (Papers with unverified metrics score low)
- **Urgency** (1-5): Is there a deadline, competitive risk, or timing window?

---

## TIER 1: Submit First (Score 16-20)

| Rank | Paper | R | I | H | U | Total | Venue | Effort to Submit |
|------|-------|---|---|---|---|-------|-------|-----------------|
| **1** | **P1 — Symthaea HAI** | 5 | 5 | 5 | 4 | **19** | PLoS Comp Bio | VERIFIED READY — submit now |
| **2** | **P4 — K-Index Unified Framework** | 5 | 4 | 5 | 3 | **17** | Neural Computation | FIXED — 9 defects resolved, submit now |
| **3** | **P2 — Zero-TrustML** | 4 | 5 | 5 | 4 | **18** | ACM CCS (Apr 29) | Needs verification + anonymization |

### Why These Three First

**P1 (HAI)**: VERIFIED READY (Feb 17 audit). All numbers confirmed against codebase. 19pp, 6/6 figures, cover letter, PDF compiled, code traceability. Novel HDC+FEP integration with no competing work. Claims are honest — explicitly labels "Phi proxy" and documents gaps. No blockers. **Submit.**

**P4 (K-Index Framework)**: Had 9 blocking defects — ALL FIXED (Feb 17). Now 17pp + 10pp supplementary, 6 embedded figures, cover letter addressed to Dr. Sejnowski, 18 wired citations, 36/36 tests, conflict of interest disclosed. **Submit.**

**P2 (Zero-TrustML)**: First empirical Byzantine detection beyond 33% barrier. 100% detection at 35% BFT with 0% FPR. 3 figures with generation scripts. Needs anonymization pass (~1 week) + deep verification audit. ACM CCS deadline April 29.

### Downgraded from Tier 1

**~~HK-1 (Historical K-Index)~~**: **INTEGRITY FAILURE** (Feb 17 audit). Claims 43 civilizations but only 4 validated in codebase. R²=0.991 not derivable from any output. All 4 figures from synthetic data (seed=42). Cannot submit. See below.

---

## TIER 2: Submit Within 2-4 Weeks (Score 13-16)

| Rank | Paper | R | I | H | U | Total | Venue | Effort to Submit |
|------|-------|---|---|---|---|-------|-------|-----------------|
| **5** | **P3 — Coherence-Guided Control** | 4 | 4 | 5 | 3 | **16** | PLoS Comp Bio | ~3 hours (embed figures) |
| **6** | **HK-2 — Coordination Collapse** | 4 | 5 | 4 | 4 | **17** | PNAS | ~1 week (polish) |
| **7** | **P10 — Topology of Collective** | 4 | 3 | 5 | 2 | **14** | Frontiers Comp Neuro | ~1 day (verify bib) |
| **8** | **P11 — Developmental Pathway** | 4 | 3 | 5 | 2 | **14** | Neural Networks | ~1 day (verify bib) |
| **9** | **P14 — Kosmic K-Vector (7D)** | 4 | 4 | 5 | 3 | **16** | Neurosci. of Consciousness | ~2 weeks (run experiments) |
| **10** | **P9 — Coherence Corridors** | 3 | 3 | 5 | 2 | **13** | PLoS ONE | ~2 days (wire citations) |

### Notes

**P3**: Most rigorous kosmic-lab paper — BCa CIs, Cliff's delta, preregistered. Only blocker is figures referenced via `../logs/` paths. Fix paths, recompile, submit.

**HK-2**: 39 civilizations, collapse threshold prediction +/-15 years. 95% complete. Pair with HK-1 — submit to PNAS after Nature submission.

**P10/P11**: Compact, complete, honest claims. Easy wins. Submit to lower-tier journals for publication momentum.

**P14**: The 7D consciousness framework is novel and well-grounded. "Commons Paradox" (r=-0.44) is a genuine finding. Run the planned experiments, submit.

---

## TIER 3: 1-2 Months of Work (Score 10-13)

| Rank | Paper | R | I | H | U | Total | Venue | Effort |
|------|-------|---|---|---|---|-------|-------|--------|
| **11** | **P19 — K_Topo (LLM Closure)** | 1 | 5 | 1 | 3 | **10** | ~~Science~~ TBD | Methodology unreliable — needs complete redo |
| **12** | **P18 — Learned HDC Genomics** | 1 | 4 | 5 | 3 | **13** | Bioinformatics | 3-4 weeks (write manuscript) |
| **13** | **HK-2B — Golden Threshold** | 4 | 5 | 3 | 3 | **15** | Phys. Rev. Letters | 2 weeks (ensure rigor) |
| **14** | **P16 — PoGQ Whitepaper** | 3 | 4 | 5 | 3 | **15** | MLSys / ICML | 4-6 weeks (write body) |
| **15** | **P6 — Phenomenal Signature** | 3 | 4 | 4 | 2 | **13** | arXiv preprint | 2-3 weeks (complete results) |
| **16** | **P5 — Mycelix MLSys** | 2 | 4 | 5 | 3 | **14** | MLSys 2026 | 3-4 weeks (expand sections) |
| **17** | **P15 — Unified Indices** | 3 | 3 | 5 | 2 | **13** | PLoS Comp Bio | 1-2 weeks (integrate corrections) |
| **18** | **HK-9 — Coordination Contagion** | 3 | 4 | 4 | 2 | **13** | Nature HB | 2-3 weeks (generate figs) |
| **19** | **HK-10 — Micro-K Framework** | 3 | 4 | 4 | 2 | **13** | Org. Science | 2-3 weeks (generate figs) |

### Notes

**P19 (K_Topo)**: **DOWNGRADED** (Feb 17 stress test). Honesty score dropped 4→1. The "140x divergence" was produced by inconsistent methodology: different formulas for GPT-4o vs Claude, different conversation lengths (40 vs 20), asymmetric error exclusion, and metric failing its own validation. Not fabrication, but deeply unreliable. **Must redesign methodology and re-run from scratch before writing manuscript.** If the finding reproduces with clean methodology, this returns to high priority.

**P18 (Learned HDC)**: Benchmarks complete, 94.5% accuracy, 40x faster than DNABERT. No competing work in this exact niche (contrastive pre-training for HDC genomics). Already in Rust. Just needs a manuscript.

**HK-2B (Golden Threshold)**: phi^-2 = 0.382 with 9 independent derivations is extraordinary but will invite "numerology" criticism. Honesty score 3 because the convergence of 9 derivations to the same value needs external validation. Submit AFTER HK-2 establishes empirical credibility.

---

## TIER 4: Significant Work Required (Score 7-10)

| Rank | Paper | R | I | H | U | Total | Venue | Notes |
|------|-------|---|---|---|---|-------|-------|-------|
| **20** | **P7 — Master Equation** | 3 | 5 | 2 | 2 | **12** | ~~Nature Neurosci~~ TBD | BLOCKED on Phi metric audit |
| **21** | **P12 — Temporal Topology** | 3 | 3 | 2 | 2 | **10** | Network Neurosci | Needs full reframe from Phi to lambda2 |
| **22** | **P17 — MATL Whitepaper** | 3 | 3 | 5 | 2 | **13** | IEEE TDSC | Polish after PoGQ accepted |
| **23** | **HK-3 — Modern Fragility** | 3 | 5 | 4 | 3 | **15** | Science / Nature | 90% but needs careful framing |
| **24** | **HK-2D — Capacity Gap** | 3 | 4 | 4 | 2 | **13** | PNAS | First draft, needs polish |
| **25** | **P13 — Adversarial Perturbations** | 2 | 4 | 4 | 1 | **11** | Science (re-target?) | Needs complete rewrite of intro + discussion |
| **26** | **P8 — HAI NeurIPS** | 3 | 4 | 5 | 3 | **15** | NeurIPS 2026 (~May) | Only if targeting NeurIPS; sync from P1 |
| **27** | **HK-11 — Modernization Paradox** | 3 | 4 | 4 | 2 | **13** | Complexity | 80%, needs figures |
| **28** | **HK-12 — Fermi Paradox** | 3 | 4 | 3 | 2 | **12** | Astrobiology | 75%, speculative but publishable |

### Notes

**P7 (Master Equation)**: Would be Rank 5-6 if the Phi metric were verified. But with the confirmed lambda2 mismatch in the same codebase, submitting to Nature Neuroscience claiming IIT Phi is a reputational risk. Must audit the actual computation first. If lambda2: reframe (drops impact), or run actual Phi (weeks of work + intractability).

**HK-3 (Modern Fragility)**: "USA, UK, France below collapse threshold" is a headline-grabbing claim. But it's also the kind of claim that will draw intense scrutiny. Needs very careful framing and bulletproof methodology. High reward, high risk.

---

## TIER 5: Backlog / Long-Term / Integrity Rework (Score <7)

| Paper | Score | Notes |
|-------|-------|-------|
| **HK-1 (Historical K-Index)** | **5** | **INTEGRITY FAILURE.** R=1,I=5,H=1,U=3. Needs real pipeline for 43 civs, real figures, verified R². 4-8 weeks. |
| HK-0 (Grand Synthesis) | 8 | Capstone — publish after 3+ papers accepted |
| HK-2C (17 Laws) | 8 | Publish after HK-2 + HK-2B accepted |
| HK-4 (Regional Divergence) | 7 | 50% framework, needs substantial work |
| HK-5 (Climate Gap) | 7 | 50% framework |
| HK-6 (Recovery Mechanisms) | 6 | 40% framework |
| HK-7 (AI Governance) | 7 | 40% but timely topic |
| HK-8 (Policy Framework) | 6 | Capstone for K-Index series |
| Phi-Lab Satellites (02-15) | 5 each | All depend on P7 acceptance; all at ~30% |
| Mycelix root-notes drafts | 4 each | Supporting material, not primary submissions |
| ERC Philosophical Works | 3 | Book/monograph format, not journal papers |

---

## Recommended Submission Schedule

### Month 1 (Weeks 1-4)

| Week | Submit | Paper | Venue |
|------|--------|-------|-------|
| 1 | SUBMIT | P1 (HAI) — VERIFIED | PLoS Comp Bio + arXiv |
| 1 | SUBMIT | P4 (K-Index Framework) — FIXED | Neural Computation + arXiv |
| 2 | AUDIT + SUBMIT | P2 (Zero-TrustML) | ACM CCS (Apr 29) |
| 2 | AUDIT + SUBMIT | P3 (Coherence-Guided) | PLoS Comp Bio |
| 3 | SUBMIT | P10 (Topology Collective) | Frontiers Comp Neuro |
| 3 | SUBMIT | P11 (Developmental) | Neural Networks |
| 3 | SUBMIT | P9 (Corridors) | PLoS ONE |
| 4 | AUDIT | HK-2 (Collapse) — verify data first | PNAS (if audit passes) |

**Month 1 output: 7-8 papers submitted** (revised from 9 — HK-1 removed, HK-2 contingent on audit)

### Month 2 (Weeks 5-8)

| Week | Action | Paper | Venue |
|------|--------|-------|-------|
| 5 | METHODOLOGY REDO | P19 (K_Topo) — redesign + re-run | Hold for clean data |
| 5-6 | COMPLETE + SUBMIT | P14 (K-Vector 7D) | Neurosci. of Consciousness |
| 6-7 | WRITE + SUBMIT | P18 (Learned HDC) | Bioinformatics |
| 7-8 | COMPLETE + SUBMIT | P6 (Phenomenal Signature) | arXiv preprint |
| 8 | SUBMIT (if HK-2 passed audit) | HK-2B (Golden Threshold) | Physical Review Letters |

**Month 2 output: 3-5 more papers submitted (10-13 total)** (revised — P19 deferred until clean methodology)

### Month 3+ (Weeks 9-12)

- P16 (PoGQ) → MLSys/ICML
- P5 (Mycelix MLSys) → MLSys 2026
- P7 (Master Equation) → TBD after Phi audit
- HK-9, HK-10, HK-11 → various journals
- P8 (HAI NeurIPS) → NeurIPS 2026 if May deadline

**Month 3 output: 5-7 more papers (19-21 total)**

---

## Decision Points (Updated Post-Audit)

1. **P2/P3 verification**: Next papers to deep audit. P2 has ACM CCS deadline Apr 29 — audit by end of Week 2.
2. **P19 (K_Topo) methodology redesign**: Design clean, consistent methodology BEFORE re-running experiments. Get informal peer review of methodology.
3. **HK series data pipeline**: Build real analysis pipeline for civilizations. Decide: fix HK-1 first, or verify HK-2 independently?
4. **P7 (Master Equation) Phi audit**: Run `phi_real.rs` with debug logging to confirm what metric feeds into C(t). This gates all phi-lab work.
5. **NeurIPS 2026**: Check exact deadline. If May, need P8 ready by April.
6. **Co-author outreach**: Consider reaching out to Wendt/Turchin for HK papers — but ONLY after data integrity is confirmed.
7. **arXiv strategy**: Preprint P1 immediately for timestamp priority. P4 to arXiv after Neural Computation submission.

---

## The Honest Assessment (Revised Feb 17)

We have 52+ papers and 0 submissions. The bottleneck is not writing — it's **integrity verification**.

The Phase 0 audit revealed that "ready" is not the same as "verified ready":
- P1 was genuinely ready (minor number updates only)
- P4 had **9 blocking defects** that would have caused immediate desk rejection
- HK-1 had **critical integrity failures** (synthetic data, unsupported claims)
- P19's headline finding was **methodologically unreliable**

**The lesson**: Every paper needs a deep verification audit before submission. The extra day of checking saves months of embarrassment.

The top 2 verified papers (P1, P4) should be submitted NOW. P2 and P3 need the same audit treatment before they go out. The HK series needs its data pipeline rebuilt from scratch.

The Phi metric issue is real and must be addressed honestly, but it only affects 2-3 papers. The other 49+ are clean.

**Priority #1: Submit P1, P4, and HK-1 this week. Everything else follows.**
