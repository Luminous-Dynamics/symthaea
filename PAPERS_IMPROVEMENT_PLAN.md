# Paper Portfolio — Improvement Plan

> Created: 2026-02-17
> Goal: Maximize submissions over next 8 weeks with honest, high-quality publications

---

## Phase 1: Immediate Submissions (Week 1-2)

### 1A. Submit HAI to PLoS Comp Bio
- **Effort**: ~2 hours (final proofread + submit)
- **Paper**: Symthaea HAI (#1)
- **Status**: 98% complete. Cover letter done, PDF compiled, all 6 figures, code traceability doc.
- **Tasks**:
  - [ ] Final proofread of hai_paper.tex (check for stale numbers — LOC count, test count)
  - [ ] Verify all figure labels match text references
  - [ ] Confirm supplementary.md benchmark commands still work
  - [ ] Upload to PLoS Comp Bio submission system
  - [ ] Simultaneously post to arXiv as preprint
- **Blockers**: None
- **Impact**: HIGH — novel HDC+FEP integration, no competing work

### 1B. Submit K-Index Framework to Neural Computation
- **Effort**: ~4 hours (review + submit)
- **Paper**: Kosmic-Lab Neural Computation (#4)
- **Status**: Most polished package. 12pp + 9pp supplementary, 6 figures, cover letter, 25 refs.
- **Tasks**:
  - [ ] Review manuscript for accuracy (are all 46/46 tests still passing?)
  - [ ] Verify cover letter is addressed to Neural Computation editor
  - [ ] Check figure quality (PNG resolution, PDF vector)
  - [ ] Submit
- **Blockers**: None
- **Impact**: MEDIUM-HIGH — unifies 5 consciousness theories

### 1C. Prepare Zero-TrustML for IEEE S&P
- **Effort**: ~1 week
- **Paper**: Mycelix Zero-TrustML (#2)
- **Tasks**:
  - [ ] Final proofread all 8 section files
  - [ ] Anonymize: remove author names, affiliations, self-citations
  - [ ] Verify `stats.tex` auto-generation works (or hardcode values)
  - [ ] Format check against IEEE S&P template requirements
  - [ ] Compile final PDF, verify page count (target 12pp)
  - [ ] Check if IEEE S&P 2026 deadline has passed — if so, retarget to USENIX Security 2026 or ACM CCS
- **Blockers**: Need to confirm current CFP deadlines
- **Impact**: HIGH — first empirical Byzantine detection beyond 33%

---

## Phase 2: Quick Wins (Week 2-3)

### 2A. Fix Kosmic Paper 2 Figures
- **Effort**: ~3 hours
- **Paper**: Coherence-Guided Control (#3)
- **Tasks**:
  - [ ] Locate all figures referenced via `../logs/` paths
  - [ ] Copy/convert to paper2/figures/ directory (PDF + PNG)
  - [ ] Update `\includegraphics` paths in manuscript.tex
  - [ ] Recompile PDF, verify all figures render
  - [ ] Submit to PLoS Comp Bio (or PLoS ONE if Comp Bio rejects HAI)
- **Blockers**: Figure source files must exist
- **Impact**: MEDIUM — good paper, rigorous stats

### 2B. Wire Citations in Kosmic Paper 1
- **Effort**: ~2 hours
- **Paper**: Coherence Corridors (#9)
- **Tasks**:
  - [ ] Add `\cite{}` commands throughout manuscript
  - [ ] Verify bibliography file has relevant entries (may need additions beyond shared bib)
  - [ ] Fix author field (currently "[Authors TBD]")
  - [ ] Recompile, submit to PLoS ONE
- **Blockers**: None
- **Impact**: LOW-MEDIUM — establishes corridor discovery methodology

### 2C. Sync HAI NeurIPS Variant (if targeting NeurIPS 2026)
- **Effort**: ~1 day
- **Paper**: HAI NeurIPS (#8)
- **Decision needed**: Is NeurIPS 2026 a target? Deadline typically May.
- **Tasks (if yes)**:
  - [ ] Port updated results from hai_paper.tex → hai_neurips2026.tex
  - [ ] Condense to 8-page NeurIPS format
  - [ ] Add all 6 figures (currently only 1)
  - [ ] Compile, review
- **Tasks (if no)**: Archive or mark as "not pursuing"
- **Impact**: HIGH if NeurIPS, but may conflict with PLoS submission timing

---

## Phase 3: Completion Work (Week 3-5)

### 3A. Complete Phenomenal Signature Paper
- **Effort**: ~2 weeks
- **Paper**: Phenomenal Signature (#6)
- **Current state**: 85% — results section truncated, no figures, no discussion
- **Tasks**:
  - [ ] Complete Results section with full statistical tables
  - [ ] Generate topology heatmap figures (Layer 22 phenomenal corridor)
  - [ ] Write Discussion section (interpret d=0.69 finding, limitations, implications)
  - [ ] Write Conclusion
  - [ ] Format references properly
  - [ ] Select venue: Neuroscience of Consciousness, Consciousness and Cognition, or arXiv preprint
  - [ ] Supporting material exists: `layer21_phenomenal_structure.md` (31K), `corridor_depth_hypothesis.md` (19K)
- **Blockers**: Need to re-run analysis if results were lost
- **Impact**: MEDIUM-HIGH — novel TDA approach to transformer consciousness

### 3B. Expand Mycelix MLSys Paper
- **Effort**: ~2 weeks
- **Paper**: Mycelix MLSys 2026 (#5)
- **Current state**: 70% — sections 1-3 complete, 4-7 need expansion
- **Tasks**:
  - [ ] Expand Section 4 (Implementation): Holochain details, PoGQ algorithm pseudocode
  - [ ] Expand Section 5 (Evaluation): Generate 4-6 figures (latency distributions, detection curves, scalability)
  - [ ] Expand Section 6 (Discussion): Compare with FLTrust, Krum, Multi-Krum in detail
  - [ ] Expand Section 7 (Conclusion): Future work
  - [ ] Complete bibliography (cite all competing systems papers)
  - [ ] Check MLSys 2026 deadline (~October 2026?)
- **Relationship to Zero-TrustML**: These are different papers — Zero-TrustML is architecture-focused (IEEE security), MLSys is systems-focused (performance). Minimal overlap.
- **Impact**: MEDIUM — MLSys is prestigious for systems work

### 3C. Complete Master Equation Paper
- **Effort**: ~3 weeks
- **Paper**: Master Equation (#7)
- **Current state**: 85% — text complete, 0/8 figures, 20/60 references
- **Tasks**:
  - [ ] **CRITICAL**: Verify Phi metric accuracy (is C(t) using real IIT Phi or lambda2 proxy?)
  - [ ] Generate 8 figures (specifications exist in phi-lab docs)
  - [ ] Expand bibliography from 20 → ~60 references
  - [ ] Write appendices
  - [ ] Internal review focusing on claims vs evidence
  - [ ] If Phi is accurate: submit to Nature Neuroscience
  - [ ] If Phi is lambda2: reframe honestly (like Temporal Topology reframe)
- **Blockers**: Metric verification is critical path
- **Impact**: VERY HIGH if claims hold — unified consciousness framework at Nature Neuroscience

---

## Phase 4: Rescue Operations (Week 5-8)

### 4A. Reframe Temporal Topology Paper
- **Effort**: ~1 week
- **Paper**: Temporal Topology (#12)
- **Current state**: 90% complete but BLOCKED by lambda2/Phi mismatch
- **Reframed version exists**: `letter_reframed.md`
- **Tasks**:
  - [ ] Systematically replace all IIT Phi references with spectral lambda2 throughout `arxiv/main.tex`
  - [ ] Rewrite abstract: "spectral connectivity in continuous-time neural architectures"
  - [ ] Update all figure captions
  - [ ] Remove Tononi/IIT consciousness claims
  - [ ] Retarget venue: Network Neuroscience, IEEE TNNLS, or Journal of Complex Networks
  - [ ] The core finding (99.2% optimal 3D small-world) is still novel and publishable
- **Impact**: MEDIUM — strong spectral topology result once honestly framed

### 4B. Reconcile Kosmic Paper 5 (Science)
- **Effort**: ~3 days investigation + variable
- **Paper**: Adversarial Perturbations (#13)
- **Current state**: Submitted Nov 2025, but LaTeX source is incomplete
- **Tasks**:
  - [ ] Determine if Science has responded (check email for decision)
  - [ ] If rejected: Complete the LaTeX source (add introduction, expand discussion, fill method gaps)
  - [ ] If under review: Note discrepancy for future revision
  - [ ] If accepted (unlikely given source state): Celebrate, then fix source
  - [ ] The adversarial finding (FGSM +136% K enhancement, d=4.4) is genuinely interesting
- **Impact**: Variable — depends on Science decision

### 4C. Submit Kosmic Papers 3 & 4
- **Effort**: ~1 day each
- **Papers**: Topology of Collective (#10), Developmental Pathway (#11)
- **Current state**: Complete, compact (5pp and 4pp)
- **Tasks**:
  - [ ] Verify bibliographies are customized (not just shared bib)
  - [ ] Add 1-2 figures if data visualizations exist
  - [ ] Submit Paper 3 → Frontiers in Computational Neuroscience
  - [ ] Submit Paper 4 → Neural Networks
- **Impact**: LOW-MEDIUM each, but easy wins

---

## Cross-Cutting Improvements

### A. Metric Audit (URGENT)
- [ ] Audit every paper that mentions "Phi" or "IIT" — verify the code actually computes IIT Phi vs lambda2
- [ ] Papers affected: #7 (Master Equation), #12 (Temporal Topology — already identified)
- [ ] Symthaea HAI (#1) uses "Phi proxy" and explicitly documents the gap — this is honest
- [ ] Create a METRIC_VERIFICATION.md documenting what each paper actually measures

### B. Figure Standardization
- [ ] Ensure every paper has figures self-contained in its directory (not `../logs/` references)
- [ ] Standard format: PDF for LaTeX, PNG for review, SVG for editability
- [ ] Minimum resolution: 300 DPI for raster, vector preferred

### C. Bibliography Deduplication
- [ ] Kosmic papers 1-5 share identical bib file — customize per paper
- [ ] Create a master bibliography (`MASTER_REFERENCES.bib`) shared across projects
- [ ] Each paper should reference only what it cites

### D. Submission Tracking
- [ ] Track actual submission dates, revision requests, acceptance/rejection
- [ ] Update PAPERS.md status column as papers progress through review
- [ ] Set calendar reminders for revision deadlines

---

## Recommended Submission Order

| Week | Action | Paper | Venue |
|------|--------|-------|-------|
| 1 | Submit | HAI (#1) | PLoS Comp Bio + arXiv |
| 1 | Submit | K-Index (#4) | Neural Computation |
| 2 | Submit | Zero-TrustML (#2) | IEEE S&P / USENIX |
| 2 | Submit | Coherence-Guided (#3) | PLoS Comp Bio |
| 3 | Submit | Papers 3 & 4 (#10, #11) | Frontiers, Neural Networks |
| 3 | Submit | Coherence Corridors (#9) | PLoS ONE |
| 4-5 | Complete + Submit | Phenomenal Signature (#6) | arXiv preprint |
| 4-5 | Complete + Submit | MLSys (#5) | MLSys 2026 |
| 5-6 | Complete + Submit | Master Equation (#7) | Nature Neuroscience |
| 6-7 | Reframe + Submit | Temporal Topology (#12) | Network Neuroscience |
| 8 | Decide | NeurIPS variant (#8) | NeurIPS 2026 (May deadline) |

**Estimated output in 8 weeks**: 8-10 papers submitted across 8+ venues

---

## Phase 5: Historical K-Index Revival (Week 6-12)

The archived 16-paper K-Index series represents a major body of work. Key decision: **un-archive and actively develop, or leave as archive?**

### 5A. Confirm Nature Submission Status (URGENT)
- **Effort**: 30 minutes
- **Paper**: HK-1 (Historical K-Index)
- **Tasks**:
  - [ ] Check email for Nature submission confirmation, reviewer response, or desk rejection
  - [ ] If not actually submitted: Submit NOW (package is 100% ready, Dec 27 2025)
  - [ ] If desk rejected: Redirect to PNAS or Nature Human Behaviour immediately
  - [ ] If under review: Prepare response to reviewers
- **Impact**: VERY HIGH — this is the flagship paper for the entire K-Index program

### 5B. Submit HK-2 (Coordination Collapse) to PNAS
- **Effort**: ~1 week (polish + submit)
- **Paper**: HK-2 (95% complete)
- **Tasks**:
  - [ ] Final review of 39-civilization dataset
  - [ ] Verify theta=0.375 threshold predictions
  - [ ] Ensure figures are publication-quality
  - [ ] Submit to PNAS (or Nature Human Behaviour as backup)
- **Dependency**: Ideally after HK-1 acceptance, but can submit independently
- **Impact**: HIGH — predicts collapse threshold with +/-15 year accuracy

### 5C. Prepare HK-2B (Golden Threshold) for Physical Review Letters
- **Effort**: ~2 weeks
- **Paper**: HK-2B (90% complete, 9 convergent derivations)
- **Tasks**:
  - [ ] Ensure all 9 independent derivations are rigorous (evolutionary game theory, percolation, bifurcation, info theory, thermodynamics, ESS, network science, MaxEnt, renormalization group)
  - [ ] Monte Carlo validation (10^5 samples, p < 10^-17 claimed)
  - [ ] Format for PRL (short format, ~3,500 words)
  - [ ] Consider: submit AFTER HK-2 to avoid "numerology" criticism
- **Impact**: VERY HIGH if accepted — phi^-2 as universal social threshold is extraordinary

### 5D. Complete HK-9 and HK-10 (Substantial Manuscripts)
- **Effort**: ~2 weeks each
- **Papers**: Coordination Contagion (~3K words) and Micro-K Framework (~3.5K words)
- **Status**: Both have substantial real content (not scaffolding), but need figures (0/paper)
- **Tasks**:
  - [ ] Generate figures for HK-9 (network contagion maps, super-spreader analysis)
  - [ ] Generate figures for HK-10 (org/city/team K-scores, scale-independence validation)
  - [ ] Polish prose and verify citations
  - [ ] Submit HK-9 to Nature Human Behaviour or equivalent
  - [ ] Submit HK-10 to Organization Science or Academy of Management Journal
- **Impact**: MEDIUM-HIGH each — novel applications of K-Index

### 5E. Long-Term K-Index Plan (Month 3-6)
- [ ] HK-3 (Modern Fragility) → complete and submit to Science/Nature (high impact if timed with current events)
- [ ] HK-2D (Capacity-Actualization Gap) → submit to PNAS
- [ ] HK-11 (Modernization Paradox) → submit to Complexity or PNAS
- [ ] HK-12 (Fermi Paradox) → submit to Astrobiology or PNAS
- [ ] HK-4 through HK-8 remain framework-stage — deprioritize until Acts 1-2 published
- [ ] Co-author outreach: Contact Wendt, Busemeyer, Turchin (emails drafted in strategy docs)

---

## Phase 6: Phi-Lab Satellite Pipeline (Month 3-8)

### 6A. Phi Metric Audit (PREREQUISITE)
- **Effort**: 1-2 days
- Before any phi-lab paper beyond P7, audit ALL Phi claims
- [ ] Verify Master Equation (P7) uses real IIT Phi vs lambda2 proxy
- [ ] Document in METRIC_VERIFICATION.md

### 6B. Satellite Paper Pipeline
- All 15 satellites depend on Paper 01 (Master Equation) acceptance
- **Priority order** (after P7 accepted):
  1. Paper 02 (AI Consciousness) → Nature Machine Intelligence
  2. Paper 12 (Computational Implementation) → PLoS Comp Bio (closest to Symthaea)
  3. Paper 04 (Binding Problem) → Neural Computation
  4. Paper 03 (Clinical Validation) → PNAS
  5. Remainder: monthly cadence
- **Combined satellites** may be more publishable than individual papers

---

## Phase 7: Grants & Funding (Parallel Track)

### 7A. NSF SBIR Phase 1
- [ ] Review draft at `.archive-2026-02-01/grants/NSF_SBIR_PHASE1_2026.md`
- [ ] Update with latest results (paper acceptance would strengthen enormously)
- [ ] Check 2026 deadline window

### 7B. Protocol Labs Grant
- [ ] Review draft + cover letter
- [ ] Align with Holochain/Mycelix FL work
- [ ] Submit when appropriate

---

## Revised Submission Timeline (Comprehensive)

| Week | Action | Paper | Venue |
|------|--------|-------|-------|
| **1** | Submit | P1 (HAI) | PLoS Comp Bio + arXiv |
| **1** | Submit | P4 (K-Index Framework) | Neural Computation |
| **1** | Confirm | HK-1 (Historical K-Index) | Nature (check status) |
| **2** | Submit | P2 (Zero-TrustML) | IEEE S&P / USENIX |
| **2** | Submit | P3 (Coherence-Guided) | PLoS Comp Bio |
| **3** | Submit | P10, P11 (Kosmic 3 & 4) | Frontiers, Neural Networks |
| **3** | Submit | P9 (Corridors) | PLoS ONE |
| **4** | Submit | HK-2 (Collapse) | PNAS |
| **4-5** | Complete + Submit | P6 (Phenomenal Signature) | arXiv preprint |
| **4-5** | Complete + Submit | P5 (MLSys) | MLSys 2026 |
| **5-6** | Complete + Submit | P7 (Master Equation) | Nature Neuroscience |
| **6-7** | Reframe + Submit | P12 (Temporal Topology) | Network Neuroscience |
| **7-8** | Submit | HK-2B (Golden Threshold) | Physical Review Letters |
| **8** | Complete + Submit | HK-9, HK-10 | Nature HB, Org Science |
| **8** | Decide | P8 (NeurIPS variant) | NeurIPS 2026 |
| **9-12** | Complete + Submit | HK-3, HK-2D, HK-11, HK-12 | Science, PNAS, etc. |

**Estimated output in 12 weeks**: 14-18 papers submitted across 12+ venues

---

## Decision Points Needed

1. **HK-1 Nature status**: Was it actually submitted? This is the #1 priority to confirm.
2. **Un-archive decision**: Move historical-K papers out of `.archive-2026-02-01/` back to active development?
3. **NeurIPS 2026**: Are we targeting this? If yes, need HAI variant ready by ~May.
4. **Dual PLoS submission**: HAI and Coherence-Guided to same journal — stagger or simultaneous?
5. **Master Equation Phi verification**: Gates Nature Neuroscience vs more modest venue.
6. **Paper 5 (Science)**: What's the actual response? Check email.
7. **arXiv preprints**: Preprint everything, or hold some for journal exclusivity?
8. **Co-author outreach**: Ready to contact Wendt, Busemeyer, Turchin? Emails are drafted.
9. **Grant timing**: Submit NSF/Protocol Labs before or after paper acceptances?
10. **ERC philosophical works**: Should the Grand Compendium or Codex be published as a book/monograph?

---

*Plan designed for maximum output with honest, verifiable claims.*
