# Fusion HDC Manuscript: Final Quality Review

**Reviewer:** Automated quality review
**Date:** 2026-03-19
**Manuscript:** `FUSION_HDC_MANUSCRIPT.md`

---

## Errors Found and Fixed

1. **Tokamak name typo (2 instances).** "Alcatel C-Mod" corrected to "Alcator C-Mod" on lines 148 (Section 3.6, Dataset) and 296 (Section 5.5, Limitations). The tokamak is named Alcator (Alto Campo Toro), not Alcatel.

2. **Misattributed citation for Johnson-Lindenstrauss lemma.** Section 3.1 cited [6] (Rahimi and Recht 2009, "Random Kitchen Sinks") for the J-L lemma. Corrected to [26] (Johnson and Lindenstrauss 1984), which is the actual J-L paper already in the reference list.

3. **Author name errors in reference [26].** "W. A. Johnson and R. L. Lindenstrauss" corrected to "W. B. Johnson and J. Lindenstrauss" (William B. Johnson and Joram Lindenstrauss).

---

## Data Accuracy: PASS

All numerical values verified against ground truth:

- **V1:** AUC 0.778, F1 0.219, Precision 0.143, Recall 0.461, Specificity 0.952, FP 2,494, TP 417, FN 488, TN 49,152 -- all correct.
- **V2:** AUC 0.820, F1 0.409, Precision 0.350, Recall 0.491, Specificity 0.984, FP 823, TP 444, FN 461, TN 50,823 -- all correct.
- **V3:** AUC 0.830, F1 0.434, Precision 0.397, Recall 0.480, Specificity 0.987, FP 660, TP 434, FN 471, TN 50,986 -- all correct.
- **Dataset:** 264,385 samples, 2,333 shots, 78 disrupted, 1,867 train / 466 test -- all correct.
- **O(1) ratio:** 1.534x across 7 orders of magnitude -- correct.
- **Energy:** 57x (desktop) / 247x (laptop) vs GPT-3 -- correct (35.5/0.622 = 57.1x, 35.5/0.144 = 246.5x).
- **Lead time:** 90-820 ms -- correct.
- **Confusion matrix consistency:** All four matrices sum correctly (TP+FN = 905 disruption samples, FP+TN = 51,646 normal samples, total = 52,551).

---

## Consistency: PASS

- Table 1 values match all in-text references (abstract, Section 4.1, 4.2, 5.3, 7).
- Test set size (52,551) is consistent across Sections 3.6, 4.1, and 5.5.
- Disrupted shot percentage (78/2333 = 3.3%) stated correctly.
- Disruption sample percentage (905/52551 = 1.7%) stated correctly.
- Energy efficiency derivation is internally consistent: 104 inferences/sec at 9.57 ms latency checks out (1000/9.57 = 104.5).
- AUC deltas in Section 4.2 are correct: V1-to-V2 = +0.042, V2-to-V3 = +0.010.
- False positive reduction factor "3.8" is correct (2494/660 = 3.78).

---

## Honest Framing: PASS

- "Competitive with" used appropriately for baseline comparisons (Sections 4.1, 5.3).
- CCNN many-shot result (AUC 0.974) explicitly disclaimed as "not our comparison target" (Section 5.3).
- "Zero-training" precisely defined as zero gradient descent, with structural priors honestly acknowledged (Section 5.5).
- Limitations section is thorough: single machine, single disruption type, offline only, class imbalance, structural priors.
- Production requirements honestly stated as unmet (AUC > 0.99, FAR < 1% needed; Section 5.4).

---

## Grammar and Style: PASS

- Scientific register maintained throughout. No informal language.
- Consistent past tense for results, present tense for methods and established facts.
- Em dashes used consistently for parenthetical asides.
- Mathematical notation is consistent (bold for vectors, italic for scalars).

---

## Structure Completeness: PASS

All required sections present and properly organized:
- Abstract, Keywords
- 1. Introduction (with contributions list)
- 2. Related Work (5 subsections)
- 3. Methods (6 subsections)
- 4. Results (5 subsections)
- 5. Discussion (5 subsections including Limitations)
- 6. Future Work
- 7. Conclusions
- References (33 entries)

---

## References

**Citations in text:** 1-21, 24, 25, 26, 31, 32, 33.

**Orphaned references (defined but never cited in text):**
- [22] Heidbrink and Sadler 1994 (fast ions)
- [23] Berkery et al. 2017 (resistive wall mode)
- [27] Greenwald et al. 1988 (original density limit paper)
- [28] Murari et al. 2009 (unsupervised disruption prediction at JET)
- [29] Lechner et al. 2020 (neural circuit policies)
- [30] Kim et al. 2020 (HDCluster)

**Action needed:** Either cite these 6 references in the text or remove them. Orphaned references will be flagged by reviewers. References [27] and [28] could be naturally cited in the Introduction or Related Work. References [29] and [30] could support the HDC discussion. References [22] and [23] appear entirely tangential and should likely be removed.

---

## Figures and Tables

Three tables are present with clear data. Two figure placeholders are included:
- **[FIGURE 1]:** ROC curves -- clearly described.
- **[FIGURE 2]:** O(1) scaling plot -- clearly described.

**Action needed:** Actual figures must be generated before submission. A third figure showing the HDC encoding pipeline (Section 3) would strengthen the Methods section and is standard for architecture papers.

---

## Remaining Issues for the Author

1. **Orphaned references** (6 entries) -- cite or remove as described above.
2. **Generate actual figures** -- the two placeholders need real plots.
3. **Consider adding an architecture diagram** -- a figure showing the encoding pipeline (sensor -> basis vectors -> bundle -> reference comparison -> free energy) would significantly improve readability.
4. **Reference [6] is now uncited** -- after the J-L fix, Rahimi and Recht (2009) is no longer cited. Either cite it elsewhere (e.g., to support the random projection encoding) or remove it.
5. **DisruptionBench zero-shot comparison** -- Section 5.3 discusses the intent to compare against DisruptionBench zero-shot results, but no actual numbers from the benchmark's zero-shot regime are provided. If these numbers are available from [31], including them would substantially strengthen the comparison argument.
6. **Threshold selection on test set** -- Section 3.6 states the threshold is selected to maximize F1 "on the test set." This is methodologically problematic and will be flagged by reviewers. The threshold should be selected on a validation set separate from the test set, or via cross-validation. If the current results use the test set for threshold selection, this overfits the threshold and the reported F1 scores are optimistic. The AUC is unaffected (threshold-independent), but F1/precision/recall should be re-evaluated with a proper validation split.
7. **"78 disrupted" vs. "15 disrupted in test"** -- Section 4.5 states "14 out of 15" disrupted test shots detected. With 78 disrupted shots total and an 80/20 split, the test set should contain approximately 16 disrupted shots (78 * 0.2 = 15.6). The stated "15" is plausible but should be verified as the exact count.

---

## Overall Assessment

The manuscript is well-written, scientifically rigorous, and honestly framed. The core results are clearly presented with appropriate caveats. The comparison to supervised approaches is notably fair -- the author resists the temptation to overclaim by explicitly distinguishing between zero-shot and many-shot regimes and by positioning the work correctly relative to DisruptionBench.

The methodology concern about threshold selection on the test set (item 6 above) is the most substantive issue. If the threshold was genuinely optimized on the test set, the F1/precision/recall numbers are biased upward and should be re-run with a held-out validation partition. The AUC results, which are the primary claim, are unaffected.

**Submission readiness:** Near-ready, pending figure generation and resolution of the threshold selection methodology.

---

## Recommended Target Venue

**Primary recommendation: arXiv preprint first**, then submit to **Nuclear Fusion**.

Rationale:
- **arXiv first** (cs.LG + physics.plasm-ph cross-list) establishes priority and allows community feedback, particularly from the DisruptionBench team at MIT PSFC. This is especially valuable given the zero-shot framing -- feedback from the supervised ML community will sharpen the comparison.
- **Nuclear Fusion** is the natural home for the final version. It publishes both physics and applied ML work on disruption prediction (references [2], [12], [16], [17], [18], [19] are all in NF). The zero-training angle is novel for this venue and addresses a widely recognized gap (ITER deployability). The honest framing of limitations and the practical deployment discussion (Section 5.4) align well with NF's applied focus.
- **Physical Review E** would be appropriate if the focus shifted toward the mathematical properties of the HDC encoding (the O(1) scaling proof, the geometric analysis in Section 5.1), but the current manuscript is more applied than theoretical.
- An alternative second venue is **Journal of Fusion Energy**, where DisruptionBench itself was published [31], which would place the work in direct dialogue with the benchmark.
