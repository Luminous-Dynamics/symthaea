# Industry and State-of-the-Art Comparison: HIV Reservoir Discovery

**Status:** Evaluation strategy, not a claim that Symthaea currently matches or exceeds external systems.  
**Reviewed:** 2026-10-29

## The correct comparison

There is no single model called “SOTA for HIV-cure discovery.” Reservoir-size prediction, cell-state classification, gene-perturbation response prediction, protein structure, and prospective therapeutic discovery are different tasks with different evidence standards.

Symthaea must be compared with the strongest methods for each specific task, on the same outcome, eligible samples, information budget and validation split. A result on transcriptomic prediction cannot establish superiority in molecular structure prediction, experimental hit discovery, or clinical development.

Recent computational-biology benchmarking reinforces this discipline. A Nature Methods perspective published 4 September 2026 emphasizes transparent evidence of reproducibility, generalisation and replicability when evaluating biomedical foundation models: https://doi.org/10.1038/s41592-026-03182-y. A 2025 benchmark found that five foundation models and two other deep-learning approaches did not outperform deliberately simple baselines for predicting the effects of unseen genetic perturbations in the tested datasets: https://doi.org/10.1038/s41592-025-02772-6. This is evidence about those perturbation tasks, not proof that foundation models fail on every HIV task. It is a strong reason not to assume that model complexity is an advantage.

## Comparator matrix

| Task | Required comparators | What would count as a gain | What would not count |
|---|---|---|---|
| Participant-level reservoir measurement | Null/mean and median; clinical-only regression; source-paper statistical pipeline; ridge/elastic net; a bounded nonlinear model | Better out-of-fold error and calibration on identical participant splits, then replication in a compatible independent cohort | Higher training fit, a single random split, or rediscovery of the source paper's published genes |
| Latent versus transcriptionally active state | Donor-aware differential-expression and pseudobulk baselines; PCA/linear models; prespecified state classifiers | Donor-held-out performance and pathway direction that remains stable across study/model shifts | Treating thousands of cells from a few donors as thousands of independent people |
| Perturbation or latency-reversal response | No-change/mean-shift baseline; linear and PCA-based methods; study-specific established method; foundation-model embeddings only where their input assumptions are satisfied | Better prediction of perturbation-specific effects on unseen perturbations or conditions, with controls that remove general activation shifts | A score that rewards predicting a generic activated-versus-control difference rather than the response to the perturbation |
| Mechanistic hypothesis ranking | Curated literature/evidence baseline; simple pathway/network analysis; blinded expert review | More independently supported, discriminating hypotheses per fixed review or experiment budget; calibrated confidence; retained negative findings | Fluent explanations, citation counts, or self-rated confidence |
| Next-experiment selection | Random; uncertainty-only; simple heuristic; best existing lab-planning baseline | Greater prespecified information gain or more useful resolution of competing hypotheses under the same budget, followed by prospective evaluation | Calling a heuristic “expected information gain” without calibrated probabilities |
| Molecular structure / binding (only when a specific molecular hypothesis needs it) | Task-appropriate established structure or binding models, including relevant current tools | Better independent structural/functional validation for that particular molecular question | Comparing structure-model scores with HIV reservoir-prediction scores |

## Where industry tools fit

- **Statistical genomics:** DESeq2/limma-voom-style methods and covariate-adjusted models are the essential starting baseline, not straw men to beat.
- **Single-cell and transcriptomic foundation models:** scGPT, Geneformer, scFoundation, UCE and other compatible models can be tested as feature extractors if the data modality, gene vocabulary, licences and compute requirements match. A model must not be excluded or included after seeing test results.
- **Genomic and molecular foundation models:** models such as Evo 2, AlphaGenome and AlphaFold 3 address sequence, regulatory or structural questions. They may become useful components for specific hypotheses, but they are not direct competitors on the primary patient-level reservoir regression task.
- **Drug-discovery organisations:** the most meaningful comparison is prospective experimental utility—validated hypotheses or hits, reproducibility, selectivity, safety evidence, time and cost—not the number of papers, model parameters, generated candidates or internal benchmark z-scores. Closed proprietary pipelines cannot be honestly ranked against a public-data-only Symthaea experiment without a shared blind evaluation.

References for the task boundaries:
- Evo 2, *Nature* (2026): https://doi.org/10.1038/s41586-026-10176-5
- AlphaGenome, *Nature* (2026): https://doi.org/10.1038/s41586-025-10014-0
- AlphaFold 3, *Nature* (2024): https://doi.org/10.1038/s41586-024-07487-w
- Industry-led call for critical evaluation of ML in real-world drug discovery, *Nature Machine Intelligence* (2024): https://doi.org/10.1038/s42256-024-00911-w
- Systema: perturbation-response metrics can overestimate performance by rewarding systematic differences rather than perturbation-specific effects, *Nature Biotechnology* (2025): https://doi.org/10.1038/s41587-025-02777-8

## The Symthaea-specific opportunity

Symthaea's credible differentiation is not “we have a bigger neural model.” It is an end-to-end, auditable reasoning loop:

1. Retrieve and preserve the source evidence.
2. Build a small set of competing mechanistic explanations, each with explicit uncertainty and contradictory findings.
3. Predict a result that distinguishes the explanations.
4. Select a next experiment against fixed-budget baselines.
5. Compare the predeclared prediction with the independently observed outcome.
6. Update the hypothesis state and preserve negative evidence.

Mycelix can preserve provenance, review and evidence history; Symtropy can later simulate a calibrated mechanistic model. Neither component should be treated as a substitute for actual biological validation.

## Proposed gates for “better than SOTA”

These thresholds are prospective targets, not achieved results:

### Gate A — Fair internal prediction
- The clinical-only and strongest established statistical/ML baselines are implemented and reported.
- Symthaea improves primary out-of-fold RMSE by at least 10% against the predeclared clinical-only baseline.
- The paired uncertainty interval excludes no improvement, calibration remains acceptable, and permutation/shuffle controls fail as expected.
- The best baseline is reported even if Symthaea loses.

### Gate B — Generalisation
- The direction and useful magnitude of improvement survive a compatible study-held-out test.
- Training/preprocessing are isolated from the held-out cohort; study identity and donor structure are respected.
- A non-comparable dataset is not relabelled “external validation” simply because it is public.

### Gate C — Discovery utility
- Candidate hypotheses are adjudicated blind to the model's preferred ranking where practical.
- The top-k hypotheses outperform random and simple literature/pathway baselines by a preregistered metric.
- Confidence scores are calibrated and wrong hypotheses are retained as explicit failures.
- Improvement in experiment selection is measured prospectively when researchers can supply the outcome.

### Gate D — Biological evidence
- Independent evidence supports the same specific mechanism.
- Any causal claim requires a suitable functional experiment conducted by qualified biomedical researchers under appropriate oversight.
- Any intervention claim needs its own safety, efficacy and translational evidence. Computational scores cannot establish an HIV cure.

## Practical compute strategy

Start with CPU-feasible statistical baselines and data audits. Add pretrained embeddings only after the simple baseline is reproducible and only where model size, licensing and input format are compatible with the available hardware. Use model ablations to test whether HDC, temporal dynamics, evidence retrieval or hypothesis comparison adds measurable value. If a component adds no value, record that result and simplify the system.

The benchmark should reward scientific reliability per unit cost, not raw parameter count.
