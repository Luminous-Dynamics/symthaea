# Avian Communication Research Contract v1

**Status:** research plan; all benchmark thresholds below are proposed targets, not measured results or released capabilities.

**Purpose:** establish a falsifiable path from passive bird-audio analysis to evidence-backed claims about signal function and, much later, reciprocal communication. Do not build a bird-to-English translator as the primary abstraction.

## 1. Paradigm shift

Move the research question from **“What sentence does this call mean?”** to:

> Given the acoustic event, its social/ecological context, and a competing set of hypotheses, what observable event is this signal associated with, what would distinguish competing explanations, and does the receiver's behavior change under an appropriate intervention?

This has five consequences:

1. **Discover structure before assigning meaning.** Learn repeatable units, boundaries, sequences, timing and variation without requiring words or human semantic primes.
2. **Separate acoustic identity from function.** Correctly naming a species or clustering a call does not establish what it communicates.
3. **Make context earn its contribution.** Compare context-aware predictions against acoustic-only, identity-only, location-only, shuffled-context and temporal-permutation controls.
4. **Use causal evidence for reference claims.** Natural correlations can generate hypotheses; they do not by themselves establish that a signal refers to a particular external event.
5. **Treat interaction as a later, governed stage.** Passive recordings come first. Playback requires species/population-specific welfare review, stop conditions and reviewed `ExpressionAuthorization`; model confidence is never authorization.

This plan composes the existing `symthaea-stt::discovery`, `symthaea-communication::animal`, `EvaluationPlan`, `DatasetManifest`, `BenchmarkReport`, and immutable run-bundle verification. Do not create a parallel evidence or capability ladder.

## 2. Existing capability ladder and claim ceiling

Use the repository's `CapabilityLevel` semantics:

| Level | Permitted claim |
|---|---|
| `Signal` | A signal/event was detected under the declared conditions. |
| `Unit` | A repeatable acoustic unit or segmentation was found. |
| `Structure` | Recurrence, sequence, timing or transition regularity generalizes to held-out data. |
| `Reference` | An externally grounded referent/function is supported by preregistered intervention and independent replication. |
| `Intent` | A stronger claim about communicative intent has dedicated evidence and alternative explanations are tested; do not infer it from structure alone. |
| `Dialogue` | A bidirectional exchange is context-sensitive and mutually contingent, beats matched/yoked controls, and replicates independently. |

No level implies access to subjective experience. Keep observed call, behavioral outcome, ecological context, model hypothesis and human interpretation as separate fields. Uncertainty or disagreement must not be collapsed into a single confidence label.

## 3. Benchmark suites and baselines

### Suite A — Detection and taxonomic context

Use a pinned subset of **BEANS** for standardized animal-audio detection/classification. Where bird species labels and licenses allow, compare against a suitable published taxonomic detector (e.g. BirdNET) as a reference baseline, not as a semantic oracle. Report precision, recall, PR-AUC and false positives at both event and recording/site level.

The global BirdNET assessment used 4,224 one-minute recordings from 67 sites and reported substantial geographic and species-level variation, with recall generally more heterogeneous than precision. That argues against random clip-only splits and aggregate-only reporting.

### Suite B — Unsupervised unit discovery

Run Symthaea's acoustic unit-discovery pipeline beside:
- a simple energy/spectral segmentation baseline;
- a pinned acoustic embedding + clustering baseline, if its model and license are recorded;
- shuffled-time and noise/negative-control inputs.

Score boundary precision/recall/F1 at a preregistered timing tolerance where strong boundary annotations exist. Also report bootstrap segmentation stability, cluster purity/ARI where valid reference labels exist, recurrence precision, and sensitivity to threshold/recording conditions. A visually persuasive spectrogram is not a metric.

### Suite C — Context-conditioned prediction

Use a defined observable target, such as the next annotated call class or a blinded, pre-defined receiver behavior. Compare the full model with:
- acoustic-only;
- context-only;
- identity-only;
- location-only;
- shuffled-context;
- temporal-permutation;
- held-out-individual and held-out-site evaluation.

All data splits must be grouped by original recording/session and must not share source segments. Train/test individuals and sites must be disjoint for the corresponding generalization claim. If individual identities are unknown, say so and do not claim individual-level generalization.

### Suite D — Grounded function

Only after passive models produce a preregistered, discriminating prediction should a species-appropriate field protocol be considered. Choose one primary behavioral endpoint, randomize candidate/control conditions, use blinded outcome coders where possible, document non-responses and missing observations, and determine sample size from a prospective power analysis. Establish a minimum practically important effect *before* exposing confirmatory outcomes.

No uncontrolled playback, nesting-season experimentation by default, repeated exposure to force a response, or unattended autonomous broadcast. Welfare concerns or adverse behavior trigger the protocol's stop conditions.

### Suite E — Reciprocal interaction

This is a later milestone, not an MVP. A successful exchange must show that the next model response depends on the bird's latest response and context, rather than merely replaying a fixed sequence. Compare against fixed-playback, randomized-playback and yoked-session controls. Require an independently replicated effect before claiming dialogue.

## 4. Proposed benchmark targets

These are initial engineering goals to evaluate and refine during pilot data characterization; they are **not results** and do not substitute for a preregistered study.

| Dimension | Initial target / gate |
|---|---|
| Dataset integrity | 100% of evaluated samples have stable IDs, source/license attribution, content hashes and immutable split manifests; reject cross-split sample, recording/session, individual or site overlap for each claimed scope. |
| Detection/classification | Publish event-level and recording/site-level PR-AUC, precision, recall, false-positive rate and per-taxon support. Beat the strongest relevant pinned baseline with a paired bootstrap 95% CI excluding zero; never hide weak taxa in a macro average. |
| Unit segmentation | Boundary F1 ≥ 0.75 at a task-specific, preregistered tolerance on strongly annotated data, plus bootstrap stability ≥ 0.80. Label these proposed floors until pilot annotation quality is assessed. |
| Context value-add | Balanced accuracy at least 0.05 absolute above the strongest required non-contextual control on the held-out evaluation, with paired 95% CI for the difference excluding zero. Use a suitable alternative primary metric if the outcome is not categorical, chosen before testing. |
| Calibration | Expected calibration error ≤ 0.10 and positive Brier skill versus a declared baseline where probabilistic predictions are emitted; provide reliability plots and sample counts. |
| Unknown/shifted inputs | Publish coverage-risk curves and out-of-distribution metrics; abstain rather than assigning a familiar label when the sample is outside declared support. No absolute threshold is released before calibration on held-out conditions. |
| Grounded function | Preregistered, powered intervention whose primary effect exceeds the predeclared minimum practically important effect, with a confidence interval excluding the null and independent replication. The required effect size is species/task-specific; do not invent one universally. |
| Dialogue | Context-contingent response beats fixed, shuffled and yoked controls under preregistration and independent replication; welfare protocol remains satisfied. No dialogue claim based only on call imitation or a single response. |
| Reproducibility | Exact source/model/toolchain/configuration, dataset manifest, hardware and per-sample results are pinned in an immutable verified run bundle; missing or stale evidence fails closed. |

If a target proves unrealistic on an independently characterized corpus, amend the threshold before looking at confirmatory results and preserve the old target and rationale in the record. Never tune the gate to make a favorite model pass.

## 5. Split, leakage and negative-control requirements

- Partition original recordings and related session/source material before extracting clips or windows.
- Where available, provide three separate evaluations: held-out individuals, held-out sites, and held-out time periods. Do not treat these as interchangeable.
- Keep multiple clips from the same underlying vocalization on one side of the split.
- Freeze label ontology and annotation rules before confirmatory scoring; record disagreement and ambiguous labels.
- Include no-call/silence, background-only and non-target sound controls when represented in the task.
- Preserve negative results, null predictions, abstentions, failed model runs and data exclusions.
- Keep taxonomic identification, acoustic clustering, context prediction and communication claims in separate report fields.

## 6. Data and license discipline

Candidate starting points:
- BEANS benchmark and datasets: https://arxiv.org/abs/2210.12300 and https://github.com/earthspecies/beans
- BirdNET model documentation (including model scope and license): https://birdnet-team.github.io/BirdNET-Analyzer/stable/models.html
- Global BirdNET evaluation: https://doi.org/10.1016/j.ecolind.2025.114550

Pin the exact dataset/version and each artifact's license before use. Public availability is not a blanket redistribution or commercial-use grant. In particular, record the actual model license and intended use for every baseline. Do not commit raw wildlife recordings into the source repository without explicit redistribution rights; publish source references, manifests and permitted derived features instead.

## 7. Implementation sequence

1. **Contract hardening (this tranche):** require all registered context controls; reject incomplete/duplicate/blank split identities and non-finite temporal-permutation evidence.
2. **Passive corpus audit:** choose one legally usable bird corpus, inspect taxa/site/recording imbalance, pin a manifest and split by source session/site/identity where known. No training claim yet.
3. **Baseline harness:** emit BEANS-compatible detection results and a separate unit-discovery report using Symthaea's existing immutable run bundle.
4. **Context prediction:** preregister one observable target and run every required control on held-out identities/sites/time.
5. **Expert review:** reconcile annotation uncertainty and review negative controls before using words like “meaning.”
6. **Intervention proposal:** only after the passive prediction gate passes, submit a species-specific welfare-reviewed protocol.
7. **Reciprocal communication:** only after grounded-function evidence replicates.

## 8. Claim boundary for this research contract

A passing software test establishes only that the declared validator rejects malformed or incomplete evaluation evidence. It does **not** establish that Symthaea recognizes bird calls, understands their meaning, communicates with birds, or is ready to conduct playback. Those are separate empirical milestones and require real, legally usable recordings and appropriately reviewed field evidence.
