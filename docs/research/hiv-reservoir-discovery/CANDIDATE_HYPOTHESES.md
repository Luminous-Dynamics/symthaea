# Initial Candidate Research Questions for HIV Reservoir Discovery

**Status:** Literature-grounded candidates for testing; none is a new Symthaea discovery.  
**Reviewed:** 2026-10-29

The purpose of this register is to start specific, falsifiable work while avoiding the common mistake of treating a synthesis of published findings as a novel mechanism. Each item has a published prior, an alternative explanation, a feasible first test and a clear limit on what that test can establish.

## H1 — Reservoir endpoints may have different predictors

**Prior evidence:** The Dryad source paper reports associations between host gene expression and total HIV DNA / residual HIV RNA, while noting that intact-DNA associations were not detected and that many intact-DNA observations were undetectable. Reviews of reservoir assays also emphasize that total DNA, intact proviruses, transcriptional activity, inducibility and replication competence are related but distinct properties.

**Question:** Do feature sets that predict total HIV DNA or residual transcription also predict intact HIV DNA, or do their apparent successes fail to transfer across endpoint definitions?

**Alternative explanations:** Differences could arise from assay sensitivity, censoring, sample size, measurement error or cohort selection rather than genuinely distinct biology.

**First test:** Reproduce the published total-DNA and residual-RNA analyses from the Dryad release; audit intact-DNA detection limits; report each endpoint separately; run prespecified sensitivity analyses for nondetects. Compare incremental held-out prediction only when the sample count supports it.

**Falsifiable prediction:** The relative ranking and calibration of candidate host-state signals will differ by endpoint, and using total-DNA or residual-RNA success as a surrogate for intact/replication-competent reservoir performance will fail at least one declared transfer test.

**What this cannot prove:** That one endpoint is the uniquely correct cure surrogate, or that a model predicts rebound or replication competence.

**Evidence:** https://doi.org/10.5061/dryad.k3j9kd5dw ; https://doi.org/10.1038/s41591-020-1022-1 ; https://pubmed.ncbi.nlm.nih.gov/37811964/

## H2 — Mitochondrial state may help distinguish latency-associated transcriptional states

**Prior evidence:** Man et al. report mitochondrial-associated expression differences between CD4+ T-cell populations with abortive versus elongated HIV transcripts; the work used flow-FISH and reports that enhancing mitochondrial function induced HIV transcription in PBMCs. The associated public transcriptomic series is GSE260588.

**Question:** Can a prespecified mitochondrial gene-set score distinguish the study's transcriptional states on donor-aware held-out data, beyond a simple global-expression or cell-state baseline?

**Alternative explanations:** The signature might reflect activation, cell composition, treatment status, sample handling or donor-specific effects, rather than a latency-specific program.

**First test:** Verify GEO sample metadata and donor-level replication structure. Reproduce the published differential-expression direction and compare a prespecified mitochondrial score against an ordinary linear/PCA baseline using donor-grouped validation. Validate direction in an independent compatible dataset only if the measured state and biological unit match.

**Falsifiable prediction:** A mitochondrial-state feature adds reproducible out-of-donor discrimination for the declared transcript state, and the gain survives a shuffled-feature negative control. If only an in-sample association appears, the predictive hypothesis fails.

**What this cannot prove:** That the same manipulation is safe or clinically beneficial, or that transcript reactivation alone eliminates a reservoir.

**Evidence:** https://doi.org/10.1371/journal.ppat.1012822 ; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE260588

## H3 — Memory-cell differentiation and reservoir seeding are temporally distinct from reservoir maintenance

**Prior evidence:** Gao et al. report that BACH2-mediated memory differentiation contributes to seeding long-lived reservoir cells, with pre-ART inflammation impeding this differentiation process. The reported study includes model-system and ex-vivo evidence. This is prior published biology, not a new claim by Symthaea.

**Question:** Can the discovery system correctly distinguish evidence about early reservoir seeding from evidence about ongoing maintenance under long-term ART?

**Alternative explanations:** A cross-sectional association in ART-suppressed participants may reflect current cell composition, clonal expansion or selection and cannot establish what happened near ART initiation.

**First test:** Create a timeline- and evidence-type-aware literature table. Keep pre-ART, early ART, long-term suppressive ART, ex-vivo perturbation and humanized-model results separate. Test whether the system assigns a causal-seeding claim to cross-sectional late-ART data; it should abstain unless longitudinal or perturbation evidence supports that inference.

**Falsifiable prediction:** A blinded evaluation with deliberately confounded timing evidence reveals whether Symthaea's evidence model can avoid upgrading an association measured after reservoir establishment into a claim about reservoir seeding.

**What this cannot prove:** That intervening on BACH2 is a safe or effective cure strategy.

**Evidence:** https://doi.org/10.1016/j.xcrm.2025.102311 ; https://pubmed.ncbi.nlm.nih.gov/40845840/

## H4 — Host-cell state and proviral context may interact, but must be measured in linked samples

**Prior evidence:** Separate studies examine host-cell mitochondrial/transcriptional states, integration context and heterogeneous HIV expression among clones with authentic intact latent proviruses. They motivate a joint-state hypothesis, but separate cohorts cannot establish a patient-level interaction.

**Question:** Given matched observations, does a model containing both host-cell state and proviral integration/chromatin context generalize better than either feature family alone?

**Alternative explanations:** Any apparent combination advantage may be driven by batch, donor identity, cell model or incompatible assays. An additive model may be sufficient; an interaction may not exist.

**First test:** First establish whether a public dataset genuinely links host-cell state and proviral context for the same biological unit. If not, do not fit a joint model across unpaired datasets. Use the separate studies only for qualitative triangulation and record the matched-data requirement as an open dependency.

**Falsifiable prediction:** In a suitable linked dataset, a preregistered interaction term improves donor-held-out prediction over both component-only and additive models, with stable direction and a negative-control failure. Otherwise the interaction hypothesis is not supported.

**What this cannot prove:** A causal interaction or therapeutic target without functional validation.

**Evidence:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE310195 ; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268010 ; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE260588

## Priority order

1. **H1 first:** the primary Dryad cohort is the fastest route to a faithful reproduction and a precise assay/endpoint audit.
2. **H2 second:** use GSE260588 as a separate, published latency-state reproduction task; small donor counts mean uncertainty and donor-aware splitting are mandatory.
3. **H3 in parallel as an evidence-reasoning challenge:** this tests whether Symthaea respects temporal and causal limits before the system generates new biomedical narratives.
4. **H4 only after matched-data availability is confirmed:** do not spend modelling effort trying to infer interactions from unpaired datasets.

## Stop conditions

Reject or downgrade a candidate when its source metadata do not support the biological unit, the original finding cannot be reproduced, the signal disappears under an appropriate split, the negative control performs similarly, or an alternative explanation accounts for the result. “Plausible” is not a status category.
