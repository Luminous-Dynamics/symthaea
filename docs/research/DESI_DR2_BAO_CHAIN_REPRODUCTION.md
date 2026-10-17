# DESI DR2 BAO Chain Reproduction Contract

**Status:** proposed external-data benchmark; no DESI artifacts have been downloaded or verified by this repository yet.  
**Purpose:** establish a reproducible bridge from analytic cosmology tests to a published, real-data posterior product.  
**Scope:** audit official released ΛCDM + all-DESI-BAO posterior chains first; independently rerunning the inference is a separate, harder gate.

## Official source and frozen target

Official DESI documentation identifies the DR2 BAO cosmology repository and documents its layout:

- Release overview: https://data.desi.lbl.gov/doc/papers/
- DR2 BAO cosmology products: https://data.desi.lbl.gov/doc/papers/dr2/
- Official product README: https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/README.html
- Product directory: https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/
- Published SHA-256 manifest: https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/dr2_vac_dr2_bao-cosmo-params_v1.0.sha256sum
- Model/data directory for this benchmark: https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base/desi-bao-all/

The official product README describes the **cobaya/** directory as containing MCMC chains and posterior samples, and **iminuit/** as containing posterior maximization results. It identifies the model/data directory convention and says the DR2 BAO likelihoods used for inference are available in the public CobayaSampler repository. The four chain files and associated metadata in the target directory are individually listed in the public index.

**Target selection:** **cobaya/base/desi-bao-all/**

This fixes the baseline model family to the base ΛCDM model and uses the combination of all DESI DR2 BAO redshift bins. The official directory's artifacts include:

- **chain.1.txt** through **chain.4.txt**: four posterior-sample chain files.
- **chain.input.yaml** and **chain.updated.yaml**: the recorded Cobaya run settings and expanded/updated settings.
- **chain.margestats**: published marginal statistics for cross-checking.
- **chain.covmat**: proposal covariance metadata where present.
- **chain.checkpoint**: sampler convergence/checkpoint summary where present.

Do not treat the existence of these outputs as evidence that this repository has parsed, validated, or reproduced them.

## Gate D0 — source integrity and acquisition

1. Download the exact desi-bao-all files needed for the benchmark.
2. Verify every downloaded file against the official SHA-256 manifest before parsing. Retain the exact manifest bytes, its own SHA-256, retrieval timestamp, and target URL in the run manifest.
3. Record file lengths and computed hashes in a Mycelix-DeSci scientific benchmark manifest.
4. Fail closed on missing files, checksum mismatches, duplicate/ambiguous paths, or unexpected column formats.
5. Preserve the original files as immutable inputs; generated summaries must never overwrite them.

**Acceptance:** every used byte matches the published digest. A successful HTTP response or a filename that looks correct is not sufficient.

## Gate D1 — posterior-chain audit (not a new inference run)

Read the official header and sample format rather than hardcoding assumed column names or parameter ordering. Establish whether chain weights, log-posterior columns, burn-in, and derived parameters are present and how the associated settings specify them.

The first implementation should calculate from the released chain samples:

- Per-chain row count, finite-value checks, and accepted/valid weight totals where the format defines weights.
- Parameter-wise weighted mean, variance, and requested quantiles using a documented, tested quantile convention.
- Effective sample size where a valid weighting/autocorrelation definition is available; otherwise report it as unavailable rather than inventing it.
- Convergence diagnostics only when their inputs and assumptions can be established from the actual sample layout.
- Comparison against chain.margestats and chain.checkpoint, preserving discrepancies and differences in quantile conventions.

**Acceptance:** a deterministic parser test on synthetic fixtures; explicit schema/header validation; summary values reproducible from frozen inputs; independent agreement with the published marginal-statistics file within a tolerance fixed before the run.

A chain summary audit is useful but is **not** the same as rerunning the DESI likelihood or reproducing a paper's scientific inference.

## Gate D2 — independent summary reproduction

Use a separately implemented checker to re-read the frozen chain files and regenerate summary statistics. Keep the parser/summary implementations independent enough that they do not merely share the same quantile or weighting bug.

**Acceptance:** independently generated summaries agree within the preregistered tolerance, the verifier consumes the exact same checksum-verified inputs, and the Mycelix manifest records distinct verifier identity and organization through the canonical signed-event/authority workflow.

## Gate D3 — inference reproduction

Only after D0–D2 pass:

1. Extract the exact chain.input.yaml and chain.updated.yaml files.
2. Record the precise likelihood versions, external priors, solver versions, nuisance parameters, parameter priors, and any post-processing/importance-sampling steps.
3. Use the official DR2 BAO likelihood products and a supported inference framework such as Cobaya with an independent, validated theory backend.
4. Run a deterministic smoke configuration on synthetic data before a full sampling run.
5. Compare posterior constraints to the released official results using a preregistered metric and tolerance; report convergence, effective sample size, and chain-to-chain diagnostics.
6. Retain the official chains as a reference, not as a substitute for the independent run.

**Acceptance:** exact settings are reproduced, posterior constraints agree within the declared numerical/statistical tolerance, and disagreements are documented instead of tuned away.

This gate is an inference reproduction, not necessarily independent reanalysis of the underlying DR2 spectra. The official DESI pages state that the underlying DR2 spectra/redshifts have not yet been released, while the BAO chain and posterior-maximization products are available.

## Negative controls

- Corrupt one byte in a copied input and verify checksum failure.
- Remove one chain file and verify an explicit missing-input failure.
- Change the header or reorder columns and verify schema mismatch instead of silently assigning incorrect parameter names.
- Insert NaN/Infinity into a synthetic chain and verify explicit rejection or documented row handling.
- Use an incorrect weight column and verify the fixture test catches the altered summaries.
- Deliberately relax the comparison tolerance and verify the run manifest records the changed contract/version rather than rewriting prior outcomes.
- Mark a non-matching summary as reproduced and verify Mycelix manifest validation rejects it.

## Planned implementation boundary

Symthaea should first provide the cosmological background equations and validated analytic reference tests. The DESI chain reader/auditor should be a separate benchmark component with minimal dependencies and deterministic output. It must not claim to implement a Boltzmann solver or a full cosmological likelihood.

Mycelix-DeSci should store the exact input hashes, output summary, settings, benchmark status, comparison tolerance, and independent reproduction attestation. Symtropy is not required for this chain-audit milestone; its role begins with validated dynamical simulations, not posterior-file parsing.

## External provenance

The official DESI product README describes:
- cobaya/ as MCMC chains and posterior samples;
- iminuit/ as posterior-maximization outputs;
- chain sample files as ASCII posterior samples;
- associated input/updated YAML and marginal statistics files;
- the official SHA-256 manifest at the product root.

Source: https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/README.html

This research contract records a proposed target and gate definitions; it does not imply that any D0–D3 gate has passed.
