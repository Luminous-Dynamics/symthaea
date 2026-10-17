# symthaea-cosmology-bench

Dependency-light parsing and audit helpers for released Cobaya posterior chain text files.

## First target: DESI DR2 BAO

The accompanying research contract identifies the official DESI DR2 BAO ΛCDM + all-BAO chain product and its published SHA-256 manifest:

- https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/README.html
- https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/dr2_vac_dr2_bao-cosmo-params_v1.0.sha256sum
- https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base/desi-bao-all/

The CLI reads an ASCII chain file whose first recognized comment header starts with “# weight minuslogpost”, then accepts a named column and emits JSON. It rejects malformed rows, duplicate columns, non-finite values, negative sample weights, absent parameters and zero-total-weight chains.

Example command:

    cargo run -p symthaea-cosmology-bench --bin symthaea-cosmology-chain-audit -- chain.1.txt H0

Output reports per-file row count, total and positive-weight sample count, weighted mean/population variance, weighted empirical q16/q50/q84, and weight-only effective sample size.

## Scientific boundaries

- Weighted quantiles use a documented left-continuous inverse empirical CDF. They are **not guaranteed to match GetDist's KDE-smoothed marginal limits**; comparisons must account for the quantile convention.
- The effective sample size is computed from sample weights only. It is not autocorrelation-corrected MCMC effective sample size.
- This code audits released samples; it does not run Cobaya, execute a DESI likelihood, reproduce the posterior, or validate the underlying spectrum data.
- Do not use it to claim the checksum-audit gate passed until the original files have been downloaded and verified against the official SHA-256 manifest.
- Input files and outputs should be bound to the Mycelix-DeSci scientific benchmark manifest, with independent summary verification kept separate from the primary parser.
