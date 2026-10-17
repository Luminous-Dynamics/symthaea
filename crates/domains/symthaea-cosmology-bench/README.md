# symthaea-cosmology-bench

Dependency-light parsing and audit helpers for released Cobaya posterior chain text files.

## First target: DESI DR2 BAO

The accompanying research contract identifies the official DESI DR2 BAO ΛCDM + all-BAO chain product and its published SHA-256 manifest:

- https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/README.html
- https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/dr2_vac_dr2_bao-cosmo-params_v1.0.sha256sum
- https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base/desi-bao-all/

The CLI accepts a named column followed by one or more ASCII chain files whose recognized comment header starts with “# weight minuslogpost”. It combines files only when their full column header and ordering match exactly, then emits one JSON summary. It rejects malformed rows, duplicate columns, non-finite values, negative sample weights, absent parameters, incompatible chain headers and zero-total-weight chains.

Verify one downloaded artifact directly against the official SHA-256 manifest, without manually copying the digest:

    cargo run -p symthaea-cosmology-bench --bin symthaea-cosmology-chain-audit -- --verify-manifest data/bao-cosmo-params/dr2_vac_dr2_bao-cosmo-params_v1.0.sha256sum cobaya/base/desi-bao-all/chain.1.txt data/bao-cosmo-params/cobaya/base/desi-bao-all/chain.1.txt

Repeat for chain.2.txt, chain.3.txt, chain.4.txt, chain.input.yaml, chain.updated.yaml, chain.checkpoint, chain.margestats, and chain.covmat as required. Use the relative path exactly as it appears in the official manifest; a missing or duplicate entry fails closed.

Audit and combine the four compatible DESI chain files for a selected parameter only after checksum verification:

    cargo run -p symthaea-cosmology-bench --bin symthaea-cosmology-chain-audit -- H0 data/bao-cosmo-params/cobaya/base/desi-bao-all/chain.1.txt data/bao-cosmo-params/cobaya/base/desi-bao-all/chain.2.txt data/bao-cosmo-params/cobaya/base/desi-bao-all/chain.3.txt data/bao-cosmo-params/cobaya/base/desi-bao-all/chain.4.txt

The JSON summary includes each exact parsed input's SHA-256 and byte size, plus row count, total and positive-weight sample count, weighted mean/population variance, weighted empirical q16/q50/q84, and weight-only effective sample size. The hashes are computed from the same in-memory bytes passed to the parser.

## Scientific boundaries

- Weighted quantiles use a documented left-continuous inverse empirical CDF. They are **not guaranteed to match GetDist's KDE-smoothed marginal limits**; comparisons must account for the quantile convention.
- The effective sample size is computed from sample weights only. It is not autocorrelation-corrected MCMC effective sample size.
- This code audits released samples; it does not run Cobaya, execute a DESI likelihood, reproduce the posterior, or validate the underlying spectrum data.
- Do not use it to claim the checksum-audit gate passed until the original files have been downloaded and verified against the official SHA-256 manifest.
- Input files and outputs should be bound to the Mycelix-DeSci scientific benchmark manifest, with independent summary verification kept separate from the primary parser.
