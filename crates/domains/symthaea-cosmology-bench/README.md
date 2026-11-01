# symthaea-cosmology-bench

Dependency-light parsing and audit helpers for released Cobaya posterior chain text files.

## First target: DESI DR2 BAO

The accompanying research contract identifies the official DESI DR2 BAO ΛCDM + all-BAO chain product and its published SHA-256 manifest:

- https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/README.html
- https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/dr2_vac_dr2_bao-cosmo-params_v1.0.sha256sum
- https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base/desi-bao-all/

The official audit mode requires the checksum manifest and explicit relative-path/local-file mappings. Each file is hashed and compared against the exact manifest entry before the very same in-memory bytes are parsed. If one digest mismatches, no weighted summary is emitted. All chain headers must match exactly before samples are combined. Duplicate manifest-relative paths and duplicate local files (including aliases resolving to the same file) are rejected to prevent accidental double-counting. The separately named `--summary-unverified` mode exists for synthetic/local exploration only and labels every input as not checksum-verified.

Verify one downloaded artifact separately, when needed for intake triage:

    cargo run -p symthaea-cosmology-bench --bin symthaea-cosmology-chain-audit -- --verify-manifest data/bao-cosmo-params/dr2_vac_dr2_bao-cosmo-params_v1.0.sha256sum cobaya/base/desi-bao-all/chain.1.txt data/bao-cosmo-params/cobaya/base/desi-bao-all/chain.1.txt

**Official checksum-gated audit:** one command verifies all chain inputs before parsing or summarizing any of them. The relative paths are taken verbatim from the publisher's manifest:

    cargo run -p symthaea-cosmology-bench --bin symthaea-cosmology-chain-audit -- --audit-manifest data/bao-cosmo-params/dr2_vac_dr2_bao-cosmo-params_v1.0.sha256sum H0 cobaya/base/desi-bao-all/chain.1.txt=data/bao-cosmo-params/cobaya/base/desi-bao-all/chain.1.txt cobaya/base/desi-bao-all/chain.2.txt=data/bao-cosmo-params/cobaya/base/desi-bao-all/chain.2.txt cobaya/base/desi-bao-all/chain.3.txt=data/bao-cosmo-params/cobaya/base/desi-bao-all/chain.3.txt cobaya/base/desi-bao-all/chain.4.txt=data/bao-cosmo-params/cobaya/base/desi-bao-all/chain.4.txt

For the small local fixture or other unverified input, use the explicitly labeled mode:

    cargo run -p symthaea-cosmology-bench --bin symthaea-cosmology-chain-audit -- --summary-unverified H0 ./synthetic-chain.txt

The JSON summary records the checksum-manifest path and SHA-256, each local path and exact publisher-relative path, expected and actual SHA-256, byte size, and a per-input `checksum_verified` boolean. The hash is computed from the same in-memory bytes passed to the parser. A manifest supplied by a caller is itself recorded and hashed but not independently authenticated by this tool; obtain it from the official publisher over a trusted channel.

## Scientific boundaries

- Weighted quantiles use a documented left-continuous inverse empirical CDF. They are **not guaranteed to match GetDist's KDE-smoothed marginal limits**; comparisons must account for the quantile convention.
- The effective sample size is computed from sample weights only. It is not autocorrelation-corrected MCMC effective sample size.
- This code audits released samples; it does not run Cobaya, execute a DESI likelihood, reproduce the posterior, or validate the underlying spectrum data.
- Do not use it to claim the checksum-audit gate passed until the original files have been downloaded and verified against the official SHA-256 manifest.
- Input files and outputs should be bound to the Mycelix-DeSci scientific benchmark manifest, with independent summary verification kept separate from the primary parser.
