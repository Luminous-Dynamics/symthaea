# PeptoneBench Pinned Source Audit and Reproduction Gate

**Audit date:** 2026-10-11  
**Status:** static source audit at an immutable upstream revision; no package installation, dataset download, model prediction, or reference benchmark execution was performed. This is an implementation-risk report, not a scientific result.

## Exact source identity

- Repository: https://github.com/PeptoneLtd/peptonebench
- Audited commit: [`f161b5220a52ebd1a715c91a68beef125101608e`](https://github.com/PeptoneLtd/peptonebench/commit/f161b5220a52ebd1a715c91a68beef125101608e)
- Upstream commit message/date: “Fix symlinks in forward-models docker image”, 2026-03-24.
- License file at this revision: Apache-2.0. This is the code license, not a substitute for checking licenses/terms on each dataset asset.
- File blob SHA-1s returned by GitHub at the pinned revision:
  - `README.md`: `b74091f19aed795d85c3db18a0fa2c1467482528`
  - `pyproject.toml`: `9dfc7705eba1fc48f39da02a98ad517b73e0b426`
  - `src/peptonebench/scripts/run_benchmark.py`: `dfe2791f4de27339c5c574812611fe9e6d261e31`
  - `src/peptonebench/reweighting.py`: `bda72db8b6b448368b94d2338361c6c702e1c314`
  - `src/peptonebench/config.py`: `4fc13e5c89c96d2bc26c7837d0ea09b22b544c7e`
  - `datasets/downloadDBs.sh`: `ce11ac91fed463b5f8bbb7b408aac7194566eecd`
  - `forward-models/README.md`: `d4564197bc1fb8f058e26512b1323e8723bce866`
  - `forward-models/Dockerfile`: `fb5f318a6f647b6e9bbb34de784cd9f09c59960a`
  - `forward-models/env.yaml`: `b91cef732dcc1168daece666b471cd415a0f76e8`
  - `forward-models/run_forward_model.py`: `8cb739ca61868d75cd6856821734b4c7b5f1f35a`
  - `LICENSE`: `d4979f5870d4f55ebe12ddc01bad61e05c3a4df0`

The blob hashes identify individual file contents. The commit SHA identifies the source tree revision. Neither verifies a locally downloaded data archive or a runtime environment.

## Source findings with experimental consequences

### 1. This is an observable-conditioned reweighting benchmark

The entry point accepts generated ensemble files and forward-model outputs. The reweighting implementation takes per-conformer observable deltas and, for SAXS, experimental intensity values. It computes an unweighted `RMSE`, optimizes weights using those same observable deltas, and reports `rew_RMSE` after weighting. For SAXS, a scale factor is fit against experimental intensity inside the score calculation.

This is a legitimate *ensemble compatibility / refinement* workflow. Its post-reweighting score is **not independent held-out predictive accuracy** for the observables used during fitting. Preserve at least these result classes separately:

1. pre-reweighting compatibility score;
2. post-reweighting score on the fitting observable (label it in-sample/refinement);
3. genuinely held-out-observable score, only where the measurement was not used in generation, selection, fitting, or tuning.

A better post-reweight score alone cannot establish that Symthaea produced a better raw ensemble.

### 2. Effective sample size is an absolute default, not an automatically normalized budget

The CLI defaults `--ess-target` to `10.0`; the current call forwards it as the absolute ESS threshold. Although the documentation describes a typical target as roughly 10% of sample count, the default does not compute that percentage. Since ensemble sizes can vary, record both the absolute target and `ESS / valid_conformer_count`. Before comparing proteins, preregister whether the target is a fixed absolute ESS or a fixed fraction, and keep that rule constant.

### 3. Dependency metadata is not a complete environment lock

`pyproject.toml` requires Python >=3.11 and declares scientific dependencies using compatible-release constraints (for example `numpy~=2.3.3`, `scipy~=1.16.2`) rather than exact hashes for every resolved artifact. Forward models have separate setup/Docker instructions. An install that succeeds later may resolve a different environment.

For any reproduction, retain the resolved package inventory, Python/platform details, Docker image digest(s) for forward models, command line, environment variables (including `N_JOBS` and `PEPTONEDB_PATH`), and hashes of all inputs/outputs. Prefer an immutable image digest over a mutable tag.

### Forward-model build is a separate, less-pinned environment

At the pinned revision, the scoring package requires Python >=3.11, while `forward-models/env.yaml` targets Python 3.8 and older numerical packages. Do not combine those environments; build/run the scoring package and forward-model container as separate, versioned stages.

The forward-model Dockerfile uses a base image tag rather than an image digest, runs mutable `apt-get update`, downloads SPARTA+ and Pepsi-SAXS directly, clones Reduce without pinning a commit, and expects UCBShift model weights to be downloaded manually from Dryad as `models.tgz`. It pins a CSpred source commit, but the manually supplied model weights and other external artifacts have no hash check in the inspected build instructions. The image also downloads/extracts PeptoneDBs during build without verifying a publisher checksum.

The upstream README points to a precomputed [`Predictions.tar.gz` archive](https://zenodo.org/records/17306061/files/Predictions.tar.gz) containing generated ensembles and forward-model predictions. This offers a better **score-only reproduction lane** before attempting a full forward-model build. Its current archive size, publisher checksum, and selective-extraction path have not yet been captured in this project's evidence bundle, so do not assume the archive is small or fetch it unbounded.

A metadata-first [verified Zenodo fetcher](../../scripts/research/fetch_verified_zenodo_file.py) has been added. Its default mode reports the exact record/file size and checksum without downloading bytes. Download mode requires an explicit byte cap, verifies the provider-declared checksum and size, retains raw record metadata, computes a local SHA-256, refuses overwrite by default, and never extracts the archive. Seventeen offline/mocked tests cover caps, malformed metadata, untrusted URLs/embedded credentials, redirect-to-host checks for metadata and binary files, checksum/size mismatch, overwrite protection, and raw metadata snapshots. All 17 passed offline using mocked responses; no live Zenodo query or archive download has been performed. This fetcher has **not** yet queried Zenodo or downloaded a dataset in the present sandbox.

### 4. The dataset download helper does not verify archive identity

`datasets/downloadDBs.sh` downloads `PeptoneDBs.tar.gz` from Zenodo, extracts it, then removes the archive. The inspected script has no checksum or expected-size verification. For a controlled run, download without destructive cleanup, verify a trusted published digest if one exists (otherwise record that no publisher digest was found and compute a local SHA-256), inspect the archive listing before extraction, and hash the extracted files used in the run. A locally computed hash proves identity from then on, not publisher authenticity.

### 5. Failure can be expressed as logs/NaNs instead of a decisive run failure

The benchmark code can return rows with NaN result fields when filtering leaves no valid conformers or a reweighting failure is caught. The command's top-level loop also catches per-directory exceptions, logs the stack trace, and continues. The inspected entry point does not turn those caught errors into a nonzero process exit status. Therefore, a completed command, output CSV, or CI green state is insufficient evidence.

The adapter/verifier must classify every expected target as exactly one of:
- `PASS_WITH_VALID_METRICS`
- `FAIL_EXECUTION`
- `FAIL_INVALID_OR_MISSING_METRICS`
- `EXCLUDED_PREDECLARED`
- `NOT_RUN`

Only the first is a scored result. NaN, missing output, a caught exception, an incomplete target census, or a missing output hash must fail closed—not disappear from denominators.

### 6. Optimizer retry path requires explicit provenance

The normal minimization path is initialized deterministically by the current caller; when minimization fails, the code retries with a randomized initial point. The run record should therefore capture whether a retry occurred, optimizer message/status, all warnings, target ESS attainment, valid/dropped conformer counts, and the initial-point/retry policy. If retries are retained in the reference path, seed and record the RNG or separately test the retry path for repeatability.

## Reproduction gate (do not skip to a controller comparison)

**Gate A — source and environment**
1. Check out the exact source commit above; record `git rev-parse HEAD` and clean/dirty state.
2. Archive `pyproject.toml`, license, the relevant source files, and hashes above.
3. Build the documented forward-model environment(s) and record immutable image digests and the fully resolved Python dependency list.
4. Download only the smallest data/asset subset needed; preserve original archives, verify any published checksums, and compute local SHA-256 hashes before extraction/use.

**Gate B0 — score-only golden reference (preferred first)**
1. Query Zenodo metadata for `Predictions.tar.gz` before downloading; record the file size, published checksum, record/version DOI, and metadata response hash. Apply an explicit byte cap to any download, preserve the archive, and verify both the publisher-declared digest and local SHA-256. Do not extract the whole archive blindly; list members and select one labelled case plus the exact dataset/observable inputs it needs.
2. Run the unmodified pinned scoring package over that already-generated case. This isolates score/reweight reproduction from the separate protein-ensemble generator and forward-model stack.
3. Retain stdout/stderr, exit status, optimizer diagnostics/retry path, target-level outcome, raw output CSVs, weights, environment lock, and hashes. Repeat the case in a fresh environment; compare deterministic outputs exactly and preregister tolerances for numerical variation.
4. Reconcile every score to source definitions. A command exit or file presence is not a reproduction pass.

**Gate B1 — regenerate forward models (separate, optional before controller work)**
Only after the score-only evaluator pass, decide whether regenerated forward-model predictions are needed. If so, separately pin the container digest, OS packages, source commits, external binaries, manual model weights, conda/pip resolutions, and data assets. Test prediction regeneration against the published predictions before using it to evaluate Symthaea.

**Gate C — Symthaea controller adapter**
Only after Gate B is reproduced: expose candidate selection/resource allocation to Symthaea while keeping experimental observables and evaluation-only metadata inaccessible to the controller. The specialist generator/forward model stays identical across arms. Compare uniform/fixed, conventional heuristic, Symthaea without HDC, and full Symthaea under a matched call/compute budget. Evaluate primary held-out evidence once, after targets, splits, code versions, resource caps, metrics, exclusions, and stopping rules are frozen.

## Run-record contract and executable validator

The fail-closed JSON Schema at [PROTEIN_DYNAMICS_RUN_RECORD_SCHEMA_v1.json](PROTEIN_DYNAMICS_RUN_RECORD_SCHEMA_v1.json) is now enforced by [validate_protein_dynamics_run.py](../../scripts/research/validate_protein_dynamics_run.py). It checks schema conformance, the exact frozen target-manifest digest, unique and expected target identities, finite metrics, observable-role consistency, explicit failure codes, target conformer counts, resource-total reconciliation, artifact path containment, and local output SHA-256 values. Remote/absolute artifact paths are refused rather than incorrectly marked verified.

Sixteen adversarial unit tests are in [test_validate_protein_dynamics_run.py](../../scripts/research/test_validate_protein_dynamics_run.py), with instructions in [the validator guide](PROTEIN_DYNAMICS_RUN_RECORD_VALIDATOR.md). The test suite loads the production schema when executed from a repository checkout. The available sandbox did not contain a repository checkout, so the local 16-test run exercised semantic guards with an isolated minimal-schema fixture; production-schema integration has not yet been executed here. No run record has been created, and the validator is not evidence that a benchmark or biological claim has passed.

## Current decision

**Reuse is promising but not yet qualified.** The implementation provides useful score/refinement machinery, but reference reproducibility, target-level failure accounting, source-observable circularity, and environment/data identity must be resolved before it is used as an evaluator for claims about Symthaea. No protein benchmark was run during this audit, and no scientific-performance claim is authorized by it.

## Inspected source links

- [README at pinned commit](https://github.com/PeptoneLtd/peptonebench/blob/f161b5220a52ebd1a715c91a68beef125101608e/README.md)
- [CLI and target loop](https://github.com/PeptoneLtd/peptonebench/blob/f161b5220a52ebd1a715c91a68beef125101608e/src/peptonebench/scripts/run_benchmark.py)
- [Scoring, ESS, minimization and failure paths](https://github.com/PeptoneLtd/peptonebench/blob/f161b5220a52ebd1a715c91a68beef125101608e/src/peptonebench/reweighting.py)
- [Dependency declarations](https://github.com/PeptoneLtd/peptonebench/blob/f161b5220a52ebd1a715c91a68beef125101608e/pyproject.toml)
- [Dataset download helper](https://github.com/PeptoneLtd/peptonebench/blob/f161b5220a52ebd1a715c91a68beef125101608e/datasets/downloadDBs.sh)
- [Apache-2.0 license](https://github.com/PeptoneLtd/peptonebench/blob/f161b5220a52ebd1a715c91a68beef125101608e/LICENSE)
