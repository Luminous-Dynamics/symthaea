# Fe-Co-Zr retrospective experiment

This crate is the local orchestration layer for the OQMD-v1.7 / Fe-Co-Zr retrospective materials experiment.

Its purpose is not to make the pipeline convenient at the expense of evidence. Each expensive stage is explicit, produces an immutable create-new artifact, and requires the previous stage's typed evidence.

A successful run establishes a controlled retrospective benchmark. It does **not** by itself establish prospective discovery, novelty, source authenticity, experimental material truth, or independent replication.

## Build before running expensive data

Do these in order:

1. Build the Rust workspace and the two SIM-PROC launchers from the exact experiment head.
2. Run the pure contract fixture.
3. Enter the pinned runtime-tools shell and validate curl/gzip/MySQL-compatible tooling.
4. Validate a tiny local import/inventory fixture before using the OQMD archive.
5. Qualify an exact qmpy 1.4.0 runtime from reviewed source commit `dede5bdf4aa3ea1187a7bc273e86336c24aadb25`.
6. Capture the official OQMD download page as immutable source-listing evidence.
7. Fill a new config and environment manifest.
8. Run `prepare` and inspect `readiness.json`.
9. Only then acquire the real archive.

The public OQMD archive expected by the frozen protocol is `qmdb__v1_7__052025.sql.gz`. Do not substitute another release under the same run directory.

## Rust-side fixture

From the repository's normal pinned Rust environment:

```sh
cargo test -p symthaea-materials-fe-co-zr-experiment --test static_contract_fixture
```

This fixture does not touch MySQL or qmpy. It verifies the pure chain from historical snapshot identity through import/schema/extraction receipts, authorized target projection, contamination audit, and authorized-audit convergence.

The fixture intentionally models a target where the exact historical structure exists but the scored `K1` label does not. A passing result therefore exercises a label-holdout case rather than a fake novelty claim.

## Pinned runtime tools

A small nested flake pins the same nixpkgs revision currently used by the repository:

```sh
nix develop ./crates/domains/symthaea-materials-fe-co-zr-experiment/runtime
```

It supplies curl, CA certificates, gzip, jq, core shell tools, and Percona Server 8.4.

Percona is supplied for the **local integration fixture** because the pinned nixpkgs exposes it as its MySQL-compatible server. Its presence does not preselect the database engine for the scientific run. The real run records and verifies the exact selected server/client binaries and their exact version strings.

## qmpy compatibility boundary

qmpy 1.4.0 advertises Python 3.7 and `Django < 2.3`. The current repository nixpkgs is much newer. Do not silently install qmpy into whatever Python happens to be present and call that reproducible.

The extraction adapter is pinned to:

```text
qmpy version: 1.4.0
source commit: dede5bdf4aa3ea1187a7bc273e86336c24aadb25
```

The qualified qmpy runtime manifest should record at minimum:

- exact Python executable path and SHA-256;
- exact qmpy source/package closure;
- exact qmpy source commit;
- dependency-lock/environment identity;
- proof that `from qmpy.materials.formation_energy import FormationEnergy` succeeds;
- no network dependency during extraction.

The adapter sets `qmdb_v1_1_pswd` to the explicit empty string before importing qmpy because qmpy 1.4's settings module indexes that variable directly. The experiment uses a passwordless account accessible only through the isolated Unix socket; no secret is added to evidence artifacts.

If an exact legacy Python closure cannot be made trustworthy, stop at the historical database state and introduce a separately reviewed direct-SQL compatibility adapter. Do not weaken the qmpy identity requirement ad hoc.

## Run layout

Use a fresh directory for every scientific lineage. A suggested layout is:

```text
run/
  source/
    oqmd-download-page.html
  manifests/
    experiment-environment.json
    qmpy-1.4-runtime.json
    empty
  data/
    qmdb__v1_7__052025.sql.gz
    qmdb__v1_7__052025.sql
    fe-co-zr-raw.ndjson
  mysql/
    data/
    mysql.sock
    mysql.pid
  evidence/
    prepared.json
    readiness.json
    acquisition-attempt.json
    acquisition-receipt.json
    import-plan.json
    decompression.json
    server-preflight.json
    mysql-import.json
    database-state.json
    extraction-plan.json
    raw-extraction.json
    normalization.json
    completed-extraction.json
    authorized-targets.json
    historical-run.json
    authorized-audit-binding.json
```

Evidence writers use create-new semantics. If a scientific input changes, create a new lineage rather than overwriting old evidence.

## Isolated MySQL-compatible server

The database server is a separately supervised prerequisite. This is intentional: the scientific executor verifies the selected server binary, PID, socket, and live configuration, but does not hide a background daemon behind an untracked shell process.

For the local Percona fixture, initialize a fresh data directory with the exact binary from the runtime shell. A representative profile is:

```sh
mysqld --no-defaults --initialize-insecure --datadir="$RUN/mysql/data"

mysqld --no-defaults \
  --datadir="$RUN/mysql/data" \
  --socket="$RUN/mysql/mysql.sock" \
  --pid-file="$RUN/mysql/mysql.pid" \
  --skip-networking \
  --character-set-server=utf8mb4 \
  --collation-server=utf8mb4_bin \
  --sql-mode=STRICT_TRANS_TABLES \
  --default-time-zone=+00:00 \
  --max-allowed-packet=67108864 \
  --innodb-strict-mode=ON \
  --lower-case-table-names=0 \
  --daemonize
```

Then create the isolated database and passwordless socket-only local account. The exact SQL and server launch profile must be captured in `experiment-environment.json` before `prepare`.

Do not copy these version strings into the configuration from documentation. Capture the literal outputs of:

```sh
mysqld --version
mysql --version
```

The preflight requires exact equality and independently hashes `/proc/<pid>/exe`.

## Configuration

Copy `config.example.json` to a new run-specific file and replace every placeholder with an absolute path or exact observed value.

`disclosure_manifest_path` and `target_requests_path` may remain `null` while constructing the historical corpus. In that state the driver may report historical execution prerequisites as available, but scoring targets are explicitly not ready.

This is expected while only the public abstract is available: publication facts do not automatically grant exact structure or quantitative-property scoring authority.

## CLI state machine

Build the driver from the exact branch/head, then invoke it with the same config for every stage.

### 1. Prepare

```sh
symthaea-fe-co-zr-experiment prepare /abs/run/config.json
symthaea-fe-co-zr-experiment ready /abs/run/config.json
```

`prepare` observes local tool bytes, constructs the acquisition profile, derives the exact MySQL import-command digest, validates the import profile, and freezes `prepared.json`.

No download/import claim is made.

### 2. Acquire

```sh
symthaea-fe-co-zr-experiment acquire /abs/run/config.json
```

This always preserves `acquisition-attempt.json`, including failed or partial transfers.

Receipt promotion is separate:

```sh
symthaea-fe-co-zr-experiment bind-acquisition /abs/run/config.json 2026-09-20T00:00:00Z
```

Use the actual UTC acquisition timestamp. Do not reuse the example timestamp.

A successful acquisition receipt establishes exact observed archive bytes over the recorded route. It is not an official provider checksum/signature.

### 3. Bind the live server

Once the isolated server is running and its PID file exists:

```sh
symthaea-fe-co-zr-experiment prepare-import /abs/run/config.json
```

The final import plan binds that concrete PID, exact server/client/decompressor binaries, exact socket/database/user, import command semantics, environment manifest, acquisition receipt and protocol.

### 4. Decompress, preflight and import

```sh
symthaea-fe-co-zr-experiment decompress /abs/run/config.json
symthaea-fe-co-zr-experiment preflight-db /abs/run/config.json
symthaea-fe-co-zr-experiment import /abs/run/config.json
```

`mysql` exit zero remains only process evidence.

### 5. Materialize database state

```sh
symthaea-fe-co-zr-experiment inventory /abs/run/config.json
```

This inventories the **whole** imported database: base tables, engines, columns, indexes, foreign keys, exact `SHOW CREATE TABLE` identities and exact row counts. Only here may the historical import receipt be minted.

### 6. Extract the Fe-Co-Zr corpus

```sh
symthaea-fe-co-zr-experiment prepare-extraction /abs/run/config.json
symthaea-fe-co-zr-experiment extract /abs/run/config.json
```

The qmpy adapter uses source-native `fit="standard"` semantics and no scientific-value prefilter.

If more than one standard-fit FormationEnergy row exists for one OQMD entry, `normalization.json` is preserved and extraction stops. That is evidence that the v1 one-row-per-entry protocol needs an explicit reviewed selector; the tool never chooses one implicitly.

### 7. Build source-authorized targets

Once sufficient publication/supplement/dataset source bytes exist:

```sh
symthaea-fe-co-zr-experiment targets /abs/run/config.json
```

An abstract-level composition mention cannot become a structure/property answer key. The existing disclosure authority decides which requests are eligible.

### 8. Audit and verify

```sh
symthaea-fe-co-zr-experiment audit /abs/run/config.json
symthaea-fe-co-zr-experiment verify /abs/run/config.json
```

`audit` re-reads the exact archive bytes and reconstructs the historical contamination audit before binding those audited labels to the publication-authorized target set.

`verify` recomputes the pure historical run and convergence binding and requires exact equality with stored artifacts.

## What is deliberately not automated yet

The following remain explicit boundaries rather than hidden convenience behavior:

- starting/stopping the long-lived database daemon;
- choosing/qualifying the legacy qmpy Python runtime;
- obtaining publication supplement/source bytes that are not publicly accessible;
- the blinded Symthaea scientific search/submission stage itself;
- later high-fidelity DFT evaluation and prospective target generation.

Those should be added as separately reviewable authority layers, not shell shortcuts.

## Stop conditions

Stop the run and preserve evidence if any of these occur:

- archive transfer is partial or final URL/status is rejected;
- local executable or manifest digest changes;
- server PID/executable/configuration does not match the plan;
- import diagnostics are truncated;
- whole-database inventory cannot be canonicalized;
- qmpy emits unexpected elements/non-standard fit/source multiplicity;
- corpus/receipt/archive identities disagree;
- scoring source lacks structure/property disclosure authority;
- stored audit fails exact replay.

A null or failed result is valid scientific evidence. Do not repair a frozen run in place.
