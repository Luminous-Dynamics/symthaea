# LL-004G — unified local/CI real-SPICE evidence runner

Status: **Phase-0 research provenance tooling. Not navigation, site selection, launch authority, rendezvous qualification, or flight evidence.**

LL-004G removes a reproducibility gap in LL-004F: GitHub Actions and local/workstation execution now call the same repository-owned orchestration runner instead of maintaining separate bootstrap/download/generation recipes.

## Entry point

```bash
python3 scripts/run_ll004f_real_evidence.py --self-test
```

A connected run with Python 3.13:

```bash
python3 scripts/run_ll004f_real_evidence.py \
  --repo-root . \
  --work-root target/ll004f-real-run
```

The runner derives the Git SHA from the checkout unless `--git-sha` is supplied explicitly.

For a pre-materialized/offline replay:

```bash
python3 scripts/run_ll004f_real_evidence.py \
  --repo-root . \
  --work-root target/ll004f-real-run \
  --offline
```

`--offline` forbids network retrieval. The exact pinned SpiceyPy wheel and every kernel named by the two checked-in LL-004F configs must already exist under the work root.

## Exact dependency boundary

The runner pins:

- SpiceyPy `8.2.0`;
- CPython 3.13 Linux x86_64 wheel filename;
- wheel source URL;
- wheel SHA-256 `b7abbcc53320d74a1d1cf3e86f1c4bcd4bcc2e733a25c3b32fffbac677606ee1`.

The wheel is verified before installation and is installed only into an isolated venv under the LL-004G work root. The project/runtime Python environment is not modified.

The kernel set is still controlled by:

- `configs/lunar_transport/ll004f_de421_pgda_bridge.json`;
- `configs/lunar_transport/ll004f_de440_reference.json`.

If two configs map the same filename to different source URLs, execution fails before download.

## Execution order

The runner performs one fixed sequence:

1. verify all three existing LL-004F tools with their dependency-free self-tests;
2. verify/download the exact SpiceyPy wheel;
3. create/reuse the isolated runner venv and verify `spiceypy.__version__ == 8.2.0`;
4. collect the exact kernel URLs from both configs;
5. materialize missing kernels, or fail in `--offline` mode;
6. run `verify_ll004f_kernel_locks.py` for DE421 and DE440;
7. generate both SPICE snapshot lineages with the isolated exact SpiceyPy interpreter;
8. derive the DE421→DE440 South-Pole frame-sensitivity receipt;
9. emit an immutable aggregate evidence manifest.

GitHub Actions now delegates this exact sequence to the runner and only adds checkout, Python setup, and artifact upload.

## Evidence manifest

`target/ll004f-real-run/evidence/artifact-manifest.json` binds:

- declared Git SHA;
- Python version;
- SpiceyPy version;
- exact wheel filename/source/SHA-256;
- runner SHA-256;
- both config SHA-256 values;
- all three existing LL-004F tool SHA-256 values;
- SHA-256 of every materialized kernel byte file;
- SHA-256 and byte size of every generated evidence artifact.

An existing differing manifest is not overwritten.

The existing generator also retains its own immutable-output behavior for the snapshot files.

## Work-root layout

```text
target/ll004f-real-run/
├── bootstrap/
│   └── spiceypy-8.2.0-...whl
├── venv/
├── kernels/
│   ├── de421.bsp
│   ├── de440s.bsp
│   ├── moon_pa_de421_1900-2050.bpc
│   ├── moon_pa_de440_200625.bpc
│   ├── moon_080317.tf
│   ├── moon_de440_250416.tf
│   └── naif0012.tls
└── evidence/
    ├── verification/
    ├── snapshots/
    ├── frame-sensitivity/
    └── artifact-manifest.json
```

## Why this matters

Before LL-004G, a local reproduction could accidentally differ from CI in wheel provenance, environment mutation, download handling, or manifest construction even while both used the same lower-level scientific scripts.

LL-004G reduces that surface to one orchestration implementation. A later GitHub runner replay and a local replay can therefore be compared as two executions of the same recipe rather than two independently maintained recipes.

## Evidence boundary

A passing LL-004G run establishes reproducibility/provenance for the declared SPICE evidence lineage. It does **not** establish:

- an operational South-Pole launcher site;
- terrain/corridor safety;
- navigation accuracy;
- terminal guidance performance;
- catcher qualification;
- rendezvous qualification;
- launch/release authority;
- economic superiority of any transport architecture.

Tracks #1542, #1595, #1602, #1612, #1615, and #1711.
