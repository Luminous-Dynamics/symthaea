# NeuroBridge Exact-Head Local Qualification Capsule v0.1

Status: **authored; not qualified until this capsule itself executes successfully**

Profile: `symthaea-neurobridge-local-qualification-v0.1`

## Purpose

GitHub-hosted runner availability must not be the only way to determine whether an exact NeuroBridge source head satisfies its authored mechanism and authority contracts.

This capsule adds a second qualification lineage that can run on a developer workstation or self-hosted Linux machine with Nix while preserving hosted GitHub Actions as an independent lineage.

It does **not** convert a local PASS into hosted CI evidence.

## Invocation

From a working tree whose committed `HEAD` contains the capsule:

```bash
bash scripts/neurobridge/qualify-local.sh
```

An alternate evidence destination may be supplied with:

```bash
bash scripts/neurobridge/qualify-local.sh --out /path/to/evidence
```

The caller working tree may be dirty. Only its committed `HEAD` is selected; uncommitted caller bytes are never qualified.

## Exact-source rule

The outer bootstrap:

1. resolves the caller repository's exact committed `HEAD`;
2. creates a detached worktree at that commit;
3. requires that exact worktree to contain the capsule, contract checker and Nix shell;
4. evaluates the qualification environment from that worktree's locked `flake.lock` and `rust-toolchain.toml`;
5. re-enters the exact-HEAD copy of the script inside that environment.

The inner qualifier records exact Git HEAD/tree commitments and requires the detached worktree to be clean before and after the test suite.

Python bytecode and Cargo build outputs are redirected into a temporary runtime directory rather than written into the source tree.

## Exact-tooling rule

`nix/neurobridge-qualification-shell.nix` is intentionally small.

It supplies:

- Rust from `rust-toolchain.toml` through the repository's locked rust-overlay;
- rustfmt and Clippy from that Rust toolchain;
- the standard native compiler/linker and `pkg-config` from locked nixpkgs for Cargo/build-script needs;
- Python 3.11 from locked nixpkgs;
- Nix, Git and the core evidence/archive utilities from locked nixpkgs.

The checked-in `flake.lock` is made read-only after the exact detached head has been evaluated.

Rust qualification uses `cargo --locked`; a lockfile that requires reconciliation therefore cannot produce a local PASS.

The exact capsule source, shell, contract checker, `flake.lock`, `Cargo.lock`, `rust-toolchain.toml`, and mirrored hosted-workflow files are all committed into retained evidence by SHA-256.

## Independent static checker

`scripts/neurobridge/check_contracts.py` is a stdlib-only closed-world checker separate from the shell orchestration.

It independently inspects the exact committed source/doc/data artifacts for the critical quarantine, atlas, WN56/RVVG, neuromaps, independence, generator, custody and scientific-input snapshot invariants.

This reduces the chance that a local capsule passes merely because it invoked the same production parser that introduced a defect.

## Qualified surface

The v0.1 capsule intentionally qualifies only NeuroBridge mechanism/source contracts that exist in the exact ancestry on which it is executed.

### Substrate evidence authority

It mirrors the focused no-auto-promotion boundary:

- `cargo fmt --check -p symthaea-core`;
- targeted `symthaea-core` substrate-validation tests with `neural_validation`;
- feature-enabled `symthaea-psych-bench` tests;
- Clippy `-D warnings` for the core and consumer surfaces.

The local lane adds `--locked` to Cargo execution.

### Neural benchmark quarantine

It freezes the same core public-authority boundary as the hosted quarantine lane:

- legacy `cortical_similarity` remains crate-private;
- no legacy public re-export exists;
- the qualified external benchmark set remains empty;
- all eight legacy experiments remain accounted for;
- NBQ and fsaverage5/Glasser promotion prerequisites remain present;
- psych-bench format/tests/Clippy remain clean.

### Atlas compiler, cross-check and semantic extractor

It runs the synthetic contracts for:

- deterministic `fsaverage5 -> Glasser360` compilation;
- independent-lineage map comparison;
- full-fsaverage FreeSurfer annotation semantic extraction.

It also independently rechecks the pinned HCP-MMP1 area namespace, Mills/Figshare acquisition metadata and non-authority/status ratchets carried by those profiles.

### Lineage B mechanism and provenance

It runs the synthetic contracts for:

- exact WN56/RVVG source-pair identity;
- neuromaps-method transform orchestration;
- generator implementation provenance;
- safe evidence-bundle custody;
- scientific-input snapshot custody.

It rechecks the pinned neuromaps commit/blob/method/template metadata, independence limitations and no-network-acquisition boundaries, and exercises public CLIs only in help/contract mode.

## Deliberately excluded surfaces

This capsule does not automatically include sibling branches that are not in its exact Git ancestry.

A result must not be interpreted as qualification of separate sibling PRs merely because they belong to the broader NeuroBridge program. Future local capsules may add those surfaces only after they are present in the exact committed ancestry being qualified.

## No real neuroscience-data execution

The local v0.1 capsule does **not** acquire or process real HCP/BALSA/Mills data and does not run a real Connectome Workbench atlas transform.

Its Lineage-B tests use controlled synthetic fixtures/fake Workbench behavior exactly as mechanism tests.

Therefore:

```text
local capsule PASS
    != real Lineage-B atlas evidence
    != FMQ-010
    != human neural alignment
    != consciousness evidence
```

A later real-data qualification capsule requires separately authorized operator inputs and a distinct evidence profile.

## Evidence output

The capsule retains, where available:

- exact Git HEAD and tree;
- exact commit metadata;
- pre/post detached-worktree cleanliness;
- exact `Cargo.lock`, `flake.lock` and `rust-toolchain.toml` digests;
- exact capsule/shell/checker/document digests;
- exact hashes of every hosted focused workflow whose semantics the capsule mirrors;
- rustc/Cargo/rustfmt/Clippy/Python/Nix/kernel identity;
- one log for every qualification phase;
- ordered phase PASS records;
- execution result, exit status and last active phase.

Every retained evidence file is covered by `MANIFEST.sha256`.

The evidence directory is archived using normalized tar ownership/mtime/order and `gzip -n`. The archive is first created as a sibling temporary file, then published without overwriting an existing archive. Its SHA-256 sidecar is likewise published without overwrite and names only the archive basename, not a machine-local absolute path.

No real participant, HCP/BALSA atlas, Workbench-run, medical, or private neuroscience data are intentionally retained by this mechanism-only capsule.

## Result semantics

A completed local `PASS` requires all of the following:

1. every authored source/mechanism phase passes on the exact detached committed head;
2. the detached source remains clean after qualification;
3. evidence manifest generation succeeds;
4. normalized archive generation/publication succeeds;
5. the adjacent archive SHA-256 sidecar is successfully published;
6. the capsule process returns exit code `0`.

`STATUS.env` describes the execution state retained inside the evidence archive. **It is not sufficient by itself to establish completed capsule qualification.**

If manifest, archive or sidecar finalization fails, the capsule returns a dedicated nonzero finalization result. The outer bootstrap also refuses to accept a zero inner result if the completed archive or sidecar is absent.

A normal test failure may still produce a normalized failure archive for diagnosis, but its process result remains nonzero.

## Evidence diversity

The intended model is:

```text
exact NeuroBridge source head
        |
        +--> local locked-Nix capsule ----+
        |                                 |
        +--> hosted focused Actions ------+--> compare exact source/workflow commitments
        |                                 |
        +--> later real-data execution ---+
                                          |
                                          v
                                  scientific qualification
```

Local and hosted lanes intentionally differ in environment and execution provider. Their agreement is more valuable than making them the same system.

If two applicable lineages disagree, the disagreement is itself a failure requiring explanation.

## Non-claims

A local v0.1 PASS does not establish:

- hosted GitHub agreement;
- authorized acquisition or correctness of real HCP/BALSA/Mills bytes;
- Connectome Workbench scientific correctness;
- Lineage-A/Lineage-B independence;
- FMQ-010;
- representational similarity to human neural measurements;
- substrate consciousness;
- Symthaea consciousness;
- protection against a compromised host/kernel/Nix daemon;
- producer authenticity merely because an evidence archive hashes consistently.

An independent hostile-input verifier and, later, an external witness/signature over accepted evidence are separate layers.
