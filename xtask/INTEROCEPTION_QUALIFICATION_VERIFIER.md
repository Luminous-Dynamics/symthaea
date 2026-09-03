# Native Interoception v0.1 Qualification Verifier

Status: tooling lineage for frozen source `1007949d5c60fd2d7dd650e8bb4521e2b2803c48`.

This verifier is deliberately external to the Native Interoception runtime. It does not alter model semantics, does not create scientific evidence, and does not turn a structurally valid receipt into truth by assertion.

## Trust states

The verifier distinguishes three states:

1. **Structurally qualified** — the schema-v2 `QualificationEvidenceBundle` is internally coherent and all five required receipts say `Passed`.
2. **Evidence verified** — archived local, scientific-capsule, and GitHub Actions objects reproduce the identities named by the bundle.
3. **Promotion authorized** — the verifier has additionally re-run all three local gates on the exact frozen clean checkout and re-resolved both Actions gates from GitHub using their exact run ID and run attempt.

Only state 3 authorizes the v0.2 implementation-start transition. `QualificationEvidenceBundle::is_qualified()`, `InteroceptionInspectBundle`, and archive-only verification are not promotion authority.

## Required checkout separation

Use two separate checkouts:

- **frozen source checkout** at exactly `1007949d5c60fd2d7dd650e8bb4521e2b2803c48`;
- **verifier checkout** on `tooling/interoception-qualification-verifier-v0.1`.

The frozen checkout must remain clean, including untracked files. All evidence directories and the final authorization output must live outside the frozen checkout. Symlinked archive roots or archive objects are rejected.

The verifier policy is source-specific. If the frozen v0.1 source changes, do not reuse this policy as though the new commit were the same qualification candidate.

## Fixed required gates

Local commands are frozen exactly as:

- `cargo fmt --all --check`
- `cargo test -p symthaea-interoception`
- `cargo clippy -p symthaea-interoception --all-targets -- -D warnings`

Actions gates are frozen as:

- `workspace_ci` -> workflow `CI`, `.github/workflows/ci.yml`
- `showroom_integrity` -> workflow `Showroom Integrity`, `.github/workflows/showroom-integrity.yml`

A benchmark workflow is not one of the five required v0.1 qualification gates.

## 1. Capture local gate evidence

Run the verifier binary from the verifier checkout while targeting the separate frozen checkout. Use a new empty evidence directory outside both source-controlled trees.

Conceptually:

```text
cargo xtask interoception-capture-local \
  --repo-root <frozen-checkout> \
  --subject-commit 1007949d5c60fd2d7dd650e8bb4521e2b2803c48 \
  --gate local_fmt \
  --out <external-evidence>/local_fmt
```

Repeat for `local_test` and `local_clippy`.

Each package contains exact transcript bytes plus an environment manifest binding the source/tree, command, lock files, toolchain identity, clean pre/post state, transcript digest, and exit status.

A historical local package is still not sufficient for final promotion: the final authorizer independently re-runs all three commands.

## 2. Capture exact GitHub Actions attempt archives

Never archive a moving `latest attempt` alias. Capture one explicit run ID and run attempt.

For each required Actions gate preserve:

- `run.json` from the exact-attempt workflow-run endpoint;
- `jobs.json` from the exact-attempt jobs endpoint, with every page represented in one complete object;
- `workflow.yml` from the exact frozen source SHA;
- generated `manifest.json`.

The workspace CI evidence identity currently under qualification is run ID `33509970064`; any accepted receipt must name the exact successful attempt that GitHub ultimately exposes. Do not combine arbitrary attempt-1 successes with attempt-2 job records.

The Showroom success is run ID `33509970147`, attempt 1, on the frozen source.

After raw objects exist, build the immutable manifest with `InteroceptionBuildActionsArchive`, then verify it with both archive-only and live verification commands.

For this frozen verifier policy, every job actually present in the evidence-bearing attempt must be terminal `success`. `skipped`, `neutral`, `cancelled`, and `failure` are rejected.

## 3. Materialize the scientific evidence capsule

The schema-v2 bundle contains digests, not file paths for every logical object. The verifier therefore uses an external capsule archive manifest to resolve:

- preregistration;
- resolved experiment configuration;
- input sequence;
- evidence-plane artifact;
- every raw `artifacts[].name` object.

Every path must be a normal relative path under one closed, non-symlink evidence root. The actual bytes must match the digests in the bundle. `Cargo.lock`, optional `flake.lock`, and optional `rust-toolchain.toml` are independently checked against the exact frozen checkout.

The evidence-plane digest must also occur in the raw-artifact digest set.

## 4. Construct the canonical schema-v2 bundle

The bundle must bind exactly the same source/model lineage across:

- qualification receipt;
- evidence capsule;
- outer qualification bundle.

Each local receipt must contain the exact environment/transcript digests produced by its archived package. Each Actions receipt must contain the exact workflow/run/run-attempt identity being archived and verified.

The final authorizer requires the raw bundle file itself to equal its compact canonical JSON representation byte-for-byte. Reformatting or adding ignored JSON fields changes the accepted artifact boundary.

## 5. Final promotion authorization

Run `InteroceptionAuthorizePromotion` with:

- canonical bundle;
- frozen checkout;
- evidence capsule root;
- three local evidence directories;
- workspace-CI Actions archive;
- Showroom Actions archive;
- a new output path outside the frozen checkout.

The authorizer fails closed unless all of the following hold:

- frozen checkout is exactly the policy source and clean;
- raw bundle is canonical and structurally qualified;
- scientific capsule bytes reproduce all declared digests;
- archived local packages bind exactly to the bundle;
- all three fixed local gates independently pass again at authorization time;
- archived Actions packages bind exactly to the bundle;
- GitHub's live exact-attempt run identity equals the archive;
- GitHub reports the exact attempt terminal `success`;
- every job in that evidence-bearing attempt is terminal `success`;
- live exact-attempt job identities equal archived job identities;
- GitHub's exact-SHA workflow Git blob equals the archived workflow bytes;
- frozen checkout remains clean after local reexecution.

Only then may the final typed envelope contain `PromotionAuthorized`.

## Reruns

A rerun creates a different attempt identity even when source SHA and run ID are unchanged. Evidence must name the attempt actually verified.

Do not construct a pass manually from job records spread across attempts. A same-SHA rerun is acceptable only when GitHub exposes the named exact attempt as the terminal successful workflow identity accepted by this policy.

## Archive filesystem boundary

Qualification archives are treated as closed trees:

- archive root may not be a symlink;
- object paths must be relative normal components only;
- `..`, absolute paths, and special components are rejected;
- traversed components and file leaves may not be symlinks;
- leaves must be regular files;
- canonical objects must remain beneath the canonical archive root;
- promotion inputs/outputs must live outside the frozen source checkout.

This is intended to prevent a content-addressed evidence package from quietly depending on mutable bytes elsewhere on the filesystem.

## Claim boundary

Promotion authorization establishes provenance and qualification of the frozen Native Interoception v0.1 substrate. It does **not** establish emotion, affect, sentience, consciousness, or the scientific success of any later v0.2 candidate.

The verifier itself must pass its own format/test/clippy review before this tooling lineage is relied on for promotion. Until then, this document and branch describe an implementation candidate, not qualified verifier software.
