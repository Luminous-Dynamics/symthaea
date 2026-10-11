# PHI-SEM-001A-R2 — Git-object-backed Phi authority-flow inventory

Parent program: #5401 / #5402
Failed R1 product: #5405
Failed exact admission: #5408

Status: measurement-only repair. No Phi estimator, threshold, controller, expression,
action, governance, telemetry, or runtime behavior changes.

## Demonstrated R1 failure

The exact #5408 qualifier run `35644463782` preserved artifact `10693782298`
with GitHub artifact digest:

```text
sha256:5a0e70c05370b5293519383b676589fae6d4644ad5499016c18064fd01fe5ba8
```

The receipt proves that:

- the qualifier subject was exact;
- the product commit/tree were exact;
- the frozen R1 audit blob was exact;
- both qualifier and product tracked worktrees were clean;
- Python compilation succeeded;
- the audit help surface executed successfully.

The first inventory execution then failed with:

```text
FileNotFoundError:
papers/evaluation/psych-bench/ablation_domains.csv
```

That tracked path is a Git symlink:

```text
mode 120000
blob a5164501d87561eb53e6064da94ddbe262a056af
```

and its Git blob contains only:

```text
../data/psych_bench/ablation_domains.csv
```

The target is not present at the path produced by filesystem-relative
resolution on the frozen subject. The same directory contains multiple tracked
CSV/BST symlink entries.

Therefore R1 demonstrated a measurement-harness defect:

```text
tracked index entry
!= necessarily materialized regular filesystem file
```

and:

```text
Git symlink blob bytes
!= referenced target file bytes
```

This is not evidence for or against any Phi semantic proposition.

## R2 repair

R2 keeps the R1 lexical engine byte-for-byte:

```text
scripts/audit_phi_authority_flow.py
git blob 5fbbd183e3cc69ba2080633a6b613b95772739e9
```

and executes it through:

```text
scripts/audit_phi_authority_flow_r2.py
```

The v2 shim changes only evidence-byte acquisition and mode binding.

### Regular tracked files

For Git modes:

```text
100644
100755
```

the exact index blob is read with `git cat-file blob <blob-id>`.

The census no longer depends on filesystem materialization for source bytes.

### Tracked symlinks

For Git mode:

```text
120000
```

R2 hashes and binds the symlink pointer blob itself, but exposes no lexical text
to the Phi matcher.

This avoids both unsafe interpretations:

```text
symlink pointer text
== source text
```

and:

```text
dereferenced target bytes
== bytes committed at the symlink path
```

Neither equation is generally true.

### Matched-file identity

Every matched inventory item gains its exact Git mode in addition to the
existing path/blob/SHA-256/byte identity.

### Phi Oracle duplicate fingerprint

The two Phi Oracle source trees are fingerprinted from tracked Git objects using:

```text
relative path
|| git mode
|| git blob id
|| SHA-256(blob bytes)
```

rather than reading working-tree paths.

### Audit-artifact identity

The frozen R1 engine's own byte identity is also computed from its tracked Git
blob and records the byte source as:

```text
git-index-object-bytes-v2
```

## Profile identity

R2 is intentionally a new profile:

```text
phi-sem-001a-lexical-v2
report_version = 2
```

The v2 profile digest domain-separates the unchanged R1 lexical category and
mandatory-witness configuration from these new I/O semantics:

```text
io_semantics = git-index-object-bytes-v2
regular_modes = 100644,100755
symlink_mode = 120000
symlink_lexical_policy = no-scan-pointer-bytes-only
subtree_fingerprint = relative-path+git-mode+blob+sha256-v2
matched_inventory_addition = git_mode
```

The R2 wrapper and this research contract are excluded from the semantic census
to avoid self-inflation.

## Qualification contract

R2 itself does not claim `PASS_INVENTORY`.

A fresh exact-subject qualifier must:

1. prove exact clean-root -> R2 ancestry;
2. bind the unchanged R1 engine blob;
3. bind the R2 wrapper blob and profile;
4. execute the v2 product twice against a detached exact product worktree;
5. require byte-identical reports;
6. require every mandatory witness selector;
7. verify the report source commit/tree;
8. verify the engine and wrapper byte identities independently;
9. verify Git-mode binding on every matched file;
10. verify the Phi Oracle duplicate fingerprints use the v2 semantics;
11. preserve the complete reports, stdout/stderr, and machine-readable receipt.

Only such an exact successful execution may establish:

```text
PASS_INVENTORY
```

for this R2 product.

## Claim ceiling

Even a qualified R2 inventory would establish only:

```text
deterministic tracked lexical Phi/integration census
under phi-sem-001a-lexical-v2
for one exact Git subject
```

It would **not** establish:

- runtime reachability completeness;
- estimator correctness;
- correctness of any Phi implementation;
- IIT validity;
- consciousness;
- calibrated epistemic confidence;
- scientific validity;
- causal validity;
- safety;
- source authority;
- expression authority;
- action readiness;
- governance legitimacy;
- execution authority.

## Architectural lesson

Qualification should prefer immutable repository objects over incidental checkout
state when the claim is about committed source evidence:

```text
Git object identity
-> deterministic measurement substrate

working-tree dereference
-> environment-sensitive convenience
```

Filesystem behavior can still be measured where it is itself the subject, but
it should not silently determine a source-census theorem.
