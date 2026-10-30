# PITB exact artifact-content audit (2026-10-30)

**Status:** scanner committed; the pinned 961-path checkout scan has not yet been run by this workflow. No candidate count or audit result is claimed.

## Purpose

The existing title and artifact-path candidate generators are useful recall-oriented triage tools. They compare metadata and vocabulary, not the bytes of the linked source artifacts. This supplement adds a separate deterministic first-pass audit over the exact linked files in the frozen upstream snapshot.

The scanner reads all 961 unique artifact paths assigned to the 372 frozen families in `splits.json`, after independently checking the registered 70/15/15 partition, the family counts, and the partition digest. It requires the upstream working tree to be exactly commit `1581d4cbdd69a0faf34b2089614fe2a814316a6a` and clean with respect to tracked files.

For every materialized regular file, it records the SHA-256 of the exact bytes and file size. Where byte-identical content occurs at distinct paths owned by different families, it emits an `exact_byte_identity_only` candidate pair with the paths, digest, split labels, and a digest of the candidate evidence.

## What a match means

A byte-identical match is a strong, mechanically reproducible source-provenance lead. It is **not** automatically a confirmed formal dependency, mathematical equivalence, or proof of benchmark contamination. It remains `unreviewed` and must be adjudicated in the signed relationship inventory.

The scan intentionally does not:
- extract text from PDFs or normalize manuscript content;
- infer mathematical equivalence from lexical similarity;
- infer a dependency from shared titles, filenames, or common terminology;
- turn candidate pairs into hard component edges;
- treat the absence of exact-byte matches as evidence that two families are independent.

The full artifact inventory is retained in the report so an external reviewer can reconstruct which paths were scanned and which exact-byte digests were observed. This is a separate evidence channel from the title-token and path-token candidate reports.

## Fail-closed behavior

The scanner rejects a noncanonical or unsafe path, duplicate family/path assignment, a changed split algorithm or digest, an upstream checkout at the wrong commit, or a tracked-dirty source checkout. It refuses to hash symlinks, non-regular files, unreadable files, missing files, or Git-LFS pointer text in place of materialized artifact bytes.

When individual linked artifacts are unavailable, it writes an `incomplete` report listing the affected family/path/status and exits nonzero. An incomplete scan must not be interpreted as a zero-candidate result.

The implementation uses only Python's standard library. The built-in self-test covers exact matches, no-match behavior, path traversal, symlink rejection, and unmaterialized Git-LFS pointers:

```sh
python3 scripts/research/pitb_exact_artifact_content_audit_v0.py --self-test
```

The full scan is wired into `.github/workflows/proof-insight-transfer-v0.yml`. The workflow checks out the exact public upstream commit, runs the scan twice, byte-compares the reports, and retains the report as a workflow artifact. The scan is an evidence-producing tool, not a benchmark qualification gate by itself.

## Next review use

Any emitted candidate should be added to the relationship-review queue with its source paths and digest, then independently assessed by the two full-universe reviewers. A reviewer may confirm a hard shared-source relationship, dismiss it with a reason, or leave it unresolved. The signed receipt must still bind the complete relationship inventory and the externally provisioned reviewer trust root.

**No result is asserted here:** the script's self-test does not access the 961 upstream files. Until an exact-head workflow artifact is available, the full scan, its completeness, and its candidate list remain unverified.
