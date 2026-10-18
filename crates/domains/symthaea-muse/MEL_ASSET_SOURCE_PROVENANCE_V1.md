# MEL-ASSET-SOURCE-PROVENANCE-001: Asset source, license-document, and attribution evidence

**Status:** Proposed contract; no asset root is deemed legally cleared by its presence or digest alone.
**Scope:** VCSL and VSCO-2-CE sample roots used by `symthaea-muse`.
**Repository:** Public `Luminous-Dynamics/symthaea`.

## Paradigm shift: separate five different claims

A reproducible audio run needs to keep these claims distinct:

1. **Asset-byte integrity:** the WAVs found under a local root match a deterministic content manifest.
2. **Indexed-bank integrity:** the files accepted by the current pitch indexer match the selected-bank content manifest.
3. **Source lineage:** the supplied directory is the root of a clean Git checkout whose origin matches the expected upstream repository; the exact Git revision is recorded.
4. **License/readme evidence:** the exact local `LICENSE` and source README bytes are hashed and kept with the reports.
5. **Terms interpretation:** the published SPDX-style license declaration and any upstream credit/use request are separate fields for human review.

A SHA-256 proves none of the other claims by itself. A clean Git checkout does not prove music quality. A source README is not silently reduced to an SPDX label. A missing license/readme or unavailable upstream revision remains explicit evidence debt instead of becoming an assumed pass.

## Upstream sources and current declaration

The matrix emits these source declarations:

| Source library | Expected upstream | Declared license identifier | Source README |
|---|---|---|---|
| VCSL | [sgossner/VCSL](https://github.com/sgossner/VCSL) | CC0-1.0 | `README.md` |
| VSCO-2-CE | [sgossner/VSCO-2-CE](https://github.com/sgossner/VSCO-2-CE) | CC0-1.0 | `Readme.txt` |

The VCSL repository describes itself as CC0 and says credit is not required. The VSCO-2-CE README permits broad use while requesting credit to Versilian Studios/Sam Gossner and/or Ivy Audio/Simon Dalzell where applicable. That request is emitted as a separate note; the code does not silently translate or discard it. Sources: [VCSL repository description and README](https://github.com/sgossner/VCSL), [VSCO-2-CE README terms](https://github.com/sgossner/VSCO-2-CE/blob/master/Readme.txt).

These are upstream declarations for the currently inspected repositories, not a legal opinion for every derivative, repackaging, or future revision. The emitted `license_document_sha256` and `readme_document_sha256` identify the actual local files inspected, so reviewers can detect terms-document changes. Missing or unreadable files are not treated as compliant.

## Machine-readable matrix fields

The `melothaea.instrument-capability-matrix.v4` report adds `source_terms_evidence` rows containing:

- source library identifier and expected upstream repository URL;
- declared license identifier and a boundary statement explaining that it is an upstream declaration, not a legal determination;
- relative paths, exact SHA-256 hashes, and explicit statuses for `LICENSE` and the source README;
- the upstream Git revision observed from the supplied root;
- whether the checkout origin matches the expected repository, whether the working tree is clean, and a status explaining why source lineage is or is not established;
- a separate upstream README credit/use request.

The source Git revision is distinct from `SYMTHAEA_SOURCE_REV`, which identifies the Melothaea/Symthaea code revision. The tool reports a sample-root revision as `clean_upstream_checkout` only when Git identifies the supplied directory itself as the repository root, its `origin` URL normalizes to the expected upstream, a valid HEAD revision is available, and `git status --porcelain --untracked-files=all` is empty. A subdirectory, archive, dirty checkout, incorrect origin, missing Git installation, or unavailable status remains explicitly non-clean/incomplete.

The tool hashes source term documents directly from the supplied root and omits absolute local paths. It does not fetch or update assets over the network, and it does not modify the source checkout.

## Acceptance benchmarks (targets, not results)

| Gate | Proposed target |
|---|---|
| Document evidence | 100% of qualification-grade source roots have exact local `LICENSE` and README digests, each matching an independent verifier recomputation |
| Source lineage | 100% of qualification-grade roots identify a valid upstream commit, exact expected Git origin, and clean checkout |
| Terms transparency | 100% of roots expose license-document status and upstream README request separately; zero silent fallback to assumed terms |
| Negative controls | Missing or modified license/readme, dirty working tree, wrong origin, non-root subdirectory and missing Git metadata cannot return a full source-provenance pass |
| Audit completeness | Every rendered sample report can be retained with the source revision, root manifest, indexed-bank digest, exact selected-file digest, and source-terms verification findings |

## Limits

A hash of a license file does not itself explain the license, and a Git remote string is not proof of remote ownership. The source revision/origin/clean-tree checks are local evidence, not a signed attestation. Stronger future work should issue and verify an authenticated source/build attestation that binds the sample archive, terms documents, renderer binary and final report to trusted subjects. Human/legal review is still needed for ambiguous or conflicting source materials.
