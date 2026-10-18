# MEL-ASSET-SOURCE-PROVENANCE-001: Asset source, license-document, and attribution evidence

**Status:** Proposed contract; no asset root is deemed legally cleared by its presence or digest alone.
**Scope:** VCSL and VSCO-2-CE sample roots used by `symthaea-muse`.
**Repository:** Public `Luminous-Dynamics/symthaea`.

## Paradigm shift: separate five different claims

A reproducible audio run needs to keep these claims distinct:

1. **Asset-byte integrity:** the WAVs found under a local root match a deterministic content manifest.
2. **Indexed-bank integrity:** the files accepted by the current pitch indexer match the selected-bank content manifest.
3. **Source checkout identity:** the supplied directory is the root of a Git checkout whose origin URL matches the expected repository; its exact local HEAD, worktree state, and license/readme hashes are recorded. This is not proof that HEAD has been published to or signed by the upstream maintainers.
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
- the local source-checkout HEAD observed from the supplied root (field `source_checkout_revision`);
- whether the origin URL matches the expected repository, whether the working tree is clean, and a status explaining the evidence state (`source_checkout_status`);
- a separate upstream README credit/use request.

The sample-root's `source_checkout_revision` is distinct from `SYMTHAEA_SOURCE_REV`, which identifies the Melothaea/Symthaea code revision. The tool reports `source_checkout_status=clean_expected_origin_checkout` only when Git identifies the supplied directory itself as the repository root, its `origin` URL normalizes to the expected repository, a valid local HEAD revision is available, and `git status --porcelain --untracked-files=all` is empty. A subdirectory, archive, dirty checkout, incorrect origin, missing Git installation, or unavailable status remains explicitly non-clean/incomplete.

The tool hashes source term documents directly from the supplied root and omits absolute local paths. It does not fetch or update assets over the network, and it does not modify the source checkout.

## Acceptance benchmarks (targets, not results)

| Gate | Proposed target |
|---|---|
| Document evidence | 100% of qualification-grade source roots have exact local `LICENSE` and README digests, each matching an independent verifier recomputation |
| Source checkout identity | 100% of qualification-grade roots record a valid local HEAD, exact expected Git origin, and clean checkout |
| Terms transparency | 100% of roots expose license-document status and upstream README request separately; zero silent fallback to assumed terms |
| Negative controls | Missing or modified license/readme, dirty working tree, wrong origin, non-root subdirectory and missing Git metadata cannot return a full source-provenance pass |
| Audit completeness | Every rendered sample report can be retained with the source revision, root manifest, indexed-bank digest, exact selected-file digest, and source-terms verification findings |

## Limits

A hash of a license file does not itself explain the license, and a Git remote string is not proof of remote ownership. The reported HEAD is the local checkout revision; the verifier does not prove that this commit exists on the official remote or was signed by its maintainers. The revision/origin/clean-tree checks are local evidence, not a signed attestation. Stronger future work should issue and verify an authenticated source/build attestation that binds the sample archive, terms documents, renderer binary and final report to trusted subjects. Human/legal review is still needed for ambiguous or conflicting source materials.
