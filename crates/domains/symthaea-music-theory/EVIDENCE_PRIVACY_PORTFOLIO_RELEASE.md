# Privacy-Safe Study Assignment and Portfolio Release Contract

This document defines the Series-14 operational boundary for listener assignment,
pseudonymous evidence storage, small-cell publication, and aggregation across
multiple authenticated studies.

The mechanisms in this document reduce accidental disclosure and duplicate
counting. They are not differential privacy, anonymous-credential systems,
ethics approval, or proof that two study-scoped pseudonyms belong to different
human beings.

## Identity boundary

Raw listener tokens exist only at the enrollment and signature-verification
boundary. They may appear inside the private signed response envelope because
the external trust system needs an exact authenticated identity claim. They are
not copied into calibration case records, study-link ledgers, public summaries,
or portfolios.

Persisted study evidence uses:

```text
SHA-256(domain || version || study_id || manifest_sha256 || raw_token)
```

The resulting value is deterministic and study-scoped. It supports duplicate
control within one study and exact migration of earlier authenticated links.
It is pseudonymization, not anonymity. Administrators must issue high-entropy
opaque tokens; guessable identifiers remain vulnerable to offline guessing.

Cohort IDs intentionally do not alter the pseudonym. Moving one listener to a
different cohort in the same study therefore cannot bypass the one-response
rule.

## Assignment registry

`CalibrationStudyAssignmentRegistry` is private administrator material. It
binds one study, manifest, package set, cohort, and complete book catalog.

For each study-scoped pseudonym, the registry records exactly one enrollment:

- `active`
- `completed`
- `revoked`

Completed or revoked identities cannot enroll again. Book selection is
reproducible and balances the least-used books before applying a deterministic
SHA-256 tie-break.

The registry is self-auditing. It rejects:

- Duplicate book indexes, IDs, or SHA-256 values.
- Non-contiguous book catalogs.
- Duplicate enrollment sequences or pseudonyms.
- Unknown or altered assigned books.
- Invalid status/response combinations.
- Unadvanced sequence counters.
- Altered enrollment or registry SHA-256 values.

Assignment assumes administrator-issued tokens. A participant allowed to choose
arbitrary tokens could grind the deterministic tie-break and should not control
the enrollment namespace.

## Transactional response attachment

`attach_assigned_listener_response_to_bundle` clones both the assignment
registry and evidence bundle before changing either value.

The operation verifies:

1. The signed response belongs to the exact study and manifest.
2. The response names the assigned book and its SHA-256.
3. The enrollment is still active.
4. The external signature verifier accepts the response.
5. All revealed judgments attach without rejection.
6. The signed-response SHA-256 is canonical and matches the exact envelope.

Only after all checks pass are both caller-owned values replaced. A wrong book,
duplicate listener, failed signature, unknown calibration case, or completion
mismatch leaves both in-memory values unchanged.

Filesystem tools write replacement JSON files through temporary paths and
rename. Coordinating the bundle and registry across storage systems remains an
operator transaction; the crate does not claim cross-filesystem atomicity.

## Small-cell publication

Private bundles retain exact authenticated links and full paired summaries.
Public projections apply `CalibrationStudyPrivacyPolicy`.

The release policy currently requires at least:

- 5 unique study-scoped assessors.
- 10 included paired judgments.

A suppressed group publishes:

- Its publication status.
- A suppression reason.

It does not publish assessor counts, included-pair counts, pairing matrices,
means, confidence, or Wilson intervals.

The portfolio public report additionally removes exact overall assessor/pair
counts whenever the overall cell is suppressed. Its public decision contains
only acceptance and failed gate codes; private observed/required values and gate
details remain in the private portfolio.

Group labels and the existence of a suppressed cell may still disclose context.
This is not differential privacy and does not provide an anonymity guarantee
against external information.

## Multi-study portfolio boundary

`CalibrationStudyPortfolio` is a private, self-auditing aggregation artifact.
It preserves:

- Exact authenticated response links.
- Source bundle SHA-256 identities.
- Per-study summaries and acceptance decisions.
- A pooled paired summary.
- Study-balance and heterogeneity measurements.
- A separately suppressed public report.

A portfolio refuses to pool bundles unless they share one exact evidence
identity:

- Music-theory engine version.
- Source revision.
- Optional source-tree identity.
- Mutation-corpus version.
- Threshold-policy version.
- Score-evidence version.

This prevents human results collected for one code revision from being used as
release evidence for another.

The release portfolio policy currently requires:

- At least 2 study identities.
- At least 60 included pairs overall.
- At least 15 included pairs per study.
- At least 75% of component studies accepted.
- Pooled separation Wilson lower bound at least 0.60.
- Pooled mean baseline-minus-mutation score at least 0.75.
- No study contributing more than 70% of pooled pairs.
- Between-study separation-rate range no greater than 0.30.
- Every component study large enough for public release.

Study pseudonyms are deliberately unlinkable across different studies. Portfolio
assessor counts are therefore **assessor-study units**, not proven unique humans.
Cross-study participant independence remains a recruitment and governance claim.

## Operator workflow

### Generate books and the private assignment registry

```bash
cargo run --release --example evidence_study_books -- \
  --manifest private/manifest.json \
  --package package-001.json --reveal package-001.reveal.json \
  --cohort-id cohort-a \
  --count 24 \
  --public-dir study-public \
  --private-dir study-private
```

The private directory now includes `assignment-registry.json`.

### Assign a book

Read the opaque token from standard input:

```bash
printf '%s' "$LISTENER_TOKEN" | \
cargo run --release --example evidence_study_assignment -- \
  assign \
  --registry study-private/assignment-registry.json \
  --write study-private/assignment-registry.assigned.json \
  --write-report study-private/enrollment.json
```

A token file may be supplied with `--token-file`; the token is never written to
the registry or report.

### Build a response payload without shell-argument leakage

```bash
printf '%s' "$LISTENER_TOKEN" | \
cargo run --release --example evidence_listener_response_payload -- \
  --book study-public/book-0007.json \
  --responses responses-0007.json \
  --source blinded-cohort-a \
  --write private/response-0007.payload.json \
  --write-message private/response-0007.message.bin
```

External verifier adapters receive a mode-0600 token file through
`--assessor-token-file`, not a raw token argument.

### Authenticate, attach, and complete the assignment

```bash
cargo run --release --example evidence_attach_listener_response -- \
  --bundle evidence-bundle.json \
  --manifest study-private/manifest.json \
  --package-set study-private/package-set.json \
  --book study-public/book-0007.json \
  --reveal study-private/book-0007.reveal.json \
  --envelope private/response-0007.envelope.json \
  --verifier /path/to/verifier \
  --assignment-registry study-private/assignment-registry.assigned.json \
  --write-assignment-registry study-private/assignment-registry.completed.json \
  --write-bundle evidence-bundle.with-study.json \
  --write-report response-0007.link.json
```

### Export a public single-bundle report

```bash
cargo run --release --example evidence_study_public_report -- \
  --bundle evidence-bundle.with-study.json \
  --write public-study-report.json
```

### Build a private portfolio and separate public report

```bash
cargo run --release --example evidence_study_portfolio -- \
  --portfolio-id release-candidate-2026-07 \
  --bundle study-a.bundle.json \
  --bundle study-b.bundle.json \
  --write-private private/study-portfolio.json \
  --write-public public/study-portfolio-report.json \
  --require-accepted
```

Never publish the private portfolio: it includes study-scoped pseudonyms and
exact authenticated response links.

## Trust limits

Series 14 does not prove:

- That a pseudonym maps to a unique human.
- That one human did not participate in two different studies.
- That recruitment was independent or unbiased.
- That suppressed labels cannot be identified using outside knowledge.
- Differential privacy, k-anonymity, or resistance to all linkage attacks.
- Authorization of listener signing keys.
- Ethical approval, informed consent, cultural validity, or perceptual
  universality.

Those responsibilities remain with study governance and the external trust
system.
