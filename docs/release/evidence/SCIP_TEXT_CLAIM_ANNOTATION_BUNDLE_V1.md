# SCIP Text-Claim Annotation Bundle V1 — V20 Evidence Note

## Boundary

V18 preregisters the text-to-claim extraction experiment. V19 freezes deterministic corpus partitioning. V20 freezes the human-reference evidence boundary that a case must satisfy before its annotation receipt can enter the V19 manifest.

```text
surface text
  -> two independent source-blind extraction annotations
  -> exact agreement OR distinct source-blind adjudication
  -> frozen surface claim inventory
  -> two independent source-visible alignments
  -> exact agreement OR distinct alignment adjudication
  -> content-addressed annotation bundle
```

The two phases are deliberately separate. Source knowledge may not be used to repair the extraction inventory after it is frozen.

## Frozen semantic policy

Policy semantic SHA-256:

`12a8431fd8a93dd8ba158758bb4089ef951960240fcab11a5178159e67f19bb9`

It binds:

- V18 preregistration `96e2ec5e1fad213f4261405c22d1f80cb6f1d106fd5621481eb0e95a6cca4530`;
- V19 corpus-manifest policy `9a30ba36b3e872aaec349647f71433579bc0932f24c96335bb32afe77498ffc0`.

Human actors are represented only by opaque SHA-256 fingerprints. Names/emails are forbidden in the bundle schema. Fingerprints are provenance handles, not authentication, expertise, independence, or identity proof.

## Extraction phase

Two annotators may see the natural-language surface and public claim schema, but not the grounded source inventory, source claim IDs, or candidate extractor output.

Exact agreement is defined by identical claim-inventory digests. A disagreement requires a third, distinct adjudicator while the source remains hidden. The resolved extraction inventory is frozen before the alignment phase.

## Alignment phase

Two different aligners may then see the frozen surface inventory and source inventory. They may not mutate the frozen surface inventory and may not see candidate extractor output.

V20 strengthens the original policy with an executable representation of the policy's allowed alignment taxonomy. Every aligned surface claim has exactly one of:

```text
exact-source-claim -> exact source-claim digest required
no-source-support  -> source digest must be null
```

Alignment outcomes are unique and canonically ordered by surface-claim digest. The validator derives `aligned_inventory_sha256` from the canonical outcome list itself; a caller cannot choose an unrelated aligned-inventory digest.

If the two aligned inventories disagree, a distinct alignment adjudicator must provide a resolved canonical outcome list. Its resolved digest is also recomputed rather than trusted.

## Role separation

All extraction annotators, extraction adjudicator when present, aligners, and alignment adjudicator when present must have distinct actor fingerprints within a case. Cross-phase role reuse fails closed.

## Receipt

The bundle receipt is:

```text
SHA256(
  "symthaea-scip-text-claim-annotation-bundle-v1\0"
  || canonical_json(bundle)
)
```

V19's `annotation_receipt_sha256` is defined to be this bundle digest. The bundle carries no self-hash field.

## Locally executed final implementation

The final pre-Git implementation/harness pair was executed locally after superseding an earlier larger unpublished prototype. The semantic policy did not change.

```text
python3 -m py_compile scripts/scip_text_claim_annotation_bundle.py \
  scripts/test_scip_text_claim_annotation_bundle.py
PASS

python3 -B scripts/test_scip_text_claim_annotation_bundle.py
PASS_ANNOTATION_BUNDLE_ADVERSARIAL
```

The suite covers, among other cases:

- extraction disagreement with required source-blind adjudication;
- exact extraction agreement without unnecessary adjudication;
- exact alignment agreement;
- alignment disagreement with distinct adjudication;
- missing adjudication rejection in either phase;
- global human-role fingerprint reuse rejection;
- frozen surface/source substitution rejection;
- candidate/source blinding declaration failures;
- PII/shadow-field rejection;
- invalid alignment outcome taxonomy;
- `no-source-support` carrying a forged source digest;
- noncanonical alignment-outcome ordering;
- caller-chosen aligned-inventory digest rejection;
- duplicate JSON-key rejection;
- receipt identity changing when the resolved extraction changes.

Exact final SHA-256 values:

```text
validator
27fe49a5b9794f0c64382f2d97a97aeaf42a806fe0e1891b92a354ccfa90dd24

hostile/positive harness
d162a6971cca754cac8efa131805b46b1564c1bc885620c5927c58b9629e3819

policy bytes
57c80c1b5b72eee745464cda18a3e9dd40b1db11f12bbcf48eb22fb7d1338af1
```

Exact local Git blob identities:

```text
validator aa20559da3cdf3c4b7aeb304460eff7c7b2f9e38
harness   5fa7b0e8ea5aa5b731895ecc5c71d34dead5d93f
policy    e4088cde2b6a629e3a3de2bd39873002814ac7ab
```

Hosted exact-head CI remains independently required.

## Claim boundary

A structurally valid V20 bundle establishes only that supplied evidence conforms to the frozen annotation/adjudication contract and produces a deterministic receipt.

It does **not** establish:

```text
human correctness
independent attestation that declared blinding actually occurred
annotator expertise or independence
extractor quality
surface semantic fidelity
factual truth
confirmatory execution authorization
runtime or action authority
```

A later operational-blinding witness may independently attest the human-process declarations; it must remain a separate evidence type rather than being inferred from this self-declared bundle.
