# LQCD portable ↔ hosted receipt comparator

Issue: #3491  
Authority: independently executed synthetic comparator evidence  
Rust execution authority: none  
Scientific authority: none

## Purpose

Freeze a fail-closed comparison boundary between #3491 portable execution
receipts and the existing `symthaea.focused-positive-receipt.v2` emitted by the
LQCD exact-head fast lane.

The comparator deliberately does **not** treat two PASS-like receipts as proof
of full cross-provider equivalence. It verifies the portable receipt's own
semantic/receipt digests, requires an externally supplied SHA-256 for the
hosted positive-receipt artifact, validates the hosted exact-subject/verifier
bindings and required gates, and then compares only fields represented with
compatible semantics in both schemas.

## Frozen subjects

- comparator SHA-256: `cae395e449405007640049b4fc98bf5d2a2ab3c846e8a8931c056068f7fef403`
- independent synthetic harness SHA-256: `f16bb757c3464575b9225035cb4eb9cce50bd9b170352b6234362bdab8883e21`
- canonical stdout SHA-256: `2874029dbe5599f9d0c6cc105922cf94238dec7aa518f280fcdab7770e222558`

## Positive theorem

A synthetic portable receipt and hosted-v2 positive receipt with matching:

- exact subject commit/tree;
- exact base commit;
- exact verifier commit/tree;
- qualification profile;
- recipe-semantics digest;
- rustc identity;
- Cargo identity;
- Clippy identity;

classify as:

`CoreEquivalentExtendedBindingUnavailable`

with `core_equivalent=true` and `cross_provider_corroborated=false`.

## Why full corroboration remains false

Hosted positive-receipt v2 does not provide compatible normalized bindings for:

1. exact base tree SHA;
2. normalized target triple;
3. numerical profile ID;
4. environment-equivalence ID;
5. a subject-input digest using the same canonical representation as portable
   receipt v1.

Therefore the comparator is mechanically unable to emit
`CrossProviderCorroborated` from the current schema pair, even when all shared
fields match.

## Negative controls

Executed controls establish:

- wrong externally claimed hosted artifact SHA-256 ->
  `HostedArtifactDigestMismatch`;
- hosted required `clippy` gate changed to FAIL ->
  `HostedRequiredGateNotPass:clippy`;
- portable authority self-promoted -> `PortableAuthorityInvalid`;
- subject mutation -> `SubjectShift`;
- verifier mutation with internally coherent hosted expected binding ->
  `VerifierShift`;
- recipe digest mutation -> `RecipeShift`;
- rustc identity mutation -> `RustcShift`.

## Negative claim theorem

This tranche uses only synthetic receipts. It explicitly records:

- `real_hosted_receipt_compared=false`
- `real_rust_execution_performed=false`
- `full_cross_provider_corroboration_authorized=false`
- `real_beta6_campaign_authorized=false`

It does not establish #3434 attribution, #3436 qualification, `RustQualified`,
EHK agreement, or any lattice-QCD numerical/physics result.

## Next step

Strengthen the hosted positive-receipt schema in #3499 so the five missing
normalized bindings are explicit. Until that schema is itself qualified,
hosted v2 may corroborate the shared authority spine but cannot promote a
portable receipt to full cross-provider corroboration.
