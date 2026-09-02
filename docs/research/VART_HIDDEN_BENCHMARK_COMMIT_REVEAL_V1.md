# VART Hidden Benchmark Commit–Reveal v1

Status: evaluation infrastructure contract; does not authorize confirmatory execution or scientific claims.

## Goal

A hidden VART benchmark should be both secret before evaluation and auditable afterward. Plain SHA-256 of fixture identifiers or small integer seeds is insufficient because low-entropy values can be guessed by dictionary/brute-force search.

This protocol therefore uses an independent 256-bit nonce for every fixture and seed, plus campaign-domain separation.

## Commitment function

For each hidden item, compute:

`SHA256("SYMTHAEA-VART-HIDDEN-COMMIT-v1\\0" || canonical_json(preimage))`

where the canonical preimage contains:

- `campaign_id`;
- `kind` (`fixture` or `seed`);
- private `nonce_hex`;
- exact private value.

Canonical JSON uses UTF-8, sorted keys, and compact separators.

Each fixture and seed MUST use a different 256-bit nonce.

## Pre-launch private source

The benchmark custodian maintains a private source file outside the development repository using schema:

`symthaea.vart-hidden-benchmark-source.v1`

It contains plaintext hidden fixture identifiers, plaintext hidden seeds, and their nonces.

The private source MUST NOT be committed to the Symthaea development repository or exposed to the subject under evaluation.

## Public commitment manifest

Before prospective freeze/evaluation, the custodian runs:

```bash
python3 scripts/vart_hidden_benchmark_commit_reveal.py commit \
  --source /private/custody/VART_HIDDEN_SOURCE.json \
  --public-out /public/anchors/VART_HIDDEN_PUBLIC_COMMITMENTS.json
```

The public manifest contains only:

- campaign/custodian identity;
- fixture and seed counts;
- lexicographically sorted salted commitments;
- SHA-256 of the exact private source bytes;
- commitment-domain identity;
- false execution/claim authority flags.

It MUST contain no fixture identifiers, plaintext seeds, nonces, expected solutions, target defects, or trap labels.

The raw SHA-256 of the public manifest SHOULD be anchored outside the eventual evidence root before evaluation.

## Post-campaign reveal

After the campaign is sealed and the reveal policy permits unblinding, verify:

```bash
python3 scripts/vart_hidden_benchmark_commit_reveal.py verify-reveal \
  --public /public/anchors/VART_HIDDEN_PUBLIC_COMMITMENTS.json \
  --reveal /private/custody/VART_HIDDEN_SOURCE.json
```

The verifier recomputes every commitment and requires exact equality of:

- campaign identity;
- custodian identity;
- fixture/seed counts;
- complete commitment sets;
- exact private-source SHA-256;
- commitment domain.

A changed fixture, seed, nonce, campaign, or source byte sequence fails verification.

## Relationship to the DEVART/VART firewall

The public fixture/seed commitment arrays are the values that should populate the VART side of `VART_BENCHMARK_FIREWALL_V1`.

The firewall remains responsible for preventing:

- overlap with DEVART commitments;
- reuse of spent hidden commitments;
- repository leakage;
- domain confusion.

Commit–reveal adds a different guarantee: the later-revealed hidden set is the same set prospectively committed before evaluation.

## Nonce rules

- 256 bits per item;
- cryptographically random;
- never reused across fixture/seed entries;
- never reused across campaigns;
- private until reveal;
- not derived deterministically from the fixture ID or seed.

The included source template contains placeholders only. Actual hidden source material belongs outside the repository.

## Claim boundary

A successful reveal proves benchmark custody consistency. It does not prove benchmark quality, independence, scientific validity, Symthaea performance, or any efficacy claim.
