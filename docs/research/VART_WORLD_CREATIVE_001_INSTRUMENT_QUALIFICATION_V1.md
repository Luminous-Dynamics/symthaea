# VART-WORLD-CREATIVE-001 — Instrument Qualification v1

Status: measurement-instrument qualification contract. It does not authorize confirmatory execution or scientific claims.

## Purpose

The VART instrument is the code that orchestrates, records, audits, verifies, and analyzes the experiment. Its identity and qualification are independent from the Symthaea/World Forge subject source being tested.

A confirmatory freeze must not merely name an instrument commit. It must bind the exact instrument files and show that the canonical acceptance path plus every registered adversarial suite passes from a clean checkout.

## Instrument manifest

The qualifier constructs a canonical manifest containing raw SHA-256 digests for the VART scripts, research contracts/schemas, and dedicated workflows used by the campaign. The manifest digest is `instrument_manifest_sha256`.

Changing any listed byte creates a new instrument identity and requires a new qualification receipt. After confirmatory freeze, an instrument change creates a new verifier lineage.

## Qualification suites

v1 executes, at minimum:

- core verifier smoke acceptance;
- frozen N1–N20 evidence-integrity suite;
- prospective execution-context suite;
- world-state equivalence suite;
- explicit world-identity suite;
- independent calibration reconstruction suite;
- anchored pilot design parity/drift suite;
- post-pilot sealed-root audit suite;
- post-pilot disposition suite;
- freeze-preparation eligibility suite.

Every suite must exit 0. The qualifier records stdout/stderr SHA-256 and exit status for each command.

## Source identity

The qualification receipt records:

- instrument repository HEAD/TREE;
- clean-working-tree status;
- `instrument_manifest_sha256`;
- each suite command and output digests;
- Python version;
- qualification timestamp;
- `all_suites_pass = true`.

A separate remote source-closure receipt proves that this instrument HEAD/TREE is fetchable from a durable ref and reproducible from a fresh checkout.

## Subject separation

The instrument checkout is not the subject checkout. Instrument qualification may change the instrument HEAD without changing the subject mechanism under scientific evaluation.

## Claim boundary

Instrument qualification establishes only that the named measurement/verification implementation passes its registered self-tests and falsification suites. It does not establish the scientific VART hypothesis or authorize confirmatory execution.
