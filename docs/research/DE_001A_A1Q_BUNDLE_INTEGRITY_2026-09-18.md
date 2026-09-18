# DE-001A1Q — Fixed-Point Evidence Bundle Integrity

Status: preregistered implementation contract; no scientific result is reported here.

## Purpose

DE-001A1Q closes the receipt-DAG integrity gap between the already-defined DE-001A stages. It does not evaluate cosmology, optimize parameters, compare cosmological models, or promote a scientific claim.

Its only question is:

> Do the exact software subject, content-addressed A0 inputs, A0/A1P/A1R/A1N receipts, and fixed-point manifests form one internally coherent, cryptographically bound execution bundle?

A1Q authority is therefore:

`evidence-bundle-integrity-only`

and every A1Q receipt must declare:

`scientific_claim=NONE`

## Non-circular bundle structure

A1Q does not validate the final qualification capsule that contains the A1Q receipt itself. That would create a circular hash dependency.

Instead the workflow first writes a `DE-001A1Q-PREQUALIFICATION-BUNDLE-v1` containing:

- exact Git HEAD and TREE;
- PR base when present;
- Rust and nixpkgs identities;
- immutable GitHub Action commits;
- Cargo.lock SHA-256;
- SHA-256 of the release A0/A1P/A1R/A1N/A1Q binaries;
- Nix A0 store path, NAR hash, and closure size;
- SHA-256 of A0, A1P, primary A1R, refined A1R, and A1N receipts;
- SHA-256 of the primary and refined A1R manifests;
- workflow run/attempt and A0/A1 chain outcome.

A1Q then verifies that bundle against the live checkout, binaries, Nix store realization, manifests, and receipts. Only after A1Q executes does the workflow write the final qualification capsule, which may safely include the A1Q receipt digest.

## Required graph edges

A1Q must establish all of the following before returning PASS:

1. The live `git rev-parse HEAD` and `HEAD^{tree}` equal the bundle subject.
2. The actual Cargo.lock and all five release binaries reproduce the bundle SHA-256 values.
3. The live Nix store path reproduces the recorded NAR hash and closure size.
4. Every supplied stage receipt and A1R manifest reproduces its bundle SHA-256.
5. A0 is a clean byte-integrity PASS and its best-fit artifact digest is internally consistent.
6. A1P is a PASS, binds the actual A0 receipt hash, binds the actual primary A1R manifest hash, and reports exact fixed-point field matches.
7. Both A1R receipts bind the actual A0 receipt and their respective manifests.
8. Primary and refined A1R reproduction verdicts agree.
9. Workflow chain outcome semantics agree with the reproduction verdict:
   - PASS -> exit 0 / workflow success;
   - NEGATIVE -> exit 1 / workflow failure retained as valid evidence.
10. A1N is a PASS and binds the actual A0 receipt, both manifests, and both A1R receipts.
11. A1N prediction-vector digests are independently recomputed from both A1R receipts and must match.

Any broken edge makes A1Q `INVALID`.

## PASS versus NEGATIVE

A1Q bundle integrity is intentionally orthogonal to the A1R reproduction result.

A coherent bundle may contain either:

- `A1R=PASS`, or
- `A1R=NEGATIVE`.

A valid NEGATIVE result may therefore receive `A1Q=PASS` because its provenance and execution graph are coherent. The workflow's final reproduction qualification still fails when A1R is NEGATIVE.

This distinction prevents an unexpected scientific/numerical result from being misclassified as corrupted evidence, and prevents corrupted evidence from being interpreted as a scientific negative.

## Qualification boundary

Even a complete A1Q PASS means only:

> the fixed-point reproduction evidence bundle is internally coherent and bound to the exact software/input subject.

It does not establish that LambdaCDM is correct, that dark energy is constant, that dark energy evolves, or that any observational anomaly exists.

Those claims remain downstream of the preregistered DE-001 scientific gates.
