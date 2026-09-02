# VART-WORLD-CREATIVE-001 — Instrument Source Closure v1

Status: measurement-instrument reproducibility contract. It does not authorize confirmatory execution or scientific claims.

## Purpose

Instrument qualification proves that a particular checkout passes the registered VART acceptance and falsification suites. Instrument source closure proves that the **same qualified instrument HEAD/TREE** is durably retrievable from a named remote ref and reconstructable in a fresh checkout.

The two receipts are distinct and mutually bound.

## Required evidence

An instrument source-closure receipt contains:

- `schema = "symthaea.vart-world-creative-001.instrument-source-closure.v1"`;
- `experiment_id = "VART-WORLD-CREATIVE-001"`;
- `status = "qualified"`;
- `instrument_source.head` / `instrument_source.tree`;
- durable repository/ref identity;
- exact `git ls-remote` match to the instrument HEAD;
- fresh detached checkout with the same HEAD/TREE and a clean worktree;
- raw `instrument_qualification_receipt_sha256`;
- `instrument_manifest_sha256` copied from and checked against the qualification receipt;
- `instrument_environment_digest` copied from and checked against the qualification receipt;
- authority bits fixed false.

## Qualification-receipt binding

The referenced instrument qualification receipt must itself say:

- `status = "qualified"`;
- `all_suites_pass = true`;
- `instrument_source.head/tree` equal the source-closure HEAD/TREE;
- `confirmatory_execution_authorized = false`;
- `claim_authorized = false`.

The source-closure qualifier independently hashes that receipt rather than trusting a filename or label.

## Fresh-checkout boundary

The remote ref must resolve uniquely to the qualified instrument HEAD. A fresh temporary repository fetches that exact ref, independently reconstructs the TREE, checks out the commit detached, and remains clean.

A local commit that has not been pushed to the durable ref cannot satisfy closure.

## Confirmatory freeze

The freeze binds both the instrument qualification receipt hash and the instrument source-closure receipt hash. Any post-freeze instrument byte change creates a new instrument/verifier lineage.

## Claim boundary

This receipt proves retrievability and identity of the measurement instrument only. It does not establish VART efficacy or authorize confirmatory execution.
