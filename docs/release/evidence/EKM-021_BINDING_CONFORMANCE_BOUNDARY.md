# EKM-021 — Canonical Evidence Binding Conformance Boundary

Status: design/evaluation scaffold only. This document does not claim executable qualification.

## Purpose

EKM-020 freezes `EvidenceRecordBinding::v1()` as the compatibility target for evidence records produced by the EKM mutation pipeline. EKM-021 observes the existing mutation writer, replay journal, and independent verifier without changing their runtime behavior.

## Required invariant

For an admitted evidence draft and resulting `EvidenceIngestionOutcome`, the ledger record named by the receipt must match the exact V1 binding over:

- claim ID
- evidence kind
- polarity
- provenance ID
- observation cycle
- canonical result context
- canonical preregistered-decision method

Every mutation-journal entry must independently point to a ledger record satisfying the same V1 binding for its stored draft identity.

## Non-authority boundary

A PASS here would establish only representation compatibility. It would not establish:

- scientific correctness of the evidence
- independence of provenance roots
- calibrated confidence
- causal truth
- permission to change belief strength
- permission to promote causal edges
- permission to affect action selection or external systems

## Qualification rule

Do not call EKM-021 qualified until the exact PR head executes repository CI. Static inspection, authoring, mergeability, or queued jobs are not PASS evidence.

The intended next migration, after qualification, is mechanical replacement of duplicated EKM-017/018/019 ledger-binding reconstruction with `EvidenceRecordBinding::v1()` while preserving the V1 conformance vectors unchanged.
