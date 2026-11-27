# MON-031 — Version-Aware Backup Lineage Contract v1

**Profile:** `mon031-backup-lineage-input-v1`  
**Status:** offline policy evaluator and adversarial unit tests implemented; pgBackRest/Debezium/Kafka runtime adapters pending.

## 1. Research basis

PostgreSQL 18.7 release notes (2026-11-12) say existing incremental backup files should not be trusted because several corruption paths were found, and recommend taking a new full backup after updating and using it as the base for future incremental backups. This is a specific compatibility boundary that a generic `backup success` flag cannot represent.

Debezium 3.7.1.Final release notes list PostgreSQL changes lost on restart when `lsn_events_processed` is 1 or when an offset is stored at a transactional logical message (DBZ-2889), and a decoderbufs BEGIN/stored-LSN replay issue (DBZ-2894). Kafka 4.3.2 fixes KAFKA-20732 and KAFKA-20979 data-loss cases. pgBackRest 2.60.1 fixes multi-repository annotate behavior and SFTP block-incremental restore/reconnect issues. Exact links and lane status are in `REGRESSION-MATRIX-v1.json`.

## 2. Invariant

A physical incremental backup candidate is eligible only when its entire lineage terminates in an independently verified full backup satisfying the configured minimum server version. For an 18.7 recovery policy, no pre-18.7 full base or pre-18.7 incremental node may occur in the candidate's parent chain.

## 3. Inputs and checks

Each backup record binds ID, type, PostgreSQL server version, system identifier, creation time, parent ID and typed integrity/WAL-coverage statuses. The policy binds the candidate, current server version, expected system identifier and optional minimum full-backup version.

The evaluator checks duplicate identities, missing candidates/parents, cycles, unsupported types, parent causality, system identity, major-version compatibility, version regressions and the required post-upgrade full base. Every record in the lineage must have verified integrity. The candidate must have independently verified WAL archive coverage.

## 4. Verdict semantics

- `qualified`: the full parent chain is complete, causal, same-system and version eligible, every integrity status is verified, and the candidate WAL coverage is verified.
- `blocked`: explicit contradiction exists, including duplicate ID, cycle, wrong system identifier, cross-major chain, invalid parent chronology, failed integrity/archive proof or a pre-18.7 base/node under an 18.7 cutoff.
- `indeterminate`: required identity, version, parent, integrity or WAL coverage evidence is missing/unknown.

A `blocked` or `indeterminate` candidate must not be used for a qualified restore claim.

## 5. Scope and limitations

This policy does not parse pgBackRest `info` output, examine backup manifests, execute `check` or `verify`, read WAL, verify storage blocks or restore a database. An adapter must collect provenance and derive integrity/WAL statuses from actual commands and artifacts. This module is a version-aware eligibility gate, not a backup engine or complete recovery verifier.

The incremental-base cutoff is policy input rather than hardcoded global truth. Future upstream advisories can add explicit version transitions without changing the evaluator's underlying chain semantics. Never pool results from different version policies.
