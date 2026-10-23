# VIS-DVPC-012: Content-addressed DVPC evidence receipt v1

**Status:** Draft reference verifier.  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8566  
**Parent contract:** https://github.com/Luminous-Dynamics/symthaea/pull/8492  
**Date:** 2026-10-24

## Why this tranche exists

VIS-DVPC-005 checks strict TSV schemas and that digest-shaped fields link consistently. It explicitly does not recompute hashes over the source-media bytes. VIS-DVPC-012 adds byte-level verification without replacing the existing semantic trace validator and without putting third-party sensor data in the repository.

Run the original VIS-DVPC-005 validator first for its full numeric/schema semantics. Then run this integrity verifier against the *same exact files*. Both are standalone Rust tools; the new verifier repeats only the schema/pairing checks needed to safely bind its receipt to the expected inputs.

The receipt is an identity and integrity record, not an authenticity oracle. A SHA-256 digest proves byte identity relative to a trusted commitment; it does not prove where a file came from, who produced it, or that the source is licensed.

## CLI

Build:

    rustc --edition=2024 -O tools/vision-dvpc-integrity-receipt.rs -o /tmp/vis-dvpc-012

Run the existing semantic validator first:

    tools/vision-dvpc-trace-validator OBSERVATIONS.tsv EVALUATIONS.tsv

Then verify a frozen bundle:

    /tmp/vis-dvpc-012 DATASET_ROOT OBSERVATIONS.tsv EVALUATIONS.tsv \
      SOURCE_MANIFEST.tsv SPLIT_MANIFEST.tsv SNAPSHOT.tsv UTILITY_SPEC.txt \
      tools/vision-dvpc-integrity-receipt.rs BUNDLE_LOCK.tsv

DATASET_ROOT is the root under which manifest asset paths resolve. Each path is relative to that root. The lock file is a separately frozen input; the verifier does not offer a mode that silently generates and trusts a replacement lock.

## Bundle lock

UTF-8 TSV schema:

    DVPC_BUNDLE_LOCK_V1<TAB>key<TAB>sha256
    LOCK<TAB>observations_sha256<TAB><64 hex characters>
    LOCK<TAB>evaluations_sha256<TAB><64 hex characters>
    LOCK<TAB>source_manifest_sha256<TAB><64 hex characters>
    LOCK<TAB>split_manifest_sha256<TAB><64 hex characters>
    LOCK<TAB>snapshot_sha256<TAB><64 hex characters>
    LOCK<TAB>utility_spec_sha256<TAB><64 hex characters>
    LOCK<TAB>verifier_source_sha256<TAB><64 hex characters>

<TAB> means a literal tab byte, not the two characters. The verifier checks each field against the exact bytes read during admission. Unknown, missing or duplicate lock keys fail closed. The lock itself is hashed into the output receipt. A custodian should create and commit/sign the lock inside the sealed evaluator workspace after the immutable paired outcome bundle exists, but before releasing those outcomes to policy developers or analysts. Changing both data and lock is a new experiment, not a valid replay of the original one.

For a transparent local workflow, a separate custody step may generate the lock using an independently reviewed hashing utility. Example for GNU coreutils on Linux (run only after the files and verifier source have been frozen):

    {
      printf 'DVPC_BUNDLE_LOCK_V1\tkey\tsha256\n'
      printf 'LOCK\tobservations_sha256\t%s\n' "$(sha256sum -- OBSERVATIONS.tsv | awk '{print $1}')"
      printf 'LOCK\tevaluations_sha256\t%s\n' "$(sha256sum -- EVALUATIONS.tsv | awk '{print $1}')"
      printf 'LOCK\tsource_manifest_sha256\t%s\n' "$(sha256sum -- SOURCE_MANIFEST.tsv | awk '{print $1}')"
      printf 'LOCK\tsplit_manifest_sha256\t%s\n' "$(sha256sum -- SPLIT_MANIFEST.tsv | awk '{print $1}')"
      printf 'LOCK\tsnapshot_sha256\t%s\n' "$(sha256sum -- SNAPSHOT.tsv | awk '{print $1}')"
      printf 'LOCK\tutility_spec_sha256\t%s\n' "$(sha256sum -- UTILITY_SPEC.txt | awk '{print $1}')"
      printf 'LOCK\tverifier_source_sha256\t%s\n' "$(sha256sum -- tools/vision-dvpc-integrity-receipt.rs | awk '{print $1}')"
    } > BUNDLE_LOCK.tsv

The lock writer is intentionally outside the verifier. Store the lock under separate custody/review or sign its exact bytes; if the lock is freely changed alongside the evaluation bundle, integrity checks only describe the replacement bundle.

The lock binds the verifier source file, not a compiled binary or compiler toolchain. Reproducible-build claims need a separate toolchain/binary digest record.

## Source manifest

Exact header:

    SOURCE_MANIFEST_V1<TAB>role<TAB>path<TAB>byte_len<TAB>sha256

Each row is:

    SOURCE<TAB>role<TAB>relative/path<TAB>exact byte length<TAB>SHA-256 of file bytes

Allowed roles are **sample**, **cheap_output**, **pre_features**, and **shift_signal**. Every sample, cheap-output, pre-feature and non-absent shift-signal digest used by an observation must resolve to a source-manifest record of the corresponding role, and every manifest row is checked against actual bytes and length.

The verifier rejects absolute paths, parent/current-directory components, backslash separators, symlinks in any manifested path component, non-files, missing files, digest/length mismatches, unknown roles and duplicate manifest paths. Files are streamed through SHA-256 so large sensor files need not be loaded completely into memory.

Paths are deliberately local inputs. The tool never downloads data or reads raw media into a receipt. Dataset files and labels remain external to the repository. Run against an immutable/read-only dataset snapshot; this reference tool is not a filesystem sandbox and does not make a concurrent hostile writer or symlink-swap race safe. The bundle lock binds the bytes observed during this run, not an adversarially mutable filesystem.

## Split manifest

Exact header:

    SPLIT_MANIFEST_V1<TAB>group_id<TAB>split

Rows:

    SPLIT<TAB>group-id<TAB>train|validation|test|private_test

The manifest must exactly equal the group-to-split assignments in the observation file. Missing, extra, duplicate or conflicting groups fail closed. The paired trace validator also checks per-group step uniqueness/contiguity. Grouping is a minimum safeguard, not proof that no near-duplicate or cross-dataset leakage exists; the dataset adapter must define a defensible independent unit (for example, source sequence/scene/session) before partitioning.

## Snapshot and utility files

The dataset snapshot descriptor uses this exact schema:

    SNAPSHOT_HEADER<TAB>key<TAB>value
    SNAPSHOT<TAB>dataset_id<TAB>...
    SNAPSHOT<TAB>dataset_revision<TAB>...
    SNAPSHOT<TAB>immutable_ref<TAB>...

The descriptor must bind the same dataset id/revision as trace metadata. The file's exact bytes must match dataset_snapshot_sha256 in the observation/evaluation metadata and snapshot_sha256 in the lock.

The utility file must begin with UTILITY_SPEC_V1; its exact byte digest must match the trace metadata and lock. This verifier binds the declared profile; it does not judge whether the utility specification is appropriate or scientifically valid.

## Receipt and claim boundary

A successful receipt contains hashes for observations, evaluations, source/split manifests, snapshot, utility profile, verifier source and bundle lock, plus row/group/asset counts. It is deterministic for the same exact bytes and file tree.

Always interpret:

- PASS_BYTES_AND_MANIFESTS as successful local byte, schema-linkage, pairing and split-manifest checks only.
- license_scope_declaration as caller-provided metadata, not a legal conclusion.
- legal_rights_verdict=NOT_ASSESSED, statistical_independence_claim=NONE, and performance_claim=NONE literally.

The bundle lock prevents accidental or unreviewed substitution relative to its trusted commitment. It does not itself establish custody, trusted execution, signature identity, hidden-label isolation, statistical independence, valid measurements, performance, downstream safety or SOTA superiority. Keep evaluator outcomes inaccessible to the online router, and keep the held-out evaluator or its signing authority independent of policy tuning.

## Qualification covered by the focused lane

- Independent SHA-256 standard test vectors, including empty input, abc, and a multi-block vector.
- Content-addressed synthetic pair admitted with deterministic receipt replay.
- Exact lock checks for observation/evaluation and verifier-source tampering.
- Manifest byte-length and content mismatch, wrong role, path traversal, duplicate path, symlink, and unresolved shift digest rejection.
- Exact split membership, duplicate split group, and conflicting assignment rejection after correctly updating the lock and metadata.
- Snapshot metadata and paired sample mismatch checks.
- Explicit no-rights/no-performance claim fields in the receipt.

No third-party source media, labels or derived sensor data are committed by this work.
