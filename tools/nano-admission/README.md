# nano-admission-verifier

A small Rust CLI for the first local artifact-integrity stage of the OC22 IS2RE-Total admission process.

This utility **does not download, extract, or parse dataset records**. It hashes a local archive, checks the upstream-published MD5, computes SHA-256, inventories tar.gz members while streaming each payload to a sink, reports duplicate/unsafe paths and special entries, and writes a JSON receipt.

## Run

From the Symthaea repository root:

    cargo run --locked -p nano-admission-verifier -- \
      --archive /path/to/is2res_total_train_val_test_lmdbs.tar.gz \
      --output /tmp/oc22-artifact-integrity.json

Exit codes:
- 0: compressed artifact identity matches the upstream MD5 and archive inventory is structurally ordinary; status is still artifact_integrity_verified_pending_schema.
- 2: upstream MD5 mismatch.
- 3: duplicate path, unsafe path, or special file entry requires review.
- 4: archive could not be parsed/fully streamed as tar.gz, or its decompressed-byte/member-count budget was exceeded.
- 5: the measured file contents changed between the initial hash and the post-inventory hash.
- 1: argument, input, or output error prevented receipt creation.

The JSON receipt embeds the pinned source-document commit, advertised artifact URL, the upstream-reported checksum/license, locally measured byte size and hashes, and inventory diagnostics. Do not treat upstream-reported license/checksum information as local artifact proof.

## Safety and claim limits

- Members are never extracted or executed.
- Archive path traversal, absolute paths, duplicate member names, and non-regular/non-directory entries are recorded as review blockers.
- The compressed stream is hashed separately from the declared tar payload bytes. Member bodies are read fully so decompression/truncation errors are not hidden by merely listing headers.
- Gzip decompression and tar member count are bounded (2 GiB expanded stream, 250,000 members) to limit resource exhaustion from malformed archives.
- The compressed artifact is hashed before and after inventory; changed content blocks a clean status. This is a mutation check, not an OS-level immutable-file guarantee.
- A matching checksum does not verify LMDB schema, train/validation/test counts, units, ID/OOD mapping, structural chemistry, or licensing for a separate model checkpoint.
- The receipt cannot state an admitted result. Its best status is artifact_integrity_verified_pending_schema; NANO-001A and NANO-001C remain independent gates.
- The archive is expected to remain in local temporary research storage and must not be committed or mirrored from this utility.

Tests use tiny synthetic tar.gz fixtures and known MD5/SHA-256 vectors; they do not reproduce the real OC22 artifact or validate its schema.
