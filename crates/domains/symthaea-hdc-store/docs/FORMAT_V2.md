# HdcStore format version 2

Format version 2 separates metadata from vector data and replaces the single
unprotected header with two independently checksummed header pages.

## Layout

| Region | Offset | Size |
| --- | ---: | ---: |
| Primary header page | 0 | 4096 bytes |
| Secondary header page | 4096 | 4096 bytes |
| Fixed-size entries | 8192 | 2080 bytes each |

Only the first 128 bytes of each header page are currently serialized. The rest
of each page is reserved. Entry metadata remains 32 bytes, followed by the
2048-byte `BinaryHV`, preserving 32-byte alignment for zero-copy reads.

## Header commits

Each header contains a monotonic generation and a CRC64-ECMA checksum. New files
start with the same generation in both slots. Every later metadata commit writes
the next generation to the inactive slot, flushes that complete page, and then
synchronizes file data before the process-local state is published.

Opening validates both slots independently and chooses the valid header with the
highest generation. A single damaged or torn header therefore does not make the
store unreadable. Two valid but different headers with the same generation are
rejected as an ambiguous conflict rather than guessed between.

## Entry commit ordering

Append ordering remains:

1. write the entry metadata and vector;
2. publish the live status byte;
3. flush the complete entry;
4. commit the next header generation;
5. publish process-local indexes.

Delete ordering remains:

1. persist and flush the tombstone status;
2. commit updated counts in the next header generation;
3. update process-local indexes.

The explicit recovery API handles the narrow crash window between an entry-state
flush and its corresponding header commit. See `RECOVERY.md`.

## Compatibility

Format-v1 files are detected and rejected with `VersionMismatch` rather than
being interpreted under the new offsets. Use the explicit migration API added by
the migration patch; do not rewrite the version field in place.
