# Logical content checksums

`StoreContentChecksum` provides a deterministic CRC64-ECMA checksum of the live
logical vector set. Records are encoded in ascending ID order as:

1. a versioned domain separator;
2. the declared live count;
3. each ID in little-endian form;
4. the complete 2,048-byte `BinaryHV` payload.

The checksum intentionally excludes header generations, physical entry indexes,
tombstones, unused capacity, and LSH snapshots. It is therefore useful for:

- proving compaction preserved logical content;
- verifying a v1-to-v2 migration;
- comparing read-only replicas;
- binding portable exports to their decoded record set.

CRC64 is an accidental-corruption detector, not a cryptographic digest. It must
not be used to establish provenance, authenticity, or resistance to malicious
modification. A future authenticated backup format should wrap the archive with
a signature or a cryptographic digest supplied by the wider Symthaea security
layer.
