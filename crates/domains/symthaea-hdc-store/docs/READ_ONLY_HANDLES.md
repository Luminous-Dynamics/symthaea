# Shared read-only handles

`HdcStoreReader` opens a format-v2 store without write permissions and holds
shared advisory locks on both the path-stable coordination inode and the current
data inode.

This provides a process-level read boundary with the following properties:

- multiple readers may coexist;
- mutable open, recovery, migration, and compaction are excluded while any
  reader is alive;
- a pending batch journal is rejected rather than interpreted by a reader;
- both header pages and the committed entry range are validated before data is
  exposed;
- IDs and exact-search ties are deterministic;
- `BinaryHV` values are returned as zero-copy references into a read-only mmap.

The reader is pinned to the data inode and header generation selected during
`open`. It does not observe later writes. Drop it and reopen to advance to a
newer generation.

## Index construction

`HdcStoreReader::open` is exact-read only and does not construct or load an ANN
index. This keeps lookup, checksum, inspection-adjacent workflows, and portable
exports proportional to the committed entry scan rather than to the number of
LSH hyperplanes. Use `open_with_index_policy` only when approximate search is
actually required.
