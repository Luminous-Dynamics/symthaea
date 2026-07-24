# Recovery model

HdcStore separates normal opening from operator-authorized metadata recovery.
This prevents ordinary startup from silently converting corruption into a new
committed state.

## Strict open

`HdcStore::open`:

- validates both header checksums independently;
- selects the highest valid generation;
- accepts one valid header when the other copy is damaged;
- validates every entry inside the committed `vector_count` range;
- rejects duplicate live IDs, invalid statuses, truncation, and count mismatch;
- never writes to the file during open.

A degraded header copy is visible through `header_health`. Call
`repair_header_redundancy` to rewrite the alternate page explicitly.

## Recovering open

`HdcStore::open_recovering` performs the same structural validation, then allows
only two bounded repairs:

1. reconstruct `live_count` and `tombstone_count` from valid entries already
   inside the committed `vector_count` range;
2. write a new generation into an invalid alternate header page.

The returned `RecoveryReport` records the selected slot and generation, header
health before repair, every repair performed, the final generation, and any
contiguous committed-looking entries immediately after `vector_count`.

## What recovery deliberately does not do

Trailing entries are never promoted automatically. Such an entry can be the
result of an append whose entry flush completed but whose header commit did not,
but it can also be stale data or operator damage. Recovery reports the count and
leaves the committed visibility boundary unchanged.

Recovery also refuses to guess when:

- both header pages are invalid;
- equal-generation headers disagree;
- a committed entry has an invalid status;
- live IDs are duplicated;
- the committed region extends beyond the file.

Those cases require offline inspection or restoration from a trusted copy.
