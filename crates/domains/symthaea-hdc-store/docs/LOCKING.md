# Path-Stable Store Locking

Mutable access is coordinated through a persistent sibling file named
`<store>.lock`. The coordination inode is opened and advisory-locked before the
canonical data file is created, opened, inspected, migrated, or replaced.

Locking the data file alone is insufficient for compaction: an atomic rename
changes which inode the canonical path names. A writer holding only the old
data inode lock no longer excludes another opener of the replacement inode.
The `.lock` inode does not move during replacement, so the same exclusive lock
remains held across staging, rename, synchronization, and reopen.

The data file continues to receive its previous advisory lock as a transition
compatibility measure. This prevents concurrent access from older builds that
know only about the data-inode lock. New builds require both boundaries for a
mutable open and use a shared coordination lock for read-only inspection.

The `.lock` file is durable coordination state, not disposable temporary data.
It should remain beside the store even when no process currently holds it.
Deleting or replacing it while a store is open can split the lock domain and
must be treated as an operator error.
