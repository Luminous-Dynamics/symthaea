# Evidence Capsule Integrity

Series 51 prevents release evidence from becoming a directory of unauthenticated files that can be accidentally or deliberately mixed across builds.

Each evidence artifact is authenticated and bound to:

- the exact release;
- source and deployment identity;
- calibration and hardware identity;
- its producer and creation time;
- a declared privacy class.

The capsule verifies required artifact classes, anti-rollback chaining, artifact freshness, unique identities and digests, independent reviewers, and separation between evidence producers and reviewers. Pseudonymous or restricted evidence must be redacted; data classified as prohibited from a release capsule is always rejected.

The module defines verification contracts but deliberately delegates signatures and digest construction to separately reviewed cryptographic code.
