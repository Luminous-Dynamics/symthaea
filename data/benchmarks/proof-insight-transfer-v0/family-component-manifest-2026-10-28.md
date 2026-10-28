# Complete PITB component manifest

Status: prerequisite implementation and synthetic controls only. This does not claim the actual relationship inventory is complete, that the primary scorer consumes component IDs, or that PITB is statistically qualified.

The existing leakage report emits cross-split hard-link components for quarantine. It is not a complete clustering key because same-split groups and isolated families are absent. The new builder maps every frozen family ID to a deterministic component ID. Confirmed and unresolved hard relations join components; dismissed hard relations and soft relations such as conceptual similarity do not. Every isolated family becomes a singleton. Component IDs are SHA-256 content addresses of sorted family IDs, and the manifest binds the frozen split digest and exact relationship-inventory bytes.

Qualification requires independently reviewed complete coverage with a receipt digest, complete family membership, and no non-dismissed hard component crossing train/validation/test. Unresolved hard links are conservatively included. A digest is not authentication and does not prove that the coverage claim is true.

The builder does not yet change the production scorer. The run configuration, power planner, and test bootstrap must all bind and consume the same manifest digest and component size vector before the minimum of 20 independent components can be enforced. Until then no confirmatory PITB result is qualified.
