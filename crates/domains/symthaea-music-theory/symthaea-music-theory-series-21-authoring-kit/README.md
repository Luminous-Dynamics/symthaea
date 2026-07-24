# Symthaea Music Theory — Patch Series 21 Authoring Kit

**Theme:** Post-recovery publication resumption with incident-boundary continuity.

This is a code-ready authoring kit, not a claim that Git mail patches were produced or compiled. The exact Patch Series 20 source snapshot is not present in the active artifact runtime, so mechanically applicable Rust diffs cannot be generated honestly here.

The kit defines:

- the complete 18-patch landing order;
- proposed public contracts and mutation-boundary rules;
- adversarial and end-to-end test matrices;
- persistence-role and compatibility requirements;
- operator workflow and release documentation;
- reproducible package checksums.

## Core rule

A recovered witness-policy anchor does **not** by itself reopen publication. Publication resumes only after a fresh checkpoint extending the selected recovery branch is witnessed under the recovered policy, observed under the configured mirror policy, free of unresolved branch conflicts, and bound into an externally authenticated resumption authorization.

## Expected base

Patch Series 20, whose application guide names Patch Series 19 tree
`3136970a475d4e70adb6f0eaf292c1eb7e103910` as its base.

Before authoring code patches, fill `BASELINE_REQUIRED.txt` with the exact Series 20 final Git tree and source archive SHA-256, then replay every patch against that byte-exact tree.
