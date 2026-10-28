# MEL-MR-EVAL-004 QA scope

- Dataset version: `MEL-EDIT-LOCALITY-SMOKE-1.0`
- Contract version: `edit-locality-v1.0`
- Cases: 132 total; 60 accept / 72 reject.
- Families: 12 distinct motif families; development 6 families / 66 cases, calibration 2 / 22, test 4 / 44.
- Per-family coverage: five valid operations plus an under-edit and five collateral-change controls.
- Operations: global transposition, duration scaling, onset scaling, one-note pitch edit, one-note duration edit.
- Negative controls: requested transposition omitted, onset corruption during transposition, pitch corruption during duration scaling, duration corruption during onset scaling, non-local pitch leakage, non-local duration leakage.
- JSON corpus records use stable case IDs and family-disjoint splits.
- QA performed: JSON parse, unique-ID check, split isolation, dataset-version consistency, target-operation oracle checks, and an independent JavaScript simulation of the same property equations.
- **Not established:** Rust compilation/execution, test pass, clippy/fmt, audio rendering or perceptual outcomes, human-annotated edit performance, commercial competitiveness, or generalization.

## Independent reference simulation

- 132 cases: 60 expected accepts and 72 expected rejects.
- Split counts: development 66, calibration 22, test 44.
- After correcting a contradictory protected-pitch check for the transpose operation, reference simulation verdict mismatches: 0.
- The Rust corpus replay test and receipt round-trip are authored but not executed in this environment; `cargo`, `rustc` and `rustfmt` are unavailable.

## Contract boundary

The v1.0 verifier is index-preserving and monophonic. It fails closed on event-count changes and unsupported/malformed requests. It verifies only pitch, onset and duration. It does not establish preservation of timbre, instrumentation, dynamics, harmony beyond represented note events, simultaneity/polyphony, or rendered audio. Thresholds are preliminary engineering values and must be calibrated without test split leakage.
