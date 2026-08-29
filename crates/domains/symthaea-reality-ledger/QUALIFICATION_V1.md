# Reality Ledger v1 Qualification

Qualification proves only the implemented provenance mechanics. It does not
establish subjective experience, metaphysical status, physical sensor truth, or
active-policy safety outside the tested contract.

## Mechanical gates

Run under the project Nix development shell:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-reality-ledger --all-targets
cargo test -p symthaea-reality-ledger
cargo clippy -p symthaea-reality-ledger --all-targets -- -D warnings
```

Then retain exact HEAD, TREE, rustc, cargo, Nix environment, architecture and
relevant build flags.

## Required semantic gates

The tests/review must establish:

1. root and derived world descriptors validate fail closed;
2. counterfactual/replay/dream worlds require explicit parents;
3. nested world entry requires the currently active parent and exact generation depth;
4. context cycles and excessive nesting are rejected;
5. ledger sequences and previous-digest links are exact;
6. empty ledgers do not count as verified evidence;
7. evidence sources cannot claim an incompatible reality layer;
8. physical derived computation is not relabeled direct physical observation;
9. counterfactual and dream memories remain `HypotheticalOnly` after recall;
10. replay/imported/unknown memories never upgrade into committed occurrence claims;
11. counterfactual materialization requires an external authority receipt;
12. the selected counterfactual state digest must equal the committed after-state digest;
13. committing a counterfactual does not mutate or relabel the source world descriptor;
14. dream worlds cannot use the counterfactual commit gate directly;
15. no API in this crate grants host mutation or actuator authority.

## Adversarial cases

Review or add tests for:

- forged parent identity;
- self-parent/cyclic nesting;
- skipped generation depth;
- duplicate record IDs;
- reordered or deleted ledger entries;
- forged previous-record digest;
- empty/unknown evidence-source identifiers;
- dream source claiming `PhysicalGrounded`;
- counterfactual event recalled as current-world history;
- replay treated as a new present observation;
- derived inference treated as raw sensor observation;
- commit receipt with a mismatched target world;
- commit receipt whose target state differs from the selected ghost state;
- missing external authority receipt.

## Policy fence

Static review should find no host mutation mechanism, actuator call, policy
selection loop, hidden scalar reward, or claim that any layer proves subjective
experience.

## Host integration remains separate

A passing crate qualification does not prove Symtropy/Bevy, robotics, dream or
replay adapters emit correct provenance. Each host adapter needs its own live
qualification against this contract.
