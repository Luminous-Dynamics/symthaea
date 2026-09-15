# IG-008C0 — Mycelix governance-config cross-repository conformance

## Status

Research / `MeasurementOnly` / `CrossImplementationConformance`.

IG-008C0 is the runtime-config authority companion to IG-008A0 voting conformance and IG-008E0 execution conformance.

## Symthaea lineage

Parent:

- IG-008E0 / draft #3254
- exact parent head `c33ca471b68fdc977300800bc09a3a72ef8c665e`

## Mycelix evidence lineage

- IG-007C0 / draft #946 — observed config-authority profile
- IG-007C1 / draft #950 — config-authority counterexamples
- exact evidence head `88922e7950ddf030026b5d4d2b06e0ed63727692`

The profile binds production subject:

`fca2c107a1ea5108823ce617ba4111b6f7f77230`

## Exact source blobs

```text
bridge/coordinator/src/consciousness_config.rs
26da234e588bf26d0e25c10dbec34502e00c191a

bridge/integrity/src/lib.rs
61b20610216e2fd69c701ecaea5322c448eda119

proposals/coordinator/src/lib.rs
eb8358353ee259ef9c3b46617a61d3439f1c714c

proposals/integrity/src/lib.rs
986bc0526aec8d37436efbe5ba798bc41705e3cf
```

## Content-bound mechanism reference

```text
id             mycelix-governance-config-observed-fca2c107-v1
revision       1
content_sha256 de4435a69356557b1812f8beb46d654c66b9c957be9d18c64bd0431f92546d5a
authority      ObservedSourceBound
```

C1 corpus:

`3009e97529934fa8f470769de5615dcfda17bde0e942683e939d2b733430a216`

## Independent Symthaea implementation

`scripts/ig008c0_mycelix_config_oracle.py`

The oracle is stdlib-only and does not import:

- Mycelix C0 validator;
- Mycelix C1 counterexample oracle;
- Mycelix Rust bridge/proposal code;
- Symthaea institutional-lab production implementation.

It independently checks the source/profile semantics needed by the corpus and reconstructs:

- `CE-CFG-01` — proposal-record existence without observed status/type inspection;
- `CE-CFG-02` — structurally valid lower runtime gate;
- `CE-CFG-03` — config-shape integrity without observed proposal/caller authority reconstruction.

## Qualification theorem

A PASS means only:

```text
exact observed config-authority profile
+ independent Mycelix implementation
+ independent Symthaea implementation
+ byte-identical canonical corpus
= CrossImplementationConformance
```

It does not establish that the observed mechanism is safe, desirable, live-exploitable, or deployment-current.

## Exact cross-repository workflow

The workflow must:

1. check out exact Symthaea PR subject;
2. check out Mycelix exact evidence head `88922e...` in an isolated nested path;
3. verify all four exact Mycelix source blobs;
4. syntax-compile the Mycelix and Symthaea implementations;
5. run Mycelix C0 validation;
6. run Mycelix C1 twice and capture canonical corpus;
7. run Symthaea independent oracle twice and capture canonical corpus;
8. require complete byte equality of the corpus files;
9. assert exact profile/corpus commitments and authority ceilings;
10. verify both working trees remain immutable.

No rebinding to branch tips and no post-hoc normalization of divergent output are permitted.

## Composition boundary

Once IG-008A0, IG-008E0 and IG-008C0 are all qualified, Symthaea can build a composed observed-governance manifest containing:

```text
VotingProfileRef
ExecutionProfileRef
GovernanceConfigProfileRef
```

That composition must explicitly enumerate what remains unmodeled. It must not transform three partial source profiles into a claim of complete institutional coverage.

## Successor comparison

After #943 is repaired, the corrected Mycelix config-authority profile should produce a new content identity. Symthaea should independently qualify the successor and run before/after differential experiments rather than rewriting this historical adapter.

## Non-claims

No live config mutation, exploit success, deployment currentness, normative threshold recommendation, or governance-safety claim.