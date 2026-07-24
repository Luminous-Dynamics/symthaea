# External Review and Confirmatory Readiness

V11 adds a final evidence boundary between pilot completion and confirmatory participant collection. It does not strengthen any musical-quality claim by itself. It prevents the confirmatory study from opening until independent review, operational validation, governance, and reproducibility evidence all agree that collection is ready to begin.

## Required external roles

The frozen review protocol requires independent coverage of four specialties:

- quantitative methods and statistical design;
- music theory and composition;
- human-subjects and participant-experience review;
- reproducibility engineering.

Each reviewer receives a reviewer-specific package. Methods reviewers remain blinded to policy assignments. Reproducibility reviewers may receive embargoed build and release authorities, but no package can contain participant-identifying data.

## Review lifecycle

1. Seal and externally timestamp the review protocol.
2. Seal the complete evidence index.
3. Build a least-privilege package for each registered reviewer.
4. Collect sealed answers to every blocking question.
5. Preserve every finding, including informational and rejected findings.
6. Resolve major and critical findings through concrete replacement evidence.
7. Obtain reviewer acceptance of each resolution.
8. Seal the review-completion record.

A major or critical finding cannot be deferred. A blocking finding cannot be rejected solely by the study author.

## Amendment boundary

The confirmatory authority snapshot binds the manifest, methodology, analysis plan, adaptive checkpoint, runner, artifact factory, preregistration receipt, and completed external review.

Any material change to an instrument, endpoint, analysis, model, hypothesis, privacy control, or participant flow requires:

- a full refreeze;
- a new external preregistration receipt;
- a new external-review resolution;
- a new authority snapshot and amendment ledger.

After confirmatory collection begins, accepted amendments are forbidden. The only allowed post-start action is an evidenced emergency safety stop.

## Ready/not-ready gate

All nine gates are blocking:

1. affected-workspace validation;
2. disposition of claim-relevant ignored tests;
3. clean pilot closure;
4. completed external review;
5. locked amendments;
6. human-study governance;
7. end-to-end synthetic dry run;
8. independent reproduction;
9. external preregistration.

The gate emits `ReadyForConfirmatoryCollection` only when every check passes. A failed check cannot be overridden by a narrative justification.

## Workspace evidence

Workspace evidence should retain the exact source revision, source-tree digest, Nix lock, toolchain versions, commands, full logs, test counts, ignored-test registry, release build, formatting, Clippy result, and clean-tree status.

The five required target names in the V11 record are:

- `symthaea-music-theory`;
- `symthaea-fep`;
- `symthaea-muse-lib`;
- `cognitive-study-bin`;
- `muse-studio-bin`.

## Synthetic dry run

The dry run must exercise the complete artifact, runner, collection, compilation, Rust analysis, independent analysis, crosscheck, policy-budget, and release path using synthetic evidence only. It must also prove that deliberate corruption is rejected.

A successful dry run demonstrates pipeline integrity. It does not demonstrate better music.

## Independent verification

The Rust CLI constructs and validates the evidence. `scripts/verify_cognition_study_v11.py` independently verifies the V11 commitments and root release using only Python's standard library.

The independent verifier expects a directory containing the conventional V11 evidence filenames documented in its `FILES` table.

## Claim boundary

A V11-ready decision means the study is operationally and methodologically ready to collect its frozen confirmatory evidence. It is not a positive experimental result, a listener-preference result, or proof that temporal cognition improves composition.
