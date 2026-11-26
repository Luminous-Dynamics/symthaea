# MON-015 semantic adapter

Standard-library Go package for normalized AAC-06 effect semantics, OAR 0.3 lifecycle projection, typed digest comparison, assurance-axis reconciliation and digital sink/observer joining.

It is not a cryptographic verifier. Maintained upstream verifiers must run first.

Run:

    go test -race -count=5 ./...
    go vet ./...
    go test -count=1 -cover ./...
    go test -run '^$' -bench . -benchmem

Fuzz targets:
- FuzzDigestMutationNeverEqualsAcrossContextChanges
- FuzzOARConfirmationCannotAppearWithoutExecution
- FuzzAACOverclaimNeverPasses
- FuzzCrossTenantScopeNeverQualifies
- FuzzOARCrossRunReferenceNeverConfirms

qualified_digital_effect does not prove physical effect. Shared control roots do not count as independent observer control. Unknown/missing evidence never grades upward.
