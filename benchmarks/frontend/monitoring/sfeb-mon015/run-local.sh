#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root/semantic-adapter"
go test -race -count=5 ./...
go vet ./...
go test -count=1 -cover ./...
go test -run '^$' -bench . -benchmem
for f in \
  FuzzDigestMutationNeverEqualsAcrossContextChanges \
  FuzzOARConfirmationCannotAppearWithoutExecution \
  FuzzAACOverclaimNeverPasses \
  FuzzCrossTenantScopeNeverQualifies \
  FuzzOARCrossRunReferenceNeverConfirms
do
  go test -run '^$' -fuzz "$f" -fuzztime=30s
done
