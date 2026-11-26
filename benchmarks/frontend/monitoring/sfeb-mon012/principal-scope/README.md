# Principal/scope semantic preflight

Standard-library Go checker for adapter-verified principal/grant scope, destination-global operation identity and independence from explicit control-root evidence.

Run:

    go test -race -count=5 ./...
    go vet ./...
    go test -count=1 -cover ./...

Verified is a normalized adapter assertion. This package does not authenticate database sessions, credentials, signatures, revocation feeds, grant issuers, COSE/SCITT receipts or control-root declarations. It is not production authorization middleware.
