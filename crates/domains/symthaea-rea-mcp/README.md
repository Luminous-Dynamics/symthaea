# symthaea-rea-mcp

Read-only stdio MCP adapter for the native REA Evidence verifier.

## Why this is an adapter, not the internal API

Symthaea and mobile applications should call the typed Rust APIs directly. This executable exists for external MCP clients that need an interoperable interface. It invokes the same functions as the offline Evidence CLI; it does not reimplement parsing or hashing.

## Tools

Exactly two tools are exposed:

1. `rea_verify_evidence_json` — accepts an Evidence JSON record as inline text.
2. `rea_verify_evidence_bundle_json` — accepts an EvidenceBundle as inline text and verifies embedded Evidence plus residual-unknown graph consistency.

Both inputs are capped at 8 MiB in this MCP boundary (the offline verifier has its own 64 MiB limit). Inputs are content, never paths. Extra input properties are rejected by the typed parameter schemas.

## Output and claim ceiling

Only a redacted summary is returned:
- operation status and stable error category;
- exact raw-input SHA-256;
- semantic Evidence ID/digest or canonical bundle digest;
- bundle record/revision counts;
- explicit claim-ceiling booleans.

The adapter never returns subject display names/paths, raw results, normalized results or embedded bundle records.

A verified result means internal semantic/canonical consistency only. It does not authenticate the producer, prove that observations occurred, validate hardware attestation or runtime integrity, or certify platform/application security.

## Effects

The server uses stdio only. The tools:
- do not read local files or accept caller-supplied paths;
- do not spawn providers or child processes;
- do not access the network, ADB or a device;
- do not extract, decrypt, write or execute artifacts;
- are annotated read-only, non-destructive, idempotent and closed-world.

Tool annotations are hints, not an authorization mechanism. The implementation itself only accepts inline JSON text and calls pure verification APIs.

## Run

After building with the pinned workspace toolchain:

```bash
cargo run -p symthaea-rea-mcp --bin rea-evidence-mcp
```

The executable writes no startup banner to stdout; stdout is reserved for MCP protocol frames.

For clients that need an in-process Rust interface, depend on `symthaea-rea-compat` directly instead of starting the MCP process.

## Validation

The dedicated workflow runs an in-memory RMCP client/server protocol test, checks the exact two-tool catalog and annotations, verifies the golden Evidence record, tests redaction and stable error categories, and consumes the pinned upstream-generated EvidenceBundle when available.

The standalone locked harness lives under `tools/rea-mcp-harness`. It compiles the exact production MCP sources and the exact Evidence compatibility source without pulling the repository's root workspace glob into the focused test.

## Claim ceiling

This adapter does not replace the full REA MCP catalogue. It does not provide APK/IPA analysis, ADB access, firmware parsing, arbitrary provider execution, or a general-purpose reverse-engineering agent. It exposes only the two local consistency-verification operations above.
