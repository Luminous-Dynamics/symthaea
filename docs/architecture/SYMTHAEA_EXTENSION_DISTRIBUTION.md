# Symthaea Extension Distribution

## Status

Proposed distribution model for the public extension ABI introduced alongside
`SYMTHAEA_EXTENSION_ABI.md`.

The design goal is deliberately conservative: Symthaea should own extension
**policy**, not invent a second WebAssembly package ecosystem.

## Core decision

Public executable extensions are standard WebAssembly Components. WIT packages
and component artifacts should remain consumable by ordinary Component Model
tooling.

For network distribution, prefer the Bytecode Alliance `wasm-pkg-tools` / `wkg`
flow and OCI registries rather than a proprietary Symthaea registry or archive
format. Local development may use a plain directory containing a manifest and
component.

This gives Symthaea:

- standard component artifacts;
- standard WIT tooling;
- OCI content-addressed distribution;
- existing registry/authentication infrastructure;
- cross-language guest support;
- fewer custom parsers and supply-chain surfaces to secure.

Symthaea still adds the pieces the generic component ecosystem intentionally
does not define: admission policy, signer authorization, permission grants,
resource budgets, semantic capability routing, evidence requirements, and
safety gates.

## Logical package

A Symthaea extension release consists logically of:

```text
ExtensionRelease
├── manifest.json          signed policy/declaration input
├── component.wasm         WebAssembly Component
├── provenance             optional build/source attestation
└── signature/identity     distribution-specific trust material
```

The logical files need not be wrapped in a custom archive. An OCI artifact may
carry them as descriptors/layers; local development may keep them in a normal
directory.

### Manifest

`manifest.json` is an encoded `ExtensionManifest` from
`symthaea-extension-core`. It describes identity, ABI compatibility,
capabilities, requested permissions, runtime kind, and resource budgets.

The manifest is authoritative for policy. Executable guest code does not get to
increase its own permissions or trust level at runtime.

### Component

`component.wasm` implements the declared WIT world(s). The baseline public guest
implements `extension-control-v1`; capability-specific worlds are added only for
real capability families.

### Provenance

Provenance is optional at the file-format level but may be mandatory under host
policy for higher-trust or safety-relevant capabilities. Examples include source
repository identity, source revision, builder identity, reproducible-build
metadata, SBOM references, and independent validation evidence.

## Distribution modes

### Local development

A developer may point Symthaea at a directory or explicit manifest/component
pair. Local installation must still run the same validation pipeline as a
network-fetched package. `--dev` may relax signer requirements only when the
operator explicitly requests it; it must not grant additional runtime authority.

### OCI / `wkg`

Public distribution should prefer standard Component Model package tooling and
OCI registries. Registry location is a transport concern, not part of extension
identity.

Do not equate any of the following:

```text
ExtensionId     != WIT package name
ExtensionId     != OCI repository/reference
ExtensionId     != signer identity
```

Those identifiers serve different purposes and should be bound together by the
signed release metadata rather than overloaded into one string.

## Admission pipeline

A downloaded artifact is not executable merely because it is syntactically valid
Wasm.

Recommended order:

```text
fetch / select bytes
        ↓
verify content digest
        ↓
parse + structurally validate manifest
        ↓
verify package signature / provenance
        ↓
authorize signer under local trust policy
        ↓
inspect component WIT imports/exports
        ↓
confirm control identity + ABI + manifest digest
        ↓
check requested permissions/resource budgets
        ↓
register metadata as admitted provider
        ↓
router may select provider for an invocation
        ↓
host instantiates with only granted imports
```

Every arrow is fail-closed.

## Signature validity is not trust

A package self-signed by an unknown key may be cryptographically intact and still
be unauthorized. Symthaea must keep these checks separate:

1. **integrity** — do the bytes match the referenced digest?
2. **signature validity** — does the signature verify for the claimed signer?
3. **signer authorization** — does local policy trust that signer for this role?
4. **capability authorization** — may this extension perform the requested class
   of work here?
5. **invocation authorization** — may it perform this specific action now?

The first three are package-admission concerns. The latter two remain runtime
policy concerns.

## Dependency resolution

`ExtensionManifest.requires` names semantic capabilities, not packages. The host
resolves requirements against the **already installed and admitted** capability
catalog.

V1 should not silently download transitive executable plugins to satisfy a
missing capability. Automatic transitive code installation creates avoidable
supply-chain and consent problems. A host may instead explain which capability
is missing and offer an explicit installation flow.

Data-only knowledge packs can later use a lower-risk dependency policy because
they carry no executable authority.

## Reproducibility

When an extension release is selected from a mutable tag or package name, the
resolved immutable OCI/content digest should be recorded by the consuming
profile or deployment evidence. Replays and evidence exports should refer to the
resolved digest, not merely a human-readable version tag.

No new Symthaea-specific lockfile is required until a concrete multi-extension
profile consumer demonstrates that existing deployment/profile evidence is
insufficient.

## Updates

Updates are new releases, never silent replacement of already-admitted bytes.
Before activation, an update must repeat admission and compatibility checks.
Permission expansion is a material policy change and should require explicit
operator approval even when the signer is already trusted.

Downgrade/rollback should remain possible by immutable digest where local policy
permits it.

## Revocation

The host should eventually support local revocation by at least:

- extension ID + version/digest;
- signer identity;
- capability grant.

Revocation data is host policy, not a field the extension controls.

## Why no `.symplug` archive yet

A custom archive looks convenient but immediately creates work around path
traversal, duplicate entries, canonicalization, signature scope, compression
bombs, metadata versioning, registry hosting, mirrors, and tooling.

A plain development directory plus standard OCI/component distribution covers
the first real consumers with less code and a smaller security surface. A custom
single-file UX can be added later as a thin transport wrapper if users actually
need it.

## Success criterion

A third-party author should be able to build and publish a Symthaea-compatible
component with standard Component Model tooling, while a Symthaea operator can
inspect exactly what the extension claims, who authorized it, what authority it
will receive, and which immutable bytes will execute.
