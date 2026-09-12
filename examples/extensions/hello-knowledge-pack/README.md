# Hello Knowledge Pack

Minimal **data-only** Symthaea extension. It contains no Rust crate, WebAssembly
component, native library, process, or script.

The example exists to make the lowest-authority extension path concrete for
researchers, educators, artists, domain experts, and other contributors who do
not need executable code.

## Files

- `manifest.json` declares package identity and the semantic capability
  `knowledge.example.hello_records`.
- `hello-records-v1.schema.json` is the capability-specific payload contract.
- `records.json` is a tiny synthetic payload satisfying that contract.

The payload schema is intentionally **not** part of the universal extension
manifest. Each semantic capability should own a typed/schema-versioned data
contract rather than making Symthaea interpret arbitrary JSON as trusted
knowledge.

## Security properties

`runtime: "data_only"` means the package cannot execute guest code. The v1
manifest contract also requires data-only extensions to request no ambient host
permissions and to expose only `pure` capabilities.

This removes an entire execution attack surface, but does not make the content
true or automatically safe. A real loader still needs to:

1. enforce package/file size and count limits before parsing;
2. validate `manifest.json` structurally and semantically;
3. select the payload parser from the admitted capability/schema contract;
4. validate the payload against that exact schema version;
5. bind manifest + payload bytes to immutable digests;
6. verify provenance/signatures where policy requires them;
7. pass resulting claims/evidence through the domain's epistemic policy before
   treating them as knowledge.

## Intended author experience

A future CLI should make this approximately:

```text
symthaea extension new --kind knowledge-pack hello-pack
symthaea extension check ./hello-pack
symthaea extension pack ./hello-pack
```

No Rust toolchain or Wasmtime should be required for a data-only pack.

## Important non-goal

This example does not define a universal knowledge-record format. The
`hello-records-v1` schema is deliberately local and synthetic. Scientific,
medical, legal, geographic, musical, or other domain packs should use their own
versioned contracts and validation/evidence requirements.
