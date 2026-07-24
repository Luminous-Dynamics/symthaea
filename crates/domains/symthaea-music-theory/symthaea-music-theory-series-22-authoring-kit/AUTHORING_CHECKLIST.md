# Series 22 authoring checklist

- Pin the exact verified Series 21 tree.
- Enumerate all public persisted contracts before choosing vector coverage.
- Reuse existing canonical functions; do not invent a parallel encoding accidentally.
- Generate vectors from independent code paths and compare exact bytes.
- Require at least one verifier not linked to the Rust crate for final acceptance.
- Bound fixture size, nesting, process output, and execution time.
- Keep public kits free of participant identities, credentials, signatures not intended for publication, and private governance records.
- Replay all vectors on 32-bit and 64-bit targets where supported.
- Rebuild archives in clean environments and require byte identity.
