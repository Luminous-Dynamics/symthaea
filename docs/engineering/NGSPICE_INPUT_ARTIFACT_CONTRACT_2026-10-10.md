# ngspice Input Artifact Contract — 2026-10-10

## Status

This note documents the first request-bound input identity primitive added in
PR #7315. It is not a solver executor, a claim that ngspice ran, or a physical
qualification record. The production adapter deliberately continues to reject
real execution until the full run boundary exists.

## Implemented: primary netlist identity

`symthaea-ngspice-bridge::input::NetlistArtifact` carries:

- the exact UTF-8 netlist bytes (line endings and whitespace are not normalized);
- the request ID to which the artifact was bound at construction;
- a lowercase hexadecimal BLAKE3 digest of those exact bytes.

Construction rejects an empty, non-canonical, over-256-byte, or control-character
request ID; an empty/whitespace-only netlist; invalid UTF-8; embedded NUL bytes;
and inputs over 4 MiB. The fields are private and read-only accessors expose
bytes and digest. `verify_for_request` recomputes the digest and rejects
request-ID mismatch. Tests cover exact hashing, LF vs CRLF identity, invalid
input, size limits, canonical request identity, request mismatch, and digest
tampering.

The shared `SimulationRequest` was intentionally not changed. This avoids
breaking the other solver bridges while the artifact semantics are being
established.

## Implemented: closed-world static include bundle

`symthaea-ngspice-bridge::bundle::ModelInputBundle` accepts a primary file and
explicit dependency bytes. It recognizes a deliberately narrow directive
subset: `.include path`, `.incpslt path`, and external `.lib path section`.
Paths use canonical relative slash form; absolute/drive paths, backslashes,
environment-expanded paths, empty components, `.`, and `..` are rejected.
Quoted paths may contain spaces. Relative nested references are resolved from
the including file's directory. This aligns with the ngspice 43+ source change
that adds the path of an included file to the search path for subsequent
includes; see the upstream [ngspice NEWS](https://github.com/imr/ngspice/blob/master/NEWS).
Actual execution must still pin a specific ngspice package/version and matching
sourcepath policy.

Construction fails closed for missing dependencies, missing requested
external library sections, duplicate paths, unused extra files, include cycles,
malformed quoting, unsupported directive operand counts, unbalanced in-file
`.lib`/`.endl` sections, and mismatched section names. External
`.lib file section` requests now verify that the dependency actually declares
the selected section, case-insensitively; a file that exists but lacks that
corner/library section is rejected. File count, aggregate byte count, and
include-directive count are bounded (256 files including primary, 32 MiB total,
and 4,096 static include directives). The bundle exports canonical versioned
manifest bytes with length-prefixed, domain-separated fields and sorted
dependency paths; its manifest digest is BLAKE3 over those exact exported bytes.
Verification checks the request identity, each file digest, closure, and
manifest digest before use.

This is **static include closure**, not a proof of full model closure or safe
execution. The parser does not discover arbitrary data files, control-language
`source` commands, dynamically generated files, Verilog-A/OSDI modules, or
other runtime loads. The legacy compatibility interpretation of one-operand
`.lib filename` is intentionally not supported; controlled execution must pin
compatible library semantics. No files are read from disk by the bundle API,
and it does not enable the production adapter.

## Explicit limitations

The type identifies **one primary file only**. It does not yet represent or
verify transitive `.include` / `.lib` model dependencies, external data files,
the solver executable bytes/version, or the environment closure. A digest is
identity, not authenticity or a claim that input is safe to execute.

The generic `SimulationBackend::run(request)` does not accept this artifact,
so the production ngspice path remains fail-closed. It will not run an ambient
`./input.sp`. Do not convert the artifact into a filesystem path and execute
it outside a controlled runtime boundary.

## Required next boundary before execution can be enabled

1. **Input bundle:** primary netlist plus every referenced model/library, each
   with canonical relative path, exact bytes, and digest; reject path traversal,
   absolute paths, unresolved includes, duplicate paths, and unlisted external
   data dependencies. Resolve the input dependency graph before hashing a
   canonical bundle manifest.
2. **Execution isolation:** run in a per-run directory inside a declared
   sandbox/container/Nix derivation with no ambient home configuration, no
   network, minimal read-only dependencies, bounded CPU/memory/time/output, and
   no writable host paths. A netlist's control-language commands must be treated
   as executable input, not assumed passive data.
3. **Pinned solver identity:** record the exact executable/package identity and
   version output, command arguments, selected simulator options, parser version,
   and effective environment allowlist. Keep a content digest for the solver
   package or immutable store path, not only a version string.
4. **Immutable artifacts:** preserve the exact netlist bundle, solver stdout and
   stderr/log, raw output, exit status, timeout state, and hashes in a unique
   run directory. Fail on stale/missing/truncated output; never reuse a previous
   run's output file.
5. **Separated statuses:** process completion, parse success, simulator
   convergence evidence, requested metric completeness, and physical/model
   validation must be represented separately. Exit status zero and successful
   numeric parsing are not convergence proof or physical validation.
6. **Independent checks:** validate the RC transient against the analytic
   reference with explicit tolerance and sampling/interpolation policy, then
   add RLC and DC-operating-point references. Negative cases must prove that
   invalid inputs and missing/contradictory outputs fail closed.

## Solver-version note

The official ngspice documentation currently lists the release manual for
version 47. Its command-line manual documents `-n` (disable user
`.spiceinit` loading), `-b` (batch mode), `-o` (batch log output), and
`-v` (version output). These options are helpful, but `-n` alone is not a
sandbox: netlist control commands and model files still require the isolation
and dependency controls above.

References:
- https://ngspice.sourceforge.io/docs.html
- https://ngspice.sourceforge.io/docs/ngspice-manual.pdf
- https://ngspice.sourceforge.io/ngspice-control-language-tutorial.html
