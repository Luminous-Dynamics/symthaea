# Rust-Native REA Compatibility Core: Architecture Decision and Migration Plan

**Status:** accepted research direction; implementation not yet qualified  
**Date:** 2026-11-02  
**Parent:** [REA-RUST-001 / #9887](https://github.com/Luminous-Dynamics/symthaea/issues/9887)  
**Mobile programme:** [MOBILE-REA-001 / #9836](https://github.com/Luminous-Dynamics/symthaea/issues/9836)  
**Research baseline:** canonical upstream `morluto/rea` main at `8250bf2320ea57aab4e664db4b4d44b8abd2004e`

## Executive decision

**Do not rewrite all of REA in Rust. Build a Rust-native, REA-compatible core in stages.**

The immediate value is not a language rewrite. It is the ability for Symthaea to consume, validate, reason over and eventually produce REA-compatible evidence natively, without requiring the Node.js REA runtime for every operation.

Use a strangler migration:
- Keep the TypeScript REA implementation as a reference implementation and provider during migration.
- Build the Evidence contract and tool/capability model in Rust first.
- Expose native contracts through a Rust CLI/MCP facade.
- Preserve specialist external providers where they are the best maintained implementation.
- Selectively port deterministic, self-contained analyzers after exact behavioral parity is demonstrated.
- Shadow-run both implementations on identical inputs before any operation is promoted.

A whole-repository rewrite is not the first milestone and is not required to obtain a useful native Symthaea integration.

## 1. Upstream baseline inspected

The canonical upstream repository is [morluto/rea](https://github.com/morluto/rea), currently an MIT-licensed TypeScript project.

At inspected commit `8250bf2320ea57aab4e664db4b4d44b8abd2004e`:

| Measure | Observed |
|---|---:|
| Package | `rea-agents` |
| Package version | 6.8.0 |
| Node runtime range | 22.23.4+, 24.22.1+, or 26.12.1+ |
| TypeScript files under `src/` | 3,550 |
| `.test.ts` files under `src/` | 1,135 |
| Runtime dependencies | 42 |
| Development dependencies | 28 |
| Java bridge files | 16 |
| Swift bridge files | 5 |
| Python bridge files | 15 |

The recursive source tree has 7,545 entries and is not truncated. These are repository inventory counts, not a claim that every file is production code or every test is a unique test case.

REA has a layered design. The source separates domain contracts, provider adapters, application workflows, composition, CLI and MCP adapters. Its documentation explicitly asks contributors to preserve identity, uncertainty, provenance and lifecycle ownership across those boundaries.

REA also delegates to specialist tooling. Ghidra, Hopper, IDA, JEB, JADX, Apktool, Binwalk, Unblob, ADB, TShark, sigrok and browser tooling are not made redundant by changing the orchestration language. The repository also has Java, Swift and Python bridge code. A Rust port would still need to manage those integrations where required.

**Implication:** the port is a substantial compatibility programme, not a source-to-source translation. The strongest reason to do it is an embedded, reproducible, type-safe Rust core for Symthaea—not eliminating Node.js at any cost.

## 2. The difficult compatibility boundary: REA Evidence

The highest-risk porting mistake would be to deserialize REA JSON and calculate a different identity.

The upstream `src/domain/evidence.ts` contract defines a strict record containing:
- `evidence_id`;
- artifact subject identity and provider identity;
- optional analysis-profile commitment;
- predicate type, operation and parameters;
- `raw_result` and `normalized_result`;
- confidence and authority;
- execution environment;
- limitations and relationship coverage;
- source locations and evidence links;
- optional `caller_input_path`.

### Evidence identity is semantic, not a hash of the JSON file

The canonical REA identifier has the form:

```
ev_<lowercase SHA-256>
```

The digest is calculated over canonical JSON semantic content. It is not the SHA-256 of the raw file bytes.

The semantic commitment includes provider, profile, operation, parameters, raw and normalized results, confidence, authority, environment, limitations, relationship coverage, locations and evidence links.

For an artifact subject, semantic identity uses the artifact digest, format and architecture. The display name and local path are excluded. `caller_input_path` is also excluded from semantic identity.

The analysis profile carries its own digest, and the Evidence schema checks that its provider identity agrees with the Evidence provider.

These distinctions matter:
- Rebinding a display path must not silently change Evidence identity.
- Changing a semantic result, source artifact digest, provider version, limitation or source location must change semantic identity.
- A valid JSON shape with a modified `evidence_id` must be rejected.
- A partial/unknown observation must never be rewritten as complete because a Rust type prefers a simpler enum.
- Raw file digest and semantic Evidence ID are separate identities and must remain separate.

### Canonical JSON requirements

REA's canonical digest implementation uses deterministic object-key ordering and ECMAScript JSON primitive serialization. Its comments explicitly align key ordering with RFC 8785 / UTF-16 code-unit ordering.

RFC 8785 defines a canonical JSON representation to make hashing and signing deterministic. Rust candidates such as `serde_jcs` and `serde_json_canonicalizer` are reasonable starting points, but their behavior must be tested directly against REA's implementation before adoption. Do not infer parity merely from both libraries claiming JCS compatibility.

Cross-language golden cases must cover:
- supplementary-plane Unicode property names;
- string escaping and control characters;
- numbers near ECMAScript formatting thresholds;
- negative zero;
- safe integer boundaries and double-precision behavior;
- nested arrays/objects and large result payloads;
- absent versus explicit null fields;
- Unicode normalization (strings must not be silently normalized);
- duplicate keys and invalid input.

The first Rust milestone is therefore an Evidence verifier/adapter, not a new analyzer.

## 3. Existing Symthaea components and reuse boundary

Symthaea already has `symthaea-evidence-plane`, but its current purpose is declared-versus-measured experimental counters and integrity checks. It is not the full REA Evidence record model and must not be stretched into one.

The directory `crates/domains/symthaea-rea-evidence-verifier` currently contains a README describing an older Unblob receipt verifier, but the inspected main tree contains no Cargo manifest or Rust source for that directory. It is not a buildable Rust package and is not proof that general REA Evidence interoperability already exists. Do not revive it by assuming missing code.

The new native implementation must live in a properly manifested package before it is added to the workspace. The workspace discovers `crates/domains/*`; a README-only directory can break workspace metadata/buildability. Do not introduce another manifest-less workspace child.

Reuse existing cryptographic and serialization dependencies when their semantics fit, but keep REA-specific domain types in a focused crate. Avoid creating a second copy of general Symthaea evidence types unless a clear boundary requires it.

## 4. Target architecture

```
        Existing TypeScript REA (reference/provider)
                         |
                    saved Evidence
                         |
                         v
          Rust Evidence admission + identity core
                         |
               shared capability contracts
                         |
             +-----------+-----------+
             |                       |
             v                       v
     Rust-native analyzers      external provider
       (selective)          adapters / specialist tools
             |                       |
             +-----------+-----------+
                         |
                         v
                Rust CLI / MCP facade
                         |
                         v
              Symthaea cognition/agents
                         |
                         v
             HARDSEC / independent oracle
```

### Plane A — Rust Evidence core

Own:
- strict record admission and schema versioning;
- canonical semantic digest and analysis-profile digest;
- raw artifact and report digests;
- artifact subjects, locations and lineage;
- provider identities and capability/effect declarations;
- coverage, limitations, unknown and unavailable states;
- import/export and portable verification.

Do not:
- reinterpret provider-specific raw reports outside their format owner;
- turn parser success into a security pass;
- authenticate a producer based only on a self-declared provider name;
- infer runtime behavior from static artifacts.

### Plane B — Rust MCP/CLI facade

Use the official Rust MCP SDK, `rmcp`, rather than writing a private MCP stack.

Generate both CLI and MCP tool registrations from one canonical catalog. Keep the catalog's public names and schemas stable. The native facade should make the source of each operation explicit:
- Rust-native;
- delegated to the pinned TypeScript REA runtime;
- delegated to a specialist external provider;
- unavailable on the current host/target.

A missing native implementation must not trigger an unannounced fallback with broader authority.

### Plane C — Provider adapters

Retain external tools when they are the mature implementation:
- Ghidra, Hopper, IDA and JEB for deep native analysis;
- JADX and Apktool for Android application and resource analysis;
- ADB for Android device interaction;
- Binwalk and Unblob for supported firmware extraction;
- TShark and sigrok for their supported packet and logic-signal workflows;
- browsers and other application-specific analysis engines.

The Rust core supervises, validates, bounds, correlates and records these providers. It does not claim to reimplement their internals.

### Plane D — selective Rust analyzers

Prioritize analyzers that are:
- deterministic and self-contained;
- security-sensitive or frequently executed;
- portable across Linux, Android and iOS;
- bounded enough to validate with independent known-answer fixtures;
- valuable without a large proprietary/native engine;
- measurable against the existing TypeScript result.

The first proposed vertical slice is bounded APK/IPA archive inventory. Android OTA/AVB and Apple IPSW/IMG4 lanes are separate follow-ons already tracked by #9839 and #9840.

## 5. Migration protocol

Each operation moves through explicit states:

1. **Reference frozen:** pin upstream REA commit, tool, schema, provider and fixture corpus.
2. **Contract captured:** preserve input, output, Evidence, error, side-effect and lifecycle behavior.
3. **Rust shadow:** compute a parallel result without replacing the TypeScript result or authorizing action.
4. **Differential qualification:** compare the complete semantic output, Evidence ID, coverage, limitations, source ranges and errors.
5. **Adversarial qualification:** fuzz malformed and ambiguous inputs, mutate source identity and inject stale/substituted evidence.
6. **Resource qualification:** measure latency, peak RSS, bytes copied, startup, cancellation and cleanup.
7. **Independent review:** verify claim ceilings and target-side effects.
8. **Selective promotion:** expose Rust as the default for only the exact operation/host/format scope that passed.
9. **Retained rollback:** keep the known-good reference implementation available until a documented deprecation gate is met.

No workflow may skip from implementation to promotion.

## 6. Benchmark programme

Compare three implementation arms against identical immutable inputs:

- **TS reference:** pinned upstream REA.
- **Rust shadow:** the new Rust implementation; its output is non-authorizing.
- **Rust native:** the operation after it passes the exact scope-specific promotion gate.

Keep provider and target cohorts separate. Do not average Android, iOS, Linux, emulator and physical-device outcomes into one number.

| Scorecard | Core measurements | Initial qualification target |
|---|---|---|
| Evidence identity | Evidence ID and profile digest | 100% parity on the frozen golden corpus |
| Schema behavior | Accepted/rejected inputs, required/null/unknown fields | 100% disposition parity |
| Tool catalog | Names, arguments, result schemas and annotations/effects | 100% for each migrated tool |
| Semantic result | Normalized result, coverage, limitations, source coordinates | 100% exact contract parity for declared-supported fixtures |
| Provenance | Artifact/provider identity, raw digest, lineage, evidence links | 100% exact expected graph on known-answer cases |
| Adversarial behavior | Duplicate keys, malformed values, digest substitution, path escape, stale/replayed records | Zero false accepts |
| Lifecycle | Cancellation, timeout, process ownership and cleanup retry | Zero orphaned owned processes; no false cleanup success |
| Authority | Mutating and unknown-effect operations | 100% denied in the read-only mobile profile |
| Performance | Cold start, p50/p95, throughput, peak RSS and bytes copied | Measure matched workloads; no unqualified aggregate claim |
| Mobile viability | Binary size, cross-target build, energy/task, memory and p95 | Report per device/architecture |
| Maintenance | Parity drift, test flakes, supported operations per engineer-month | Track total lifecycle cost |

Do not preregister a blanket Rust performance win before measuring representative workloads. A 30% improvement in p95 latency or peak RSS on a selected pure-Rust hot path is a reasonable stretch goal, not a promised result.

### Mutation corpus

Include, at minimum:
- semantic field mutation with stale Evidence ID;
- display path/name mutation with unchanged semantic ID;
- profile provider/digest substitution;
- explicit null versus omitted optional member;
- unknown member injection;
- duplicate JSON keys;
- invalid number/Unicode/escaping cases;
- stale provider version and source artifact digest;
- replayed evidence and evidence-link substitution;
- partial coverage rewritten as complete;
- path traversal and archive duplicate-path cases;
- cancellation while provider child processes are active and during final cleanup.

Every negative control must either be rejected or remain explicitly unknown/partial. None may create a verified, complete or equivalent result.

## 7. Industry comparison

| Reference | Existing strength | Appropriate role in this programme |
|---|---|---|
| Upstream REA | Broad, fast-moving provider/workflow catalogue and large test suite | Behavioral oracle and source of compatible Evidence |
| Official Rust MCP SDK (`rmcp`) | Maintained Rust client/server protocol implementation | Native MCP protocol layer |
| RFC 8785 / JCS | Language-independent canonical JSON rules | Interoperability requirement for semantic commitments |
| Ghidra / IDA / Hopper / JEB | Mature reverse-engineering engines | Retain as external providers |
| JADX / Apktool | Android application and resource decoding | Retain as external providers |
| Binwalk / Unblob | Firmware region discovery and extraction | Retain as external providers |
| Symthaea | Rust cognitive reasoning, causal analysis and decision-aware experiment selection | Add value above the provider output |
| HARDSEC | Independent claim qualification and evidence assurance | Prevent self-certification |

The competitive thesis is not "Rust has more capability than TypeScript". It is:

**Symthaea can natively reason over REA-compatible evidence, preserve the same semantic identities across languages, and progressively move high-value analysis into a reproducible Rust runtime without losing existing REA capabilities.**

The only evidence for a superiority claim is the matched benchmark.

## 8. Licensing and governance

Upstream REA is MIT-licensed; Symthaea is AGPL-3.0-or-later. A compatible independent implementation based on public behavior/contracts is distinct from copying upstream source. If any upstream implementation code is reused, retain the MIT notices and audit the licenses of the relevant submodules, bridges and bundled dependencies. Do not copy code or third-party tooling into the core merely to speed up a port.

No change is proposed to the upstream REA repository in this work item. Maintain the native compatibility work in Symthaea and keep upstream contribution/maintainer coordination separate.

## 9. First implementation tracks

- [#9888 — REA-RUST-002: Evidence identity/schema parity](https://github.com/Luminous-Dynamics/symthaea/issues/9888)
- [#9889 — REA-RUST-003: Rust MCP/CLI facade and tool/effect parity](https://github.com/Luminous-Dynamics/symthaea/issues/9889)
- [#9890 — REA-RUST-004: bounded APK/IPA inventory vertical slice](https://github.com/Luminous-Dynamics/symthaea/issues/9890)
- [#9838 — read-only mobile REA gateway](https://github.com/Luminous-Dynamics/symthaea/issues/9838)
- [#9839 — Android OTA/AVB semantic evidence](https://github.com/Luminous-Dynamics/symthaea/issues/9839)
- [#9840 — Apple IPSW/OTA semantic evidence](https://github.com/Luminous-Dynamics/symthaea/issues/9840)

## 10. References

- Canonical REA: https://github.com/morluto/rea
- Pinned REA `package.json`: https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/package.json
- REA Evidence: https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/src/domain/evidence.ts
- REA canonical digest: https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/src/domain/canonicalDigest.ts
- REA profile: https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/src/domain/analysisProfile.ts
- Official Rust MCP SDK: https://github.com/modelcontextprotocol/rust-sdk
- RFC 8785: https://www.rfc-editor.org/rfc/rfc8785.html
- Candidate Rust JCS implementation: https://docs.rs/serde_jcs/latest/serde_jcs/
- Candidate Rust JCS implementation: https://docs.rs/serde_json_canonicalizer/latest/serde_json_canonicalizer/

## Claim ceiling

This ADR and benchmark manifest define the Rust-native migration strategy. No general REA Rust port, canonical Evidence Rust verifier, Rust MCP server/client, APK/IPA Rust parser, provider replacement or performance result is implemented or claimed by this document.


## 11. Direct native API first; MCP is an adapter

For Symthaea's own process and mobile embeddings, call the Rust library through a typed in-process API. Do not route every native analysis request through a local MCP server or serialize the same large Evidence payload through JSON-RPC.

Expose the same core through thin adapters:
- **in-process Rust API** for Symthaea cognition and other Rust services;
- **C ABI / Swift bridge** for iOS application integration, with narrow ownership/lifecycle rules;
- **Android JNI or Rust system-service boundary** only where platform integration requires it;
- **CLI** for deterministic automation and offline evidence inspection;
- **MCP** for external coding agents and other MCP clients.

The adapters must call the same core contracts and share the same known-answer corpus. They may differ in transport and delivery, but not in Evidence identity, operation semantics, authority or claim ceilings.

MCP is not an authorization mechanism. The direct API also enforces capability/effect policy; using Rust internally must not bypass the read-only gateway or independent qualification requirements.
