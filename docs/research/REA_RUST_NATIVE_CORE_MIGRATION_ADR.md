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


## 12. Mobile archive inventory: port the reader contract, not just the ZIP parser

A bounded APK/IPA inventory is the first native analyzer, but the upstream reference is more sophisticated than a conventional archive listing.

Pinned REA source includes:
- `src/artifacts/ArtifactReader.ts`: archive-neutral reader contract, entry facts, source identity, declared integrity, coverage, omissions, per-entry open, provenance, container integrity and explicit close.
- `src/artifacts/ZipArtifactReader.ts`: ZIP64-capable lazy reader over `@zip.js/zip.js`, checking signature and overlapping entries, preserving duplicate occurrences through adapter identity, validating CP437/UTF-8 names, and distinguishing symlinks, directories, encrypted entries and regular files.
- `src/artifacts/ArtifactResourceScope.ts`: cancellation, stream ownership, cleanup retry and retention of failed cleanup owners.
- `src/artifacts/inventory/scanReader.ts`: shared path normalization, collision detection, parent resolution, streamed hashing, graph construction and integrity contradiction reporting.
- `src/domain/zipEndOfCentralDirectory.ts` and `zipPackageFormat.ts`: candidate EOCD scanning and declared non-entry limitations.

The Rust migration must preserve the behavior-bearing contracts:
1. Entry occurrence identity is distinct from logical path; duplicate names are not silently collapsed.
2. Entry metadata, actual bytes, declared integrity and content hash are separate observations.
3. The original file identity is rechecked around random reads and streamed entry reads.
4. Overlap, CRC, decompressed-size and archive-signature checks are recorded independently.
5. Declared and observed expansion are bounded before and during decompression.
6. Cancellation and cleanup are first-class outcomes; a failed close retains an owned resource for retry.
7. Encrypted, unsupported and intentionally omitted entries are explicit partial/unavailable states.
8. Inventory does not write extracted files. Extraction has a separate authority and owned output-tree lifecycle.
9. Coverage indicates what was traversed, not whether every claim about the package is true.
10. Leading stubs, APK Signing Block regions before the central directory and trailing ZIP comments remain outside entry inventory unless a separate region-analysis operation reports them.

### Parser selection

The Rust `zip` crate is a candidate parser and differential oracle, not an automatic replacement for this whole contract. Version 8.6.0 is documented with Rust 1.88 MSRV, lower than the workspace's Rust 1.96 toolchain. It supports ZIP64-related formats and common compression modes, but does not support multi-disk archives. Its default feature set enables more than metadata-only inventory needs; do not enable encryption/decompression backends without a separate use case and threat review.

The current Symthaea lockfile already carries zip 1.1.4, 6.0.0 and 7.2.0 transitively. Compare their feature trees and security/lifecycle semantics before adding another direct ZIP version. For the first metadata-only operation, a small audited parser plus the mature crate as an independent differential oracle may be safer and lighter than enabling every codec. For member hashing, stream bytes through bounded readers; never buffer an archive member without an explicit size budget.

Do not add `object` or `goblin` to the first ZIP inventory slice. Evaluate those later for ELF/Mach-O/PE semantic inspection, after archive occurrence and Evidence parity is qualified.

### Archive-reader scorecard

Add a separate archive contract suite with independent fixture IDs for:
- EOCD/comment candidate ambiguity, ZIP64 locator/record, split/multi-disk refusal and trailing data;
- prepended stubs and offset adjustment;
- duplicate paths and duplicate content with distinct occurrence IDs;
- CP437 names, UTF-8 flag, Unicode Path extra fields, invalid byte sequences and canonical path rejection;
- central-directory/local-header mismatch and overlapping compressed extents;
- encrypted entries, symlink entries and unsupported compression;
- CRC mismatch, truncated streams, declared/observed size disagreement;
- archive mutation between admission, enumeration and entry open;
- cancellation during enumeration, member streaming and final cleanup;
- declared-expansion and actual streamed-expansion budgets;
- case and Unicode-normalization collision policy.

Report inventory parity, source stability, false-accept/abstention rate, bytes read, peak memory, p50/p95 and cleanup failures independently. A ZIP parser passing its own archive test suite is not evidence of REA artifact-inventory parity.

References:
- REA ZIP reader: https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/src/artifacts/ZipArtifactReader.ts
- REA reader contract: https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/src/artifacts/ArtifactReader.ts
- REA scanner: https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/src/artifacts/inventory/scanReader.ts
- zip-rs 8.6.0: https://docs.rs/zip/8.6.0/zip/
- object: https://docs.rs/object/latest/object/
- goblin: https://docs.rs/goblin/latest/goblin/


## 13. Measure Symthaea's reasoning value separately from parser parity

Rust-native compatibility can establish that the new implementation preserves REA's semantics. It does **not** establish that Symthaea adds better reverse-engineering decisions. That is a separate experimental question tracked in [REA-RUST-008 / #9911](https://github.com/Luminous-Dynamics/symthaea/issues/9911).

Use three blinded, budget-matched arms on held-out source-derived mobile targets:

- **A — REA alone:** pinned REA tools and their documented output.
- **B — Symthaea without REA-derived analysis:** equivalent permitted raw context, but no derived REA Evidence.
- **C — REA + Symthaea:** Symthaea reasons over immutable REA Evidence and the same raw context.

The target task is not "summarize this binary diff". It is "identify the component or admit uncertainty, then select the next experiment that best separates competing hypotheses."

### Causal investigation loop

1. Build a typed hypothesis graph: candidate cause, supporting observations, contradiction, assumed dependencies, and unobserved links.
2. Identify the competing hypotheses that remain observationally equivalent.
3. Select the next permitted experiment using expected information gain divided by time, energy, execution risk and target disruption.
4. Execute only an authorized, bounded and reversible experiment.
5. Bind the intervention, input state, output trace and environment to separate evidence records.
6. Update the hypothesis distribution and calibration.
7. Abstain where the corpus provides no discriminating evidence or the evidence is stale/incomplete.

Do not let a Symthaea-generated explanation or synthetic trace qualify its own causal hypothesis. Use independently generated planted faults, held-out source mappings and blinded graders.

### Outcome metrics

- Top-1/top-3 component localization and mean reciprocal rank.
- Cause precision/recall and false-cause rate.
- Brier score, expected calibration error and reliability by fault family.
- Selective risk versus coverage as the model abstains.
- Predicted versus measured reduction in hypothesis entropy after the selected experiment.
- Time-to-first-correct diagnosis and total experiment cost.
- Generated test precision/recall and reproduction rate.
- Citation completeness from claim to exact evidence and intervention.
- p50/p95 latency, peak memory and energy per investigation on named hardware.

The proposed gate is at least 20% relative top-3 localization improvement over the stronger single-system baseline without increasing false-cause rate. Calibration ECE <= 0.05 and 15% lower median time-to-correct diagnosis are stretch targets. They are not current results or universal standards.

Keep Android, Apple applications, emulators, simulators and physical-device cohorts separate. Report confidence intervals, per-fault-family outcomes, negative controls and abstention performance. Never pool them into one headline score.

The machine-readable benchmark manifest includes this ablation as its tenth family. See [#9911](https://github.com/Luminous-Dynamics/symthaea/issues/9911).


### ECMAScript number semantics are a compatibility requirement

REA parses JSON through JavaScript, where every JSON number becomes an IEEE-754 binary64 Number before semantic hashing. Rust's serde_json::Value can retain integer tokens as exact i64/u64 values. Passing those directly to an otherwise JCS-compatible serializer can therefore produce different bytes for unsafe integers.

For example, JavaScript parses `9007199254740993` as `9007199254740992`; a Rust parser may retain the original integer exactly. The Rust compatibility core now wraps JSON nodes with a serializer that emits every number as f64 before canonical JCS output. This avoids cloning/normalizing the full tree and matches the upstream ECMAScript numeric representation.

The test corpus now includes unsafe positive/negative integers and the u64 boundary, as well as negative zero and exponent/formatting boundaries. See the regression test in the current implementation PR #9910. This is a required interoperability behavior, not an optional parser hardening.


## 14. External benchmark alignment: public mobile challenges plus held-out agentic RE

The benchmark programme now has 13 machine-readable families. It adds two independent external tracks, tracked in [REA-RUST-010 / #9969](https://github.com/Luminous-Dynamics/symthaea/issues/9969).

### 14.1 SRE-Bench: realistic agentic reverse engineering

SRE-Bench (updated October 2026) is the strongest external reference identified for general agentic binary reverse engineering. Its published snapshot contains 19 private programs, 262 binary instances and 1,572 deterministically graded tasks. The five domains are firmware, malware, games, file formats and network protocols; it includes four implementation languages and 44 anti-analysis mechanisms. It isolates each task in a container, withholds source and reference solutions, supplies a standard RE toolkit, and grades task outputs deterministically. Its per-binary score is the number of tasks passed out of six, with fully solved and zero-credit instance counts reported as well. [SRE-Bench methodology and snapshot](https://daplab.cs.columbia.edu/benchmarks/sre-bench/)

**What this means for us:** build an official evaluation adapter if the benchmark offers authorized access. Do not mirror private artifacts, reverse engineer the grader, or claim an SRE-Bench result from a locally reconstructed subset. Report eligible, attempted, completed and graded tasks separately, together with refusal/abstention, cost and resource use. The public leaderboard is a dated snapshot; model and configuration differences must be disclosed. A SRE-Bench result is not a mobile-specific result.

### 14.2 OWASP MAS Crackmes: public Android and iOS evaluation

Use OWASP MAS Crackmes for reproducible mobile-specific tasks: Android UnCrackable Levels 1–4, the DRM and License Validator challenges, and iOS UnCrackable Levels 1–2. The collection is used in the OWASP Mobile Application Security Testing Guide and includes both Android and iOS examples. [OWASP MAS Crackmes](https://mas.owasp.org/crackmes/)

Public challenges are useful for reproducibility but are vulnerable to solution contamination because challenge descriptions and community solutions are public. Preserve the upstream release, input digest, license and oracle version; add newly built, source-known variants with randomized secrets, library versions, architecture, and defensive transformations as a held-out cohort. Never call the public cohort contamination-free.

### 14.3 Compare tools by the task they are designed to perform

| System | Established role | Fair comparison |
|---|---|---|
| Ghidra | General reverse-engineering workbench: disassembly, decompilation, graphing and scripting | Function recovery, call/data-flow localization, cross-architecture coverage and analyst-assisted task completion |
| MobSF | Mobile application security platform with static and dynamic analysis for Android, iOS and Windows Mobile | Manifest/resource/security checks, vulnerability findings, runtime observations and integration cost |
| JADX | Android DEX-to-Java decompilation | Decompiled-code recovery, class/method localization and semantic parity against source-built variants |
| apktool | Android resource and manifest decode/rebuild workflow | Resource/manifest fidelity and round-trip behavior |
| REA | Structured artifact inventory and provenance-bearing Evidence contract | Inventory correctness, coverage, evidence identity, schema and provenance integrity |
| Symthaea | Cognitive orchestration, evidence-grounded hypothesis generation and next-experiment selection | Diagnosis lift, calibration, abstention, information gain and investigation time/cost |

Do not compare outputs with different task definitions as one aggregate score. Measure Ghidra on code-understanding tasks and MobSF on its static/dynamic mobile security scope; measure REA on artifact/evidence contracts; measure Symthaea on the extra decision quality it contributes. Then compare the composed stack with every constituent arm to test whether integration helps or merely adds latency.

The official MobSF project describes an automated mobile security framework with both static and dynamic analysis; the Ghidra project describes a multi-platform software reverse-engineering suite including disassembly, decompilation, graphing and scripting. [MobSF project](https://github.com/MobSF/Mobile-Security-Framework-MobSF) · [Ghidra project](https://github.com/NationalSecurityAgency/ghidra)

### 14.4 Required ablations and baselines

For each pinned APK/IPA cohort, evaluate:
- REA alone;
- Symthaea without REA-derived analysis;
- REA + Symthaea;
- specialist tools alone (MobSF, Ghidra, JADX, apktool where applicable);
- specialist stack + REA + Symthaea;
- human expert as a separately budgeted reference.

All automated arms receive the same binary, auxiliary material, CPU/RAM/time/network/tool budgets and grader. Separate public and held-out splits. Blind the graders. Freeze task oracles before evaluating. Store the exact model, prompts, tool versions, configuration, input hashes, evidence IDs, environment and result receipt.

For the public SRE-Bench snapshot, report:
- mean passed tasks per binary out of six;
- fully solved and zero-credit instances;
- graded/eligible coverage and refusal rate;
- cost per passed task, latency and resource consumption.

For mobile challenge cohorts, report:
- deterministic challenge pass rate and complete challenge solve rate;
- class/function top-1/top-3 and MRR;
- evidence-backed claim precision/recall and false-cause rate;
- calibration and selective risk versus coverage;
- time and cost to a correct finding;
- runtime observations, package coverage, crash/resource cost and energy where measured.

### 14.5 What "better than SOTA" means here

There is no defensible single "reverse engineering SOTA" number. SRE-Bench is a general, five-domain agentic RE benchmark; OWASP MAS Crackmes are mobile-centric educational challenges; Ghidra and MobSF have materially different products and task surfaces. Do not pool the scores or compare unaligned outputs.

The first target is not to claim the best decompiler. It is to show that evidence-bound orchestration improves the **quality of decisions**:
- >=20% relative top-3 localization gain versus the stronger single-system baseline, without higher false-cause rate;
- >=15% lower median time-to-correct-diagnosis as a stretch target, without >5% higher investigation cost;
- Brier/ECE and selective risk that improve or at least do not degrade as coverage increases;
- zero confident causal claims on seeded unobservable and ambiguous negative controls;
- 100% accepted causal claims linked to exact source Evidence and intervention records;
- explicit coverage, refusals, timeouts and unsupported cases in all denominators.

These are proposed gates, not measured results and not universal industry standards. The benchmark manifest includes separate families for the SRE-Bench external track and OWASP public mobile challenges.

#### Published SRE-Bench public-facing leaderboard snapshot

At the October 2026 snapshot, the public-facing results table reports GPT-6-Astra at 5.55 tasks passed per graded binary, with 149 fully solved of 181 graded (82.3%) and 7 zero-credit (3.9%). The same table reports GPT-5.6-Sol at 3.69 tasks per graded binary, with 80 fully solved of 254 graded (31.5%). These are different graded-instance counts, so quoting solved percentages alone hides differences in coverage and refusal. The benchmark page explicitly states that GPT-6-Astra was graded on 181 of 262 instances because of cyber-safety refusals.

For our own reporting, keep the official SRE-Bench score unchanged, but add two non-substituting coverage measures:
- **eligible coverage** = attempted / eligible tasks;
- **coverage-adjusted task yield** = passed tasks / all eligible tasks.

Also publish refused, abstained, unsupported, timeout and tool-failure counts separately. Do not relabel the coverage-adjusted measure as the official SRE-Bench score. This is the guard against a system appearing stronger merely because it declines more difficult tasks.


## 14. External benchmark baselines added: SRE-Bench, Decompile-Bench and OWASP MASTG

The reverse-engineering programme now includes [REA-RUST-009 / #10026](https://github.com/Luminous-Dynamics/symthaea/issues/10026).

### SRE-Bench (agentic binary reverse engineering)

The October 2026 benchmark page reports 19 private programs, 262 binary instances and 1,572 graded tasks across firmware, malware, games, file formats and network protocols. It includes C/C++/Rust/Go, 44 in-house anti-analysis protections, isolated-container task environments and deterministic grading. Its public-facing model results include GPT-6-Astra at 5.55/6 mean score and 82.3% fully solved on 181 graded instances; its graded denominator is not the whole corpus.

This is the strongest currently identified external reference for realistic agentic RE. Its program instances are private; access and participation must follow the maintainers' process. The benchmark's exact task, budget, tooling and safety settings must be reproduced before comparing scores. Do not mirror private task data.

### Decompile-Bench (function-level decompilation)

Stanford AIMS lists 72,059 released decompiler outputs from IDA and Ghidra over HumanEval, MBPP and GitHub2025 functions. Its edit-similarity metric measures textual similarity, not executable correctness. The published response matrix reports roughly 0.206 mean for IDA and 0.202 for Ghidra.

Use the released metric as a text-reconstruction baseline only. Add compiled/tested behavior equivalence and semantic graph metrics as distinct outcome dimensions.

### OWASP MASTG / MASVS / MASWE

OWASP explicitly cautions that automated scanners alone cannot complete MASVS verification; contextual architecture and business-logic assessment are necessary. The comparative plan therefore separates scanner results from manual/agentic test evidence. MobSF is a useful automated baseline, but Android root/Frida Gadget/physical configurations and Corellium/jailbroken iOS configurations remain separate cohorts.

### Experimental controls

Use matched, blinded REA-only, Symthaea-only, REA+Symthaea, specialist decompiler and MobSF arms where relevant. Report per-domain, per-platform, per-hardening family and test configuration. Never combine SRE-Bench, Decompile-Bench and mobile security into one headline score.

References:
- SRE-Bench: https://daplab.cs.columbia.edu/benchmarks/sre-bench/
- Decompile-Bench: https://aims.stanford.edu/measurement-db/decompile_bench
- OWASP assessment and automation limitations: https://mas.owasp.org/MASVS/04-Assessment_and_Certification/
- MobSF dynamic analysis requirements: https://github.com/MobSF/docs/blob/master/develop.md


## 14. External benchmark alignment (verified 2026-11-02)

The comparison programme should anchor itself in independently maintained benchmarks instead of inventing one monolithic score.

### SRE-Bench — agentic reverse engineering

The official DAPLab SRE-Bench page reports its October 2026 release state:
- 19 private programs and 262 binary instances;
- 1,572 deterministic tasks, six per binary;
- five domains (firmware, malware, game, file format, network protocol);
- C, C++, Rust and Go targets and 44 in-house anti-analysis mechanisms.

Its public-facing October scoreboard currently reports GPT-6-Astra at 5.55/6 on 181 graded instances, 149 fully solved, and 7 zero-credit instances. The denominator matters: 181 of 262 instances were graded (about 69.1%). A system that reports only its score on graded cases would conceal ungraded/refused instances. The benchmark manifest must publish eligible, attempted, completed, graded, refused, unsupported, timeout and abstained denominators alongside quality.

Do not treat the official leaderboard as a benchmark we can reproduce locally. Use the official access/grader path or request participation. Do not copy or rehost private binaries, reference solutions or grading code. Run our own public/authorized cohort separately.

### Stanford AIMS Decompile-Bench — output quality, not executable correctness

The Stanford AIMS measurement database currently lists 72,059 released function-level outputs from IDA and Ghidra over HumanEval, MBPP and GitHub2025. The published text edit-similarity means are 0.206 for IDA (71,496 observations) and 0.202 for Ghidra (69,635 observations). The authors explicitly describe this metric as textual similarity, not executable correctness.

Therefore the Symthaea evaluation needs two additional tracks:
- reproduce the published metric exactly, with the released CC0 dataset and MIT metric code, keeping each original source project's license;
- independently measure compile rate, test equivalence on held-out vectors, recovered control/data-flow edges, semantic localization and behavior-preserving edit cost.

Do not use a decompiler text-similarity score as a substitute for recovered behavior or vulnerability discovery.

### OWASP MAS Crackmes — public mobile task corpus

The official OWASP collection includes Android UnCrackable L1-L4, Android UnCrackable DRM, Android License Validator, and iOS UnCrackable L1-L2. Use the official artifacts for a reproducible public evaluation, but keep platform and environment cohorts separate. In particular, the iOS challenge notes signing/provisioning and device/jailbreak constraints; an iOS result on a re-signed/jailbroken target is not comparable to stock iPhone behavior.

Keep solution/source material out of hidden variants and run a source-to-binary mapped holdout where the grader is separated from the agent. Record artifact hashes, build and instrumentation scope.

### Revised comparative strategy

1. **Official external scoreboards:** SRE-Bench and Decompile-Bench, under their official protocols.
2. **Public mobile task suite:** OWASP MAS Crackmes and selected MASTG tests.
3. **Private Luminous holdout:** source-derived mobile apps, generated firmware and binary fixtures with deterministic oracles.
4. **Reasoning ablations:** REA, Symthaea, REA + Symthaea, specialist tools, and human analyst comparison on matched tasks/budgets.

Report success quality and denominators separately. A refusal is not a solved task; an unsupported operation is not a pass; an ungraded case is not absent from the report.

References:
- SRE-Bench official page: https://daplab.cs.columbia.edu/benchmarks/sre-bench/
- Decompile-Bench: https://aims.stanford.edu/measurement-db/decompile_bench
- OWASP MAS Crackmes: https://mas.owasp.org/crackmes/
- OWASP Android crackmes: https://mas.owasp.org/crackmes/Android/
- OWASP iOS crackmes: https://mas.owasp.org/crackmes/iOS/


## 15. Implementation snapshot (2026-11-02)

The following work has moved beyond architecture-only status:

- **Evidence identity core:** [PR #9910](https://github.com/Luminous-Dynamics/symthaea/pull/9910), exact dedicated run [#41409682777](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41409682777) passed: 16 Rust unit tests and one upstream-generated differential integration test. General repository CI was skipped; the isolated Cargo harness does not establish root workspace health or full REA schema/bundle parity.
- **Offline verification CLI:** [PR #10058](https://github.com/Luminous-Dynamics/symthaea/pull/10058) adds bounded local file verification, stable JSON/text output and an explicit claim ceiling. Its dedicated exact-head workflow is pending as of this snapshot.
- **Causal experiment planner:** [PR #10074](https://github.com/Luminous-Dynamics/symthaea/pull/10074) adds a pure-Rust Bayesian EIG planner, root-evidence lineage, authorization/reversibility/target-impact/resource budgets and abstention. Its dedicated exact-head workflow is queued as of this snapshot.
- **ZIP metadata reader:** [PR #9965](https://github.com/Luminous-Dynamics/symthaea/pull/9965) separately implements a bounded std-only archive metadata reader and has a pending dedicated workflow. It remains distinct from full ArtifactReader parity.

The Evidence core is the only one of these four implementation tracks with a completed passing dedicated workflow at this snapshot. Do not infer status from issue text or a parent PR; always bind a validation claim to the exact head SHA and completed run.


## 14. Decision quality under likelihood misspecification

The point-EIG experiment planner (#10074) and scenario-robust EIG extension (#10105) are deliberately separate. Robust EIG computes expected, worst-case and lower-tail information gain across a declared shared scenario ensemble. Its worst-case is only over the supplied finite set; no claim is made that the true data-generating process is in that set.

A further paradigm shift is to make the final engineering decision the objective. Harikumar et al., *Robust Bayesian Decision Making under Adversarial Uncertainty*, UAI 2026 / PMLR 337:1983–2007, argues that nominal information gain can lead to high-confidence but fragile downstream decisions under hidden or weakly modeled effects. Their work motivates decision-stability-aware sequential design:
https://proceedings.mlr.press/v337/harikumar26a.html

Track [#10109 — decision-aware robust investigation](https://github.com/Luminous-Dynamics/symthaea/issues/10109) extends the planner with:
- terminal decisions and explicit utility/loss across hypotheses;
- expected value of sample information (EVSI);
- scenario-specific posterior-optimal decisions;
- shared-policy decision regret across plausible scenarios;
- worst-case and lower-tail decision utility/regret;
- abstention for incomplete/unsupported utility scope.

Keep the metrics distinct:
- **EIG** measures uncertainty reduction.
- **EVSI** measures expected improvement in a decision under a declared utility function.
- **Decision regret/stability** measures how badly the chosen policy performs across plausible worlds.

The utility model must not be treated as objective truth. Record who authored it, which stakeholders/consequences it represents, what costs or harms are omitted, and which actions remain outside authorization. No action is executed by the planner.

## 15. Latest external agentic RE baseline: AgentRE-Bench v3.1

Add AgentRE-Bench v3.1 (October 2026) as a second independent, public source-free binary reasoning baseline:
https://www.agentre-bench.ai/

Its published Windows PE suite has 10 programs evaluated with and without symbols, 20 artifacts, five models and five attempts per artifact (500 episodes), and a 25-call budget. It records tool calls, evidence, time, coverage, failures and versioned ground truth; unsupported claims are penalized. The published v3.1 results show the top two overall scores close together (87.37 and 87.04), with 95% intervals and valid-answer counts.

Keep AgentRE-Bench and SRE-Bench separate:
- AgentRE-Bench v3.1 is a 20-artifact Windows PE protocol.
- SRE-Bench spans 19 private programs, 262 binary instances, five reverse-engineering domains, four source languages and 44 anti-analysis mechanisms.
- Their score units, artifact cohorts and budgets are incompatible; do not pool or compare headline percentages.

For the Luminous evaluation, adopt AgentRE's 25-call budget as one constrained tool-use condition, and publish exact per-claim evidence/ground-truth results, invalid answers, abstentions, missing attempts and confidence intervals. A separate unconstrained condition may exist, but must never replace the fixed-budget leaderboard.

Official references:
- AgentRE-Bench: https://www.agentre-bench.ai/
- SRE-Bench owner page: https://daplab.cs.columbia.edu/benchmarks/sre-bench/


## 14. Decision quality is a separate objective from information gain

The investigation planner now has three conceptually distinct layers:
1. Point EIG: rank probes by expected entropy reduction.
2. Scenario-robust EIG: expose expected, worst-case and lower-tail information gain under supplied likelihood models.
3. Decision-aware EVSI/minimax regret: rank probes by expected improvement in a caller-declared terminal engineering decision, and optionally minimize worst-case expected regret of a shared outcome-to-action policy.

The implementation is in [#10171](https://github.com/Luminous-Dynamics/symthaea/pull/10171), stacked on the scenario-robust planner.

### Why this is a genuine paradigm shift

A probe can reveal a large amount of information about a nuisance variable without changing the correct remediation. A different probe can yield less entropy reduction while sharply distinguishing between two costly remediation choices. The benchmark must reward the latter when it improves the engineering decision.

### Exact finite minimax policy

For each experiment, enumerate deterministic policies mapping every declared outcome to one terminal decision. For every supplied scenario, compute the expected posterior decision regret of that shared policy against the scenario-specific posterior-optimal action. Select the policy with minimum maximum scenario regret.

This implementation:
- uses scenario/outcome probabilities to compute expected regret in each supplied scenario;
- uses deterministic lexicographic tie-breaking over decision IDs;
- exposes a hard policy-evaluation cap (default 65,536);
- returns an explicit per-experiment `PolicySearchBudgetExceeded` rejection if the policy space is too large;
- never silently falls back to the mixture-Bayes policy;
- preserves the existing authorization, reversibility, destruction, target-impact, time, energy and risk gates.

The guarantee is exact only over the supplied finite scenario set and declared utility matrix. It is not an adversarial guarantee over unenumerated conditions, and it does not execute experiments or remediations.

### Matched benchmark

Compare point EIG, scenario-robust EIG, expected EVSI, worst-case/CVaR EVSI, EVSI-minus-regret, exact finite minimax regret, random authorized probe and cost-only baselines. Use asymmetric loss, hidden scenario shift, likelihood misspecification, uninformative high-EIG probes, lower-EIG decision-changing probes, unauthorized actions and search-budget overflow.

Measure realized held-out regret, wrong terminal decisions, EVSI prediction error, policy stability, target impact and total investigation cost. Proposed target: reduce 90th-percentile realized regret by at least 20% versus point EIG on a preregistered misspecification corpus, without increasing target impact or investigation cost beyond the declared tolerance. This is a research target, not a measured result.

The machine-readable benchmark manifest now includes the decision-value/minimax-regret suite as a separate family.


### Held-out distribution-shift evaluation

Exact minimax over the supplied finite scenarios can still fail under an omitted true data-generating process. Therefore the benchmark now has a distinct hidden-DGP track tracked by [#10175](https://github.com/Luminous-Dynamics/symthaea/issues/10175).

The hidden track withholds true likelihoods and tests prior shift, omitted latent variables, missing device/build/environment modes, correlated or non-stationary outcomes, optimistic in-ensemble models, utility perturbation, externalities and cost/impact estimation errors. It reports in-ensemble optimality separately from out-of-ensemble realized regret, wrong-action rates and regret CVaR.

The planner is not given held-out scenarios; it cannot optimize directly for them. The benchmark freezes declared priors/scenarios/utilities before revealing the hidden DGP. Proposed target: at least 20% lower p90 realized regret than point EIG on the preregistered misspecification corpus, with no increase in wrong-action rate or target impact. This remains a research goal, not a measured claim.


## 14. Decision-aware experiment selection: optimize downstream engineering value

The Rust investigator now has a decision-aware research branch and additive API tracked by [#10109](https://github.com/Luminous-Dynamics/symthaea/issues/10109). This follows the UAI 2026 work on robust Bayesian decision making under adversarial uncertainty: nominal entropy reduction can be high while downstream decisions remain fragile.

Keep four quantities separate:
- **EIG:** expected reduction in uncertainty about hypotheses.
- **Scenario robust EIG:** sensitivity of that reduction to supplied alternative likelihood models.
- **EVSI:** expected improvement in the best downstream decision under a declared utility model.
- **Decision regret/stability:** how much value is lost when one shared decision policy must work across plausible likelihood scenarios.

The additive decision API takes caller-owned utility matrices with explicit owner, scope, units and exclusions. It evaluates a shared outcome-to-decision policy, reports expected/worst/lower-tail policy EVSI and regret, and can enumerate a minimax-regret policy within an explicit combinatorial budget. Terminal decision impact is reported but never executed or authorized by this planner.

Two research gaps remain independent of the implementation:
- [#10207](https://github.com/Luminous-Dynamics/symthaea/issues/10207): independent likelihood calibration receipts, target-scoped reliability, effective sample size and drift expiry.
- [#10208](https://github.com/Luminous-Dynamics/symthaea/issues/10208): utility-model uncertainty, Pareto dominance, hard non-compensable constraints and preference sensitivity.

The machine-readable benchmark manifest contains a separate decision-aware value-of-information family. It compares point EIG, robust EIG, EVSI, lower-tail policy EVSI and minimax regret, with known-answer and misspecification tests. A target of 20% lower 90th-percentile decision regret versus point EIG is proposed, not measured.


## 16. Industry benchmark update: SRE-Bench denominator and OWASP MASTG knowledge graph

### 16.1 SRE-Bench — the leading public-facing number is conditional

The official October 2026 SRE-Bench snapshot reports:

| Public-facing model | Mean passed tasks / 6 | Graded instances | Fully solved / graded | Zero-credit / graded |
|---|---:|---:|---:|---:|
| GPT-6-Astra | 5.55 | 181 / 262 (69.1%) | 149 / 181 (82.3%) | 7 / 181 (3.9%) |
| GPT-5.6-Sol | 3.69 | 254 / 262 (96.9%) | 80 / 254 (31.5%) | 41 / 254 (16.1%) |
| Claude-Fable-5.1 | 2.58 | 223 / 262 (85.1%) | 60 / 223 (26.9%) | 96 / 223 (43.0%) |
| Claude-Opus-5 | 1.91 | 256 / 262 (97.7%) | 32 / 256 (12.5%) | 118 / 256 (46.1%) |

The leading 82.3% full-solve figure is conditional on 181 graded instances; the graded coverage is 69.1%. The 149 fully solved instances are 56.9% of the full 262-instance corpus. The latter is a derived coverage-adjusted figure, not the benchmark owner's reported full-solve rate.

The official table's setting enables model-side cyber safeguards. A separate unconstrained setting removes safeguards and budget limits. Those are different operating points, not one leaderboard. Preserve the original setting and score while reporting eligible/attempted/completed/graded/refused/abstained/unsupported/timeouts and cost separately. Never optimize the headline by leaving the hard or refused cases out of the denominator.

For the Luminous system, three additional measures should appear alongside the official metric:
- **coverage-adjusted task yield:** passed tasks divided by the eligible task universe;
- **selective risk versus coverage:** error rate as the system changes its abstention threshold;
- **cost-to-correct:** total bounded investigation cost per independently graded correct task.

These are internal derived measures and must not be attributed to SRE-Bench.

### 16.2 OWASP MASTG v2 / MASWE v1 — consume the standard as a graph

The OWASP programme is no longer best treated as a prose checklist. MASTG v2.0.0 (4 July 2026) is the first stable non-beta release of its refactored, machine-addressable guide. The current published index lists 285 tests, 152 demos, 167 techniques, 135 tools, 140 knowledge articles and 72 best practices. MASWE v1.0.0 (17 August 2026) is the first stable, consistently structured and stably numbered weakness taxonomy.

This creates an opportunity for Symthaea to compile a **mobile verification graph**:

```
MASVS control
   -> MASWE weakness
   -> MASTG test
   -> technique / tool / prerequisite
   -> raw observation
   -> versioned evaluation rule
   -> REA Evidence
   -> independently qualified control claim
```

MASTG test records include platform and test type metadata, relate to weakness/profile/knowledge/technique records, and separate raw Observation from Evaluation. Preserve that distinction in our own evidence model. Do not let a model-produced explanation replace raw output or the test's explicit evaluation condition.

The planner should only instantiate tests whose platform and environment prerequisites are satisfied. It must represent prerequisites and outcomes explicitly:
- not tested;
- not applicable, with scope reason;
- unavailable or unsupported;
- inconclusive;
- observed but not evaluated;
- evaluated pass or fail.

No missing test, unsupported platform or unavailable instrumentation may be converted into a pass. A MASTG test pass is not automatically a MASVS control pass: controls may require multiple tests, contextual review, manual testing or business-logic inspection.

The useful Luminous contribution is a compiler that turns pinned OWASP relationships into evidence obligations and an auditable test DAG, then binds every observation/evaluation to artifact digest, target configuration, tool identity and receipt. This should complement OWASP's catalogue, not fork or replace it.

### 16.3 Revised comparative claim

The question is not simply whether Symthaea can beat a scanner or a decompiler. It is whether the REA + Symthaea stack can:
1. preserve or improve independent task correctness on the same benchmark items;
2. increase eligible-case coverage without inflating unsupported certainty;
3. localize correct components and select better discriminating tests;
4. produce higher traceability from the accepted claim to raw observation and evaluation;
5. do so at a measured cost/resource budget.

For mobile verification, report MASTG test coverage, MASWE detection and MASVS control qualification separately. For general reverse engineering, keep SRE-Bench, AgentRE-Bench and Decompile-Bench score units separate.

References:
- SRE-Bench official October 2026 snapshot: https://daplab.cs.columbia.edu/benchmarks/sre-bench/
- MASTG v2.0.0 announcement: https://mas.owasp.org/news/2026/07/04/mastg-v200-release/
- MASTG v2.0.0 release: https://github.com/OWASP/mastg/releases/tag/v2.0.0
- MASWE v1.0.0 announcement: https://mas.owasp.org/news/archive/2026/
- MASWE catalogue: https://mas.owasp.org/MASWE/


## 14. From-scratch firmware reproduction

The public reverse-engineering benchmark suite should include a pre-environment task cohort, tracked by [REA-RUST-016 / #10298](https://github.com/Luminous-Dynamics/symthaea/issues/10298).

ReproBench (published 28 September 2026) identifies a meaningful gap: most security benchmarks start with a prepared source tree, binary or environment. Its study starts from a CVE identifier and evaluates six sequential phases: information gathering, firmware acquisition, extraction, binary/root-cause identification, service rehosting and vulnerability triggering. On its 30-IoT-firmware-CVE, 450-run study, it reports 45.3% simulation substitution and only 5.3% successful real-world reproductions among CVE–model pairs. These are results from that study, not Luminous measurements.

The paradigm shift is **from binary comprehension to verified reconstruction of the target environment**. REA should preserve source provenance and artifact lineage across every phase. Symthaea should reason over the graph, identify where uncertainty blocks progress and choose the next discriminating experiment. Independent phase oracles must decide whether each step really happened.

Do not credit a mock service, host-native substitute or static-only hypothesis as a real firmware reproduction. Start with authorized public/vendor firmware and safe local validation: deterministic crash/marker evidence or patched-vs-unpatched differential tests inside a disposable, network-isolated environment. No third-party live-device test or remote exploitation is part of this initial benchmark.

The six-phase evidence, A/B/C ablation and scorecards are defined in #10298 and the machine-readable benchmark manifest. Keep this separate from SRE-Bench, AgentRE-Bench and Decompile-Bench; they have different task starts and scoring contracts.


## Upstream Evidence-contract drift audit (2026-11-04)

A fresh audit compared the six Evidence identity/schema source files at the pinned reference commit
`8250bf2320ea57aab4e664db4b4d44b8abd2004e` against upstream `main` at
`89317a39fab23df81efa4331311d09d2110377bd` (2026-11-04 12:43:04 UTC).

| Upstream source | Pinned bytes | Current-main bytes | Exact content match |
|---|---:|---:|---|
| `src/domain/evidence.ts` | 18,468 | 18,468 | yes |
| `src/domain/canonicalDigest.ts` | 3,473 | 3,473 | yes |
| `src/domain/analysisProfile.ts` | 2,571 | 2,571 | yes |
| `src/domain/evidenceSubject.ts` | 1,784 | 1,784 | yes |
| `src/domain/relationshipCoverage.ts` | 1,722 | 1,722 | yes |
| `src/domain/evidenceBundle.ts` | 23,568 | 23,568 | yes |

The current upstream package metadata still reports `rea-agents@6.8.0`. The six files above were byte-for-byte identical in this comparison; this is a narrow contract-drift result, **not** a claim that the complete REA repository is unchanged.

### Engineering consequence

- Keep the original pinned SHA as the immutable golden-corpus generator and compatibility oracle.
- Record the current upstream HEAD separately as a drift-audit checkpoint.
- Re-run this exact-file audit before changing the pinned oracle or promoting a native operation.
- If any contract file changes, create a new schema/compatibility review and corpus version; do not silently regenerate old fixtures from new upstream code.

This removes an avoidable source of uncertainty: as of this audit, the six Evidence semantic-contract files do not require a new pin merely because upstream `main` advanced.


## Exact-head implementation status refresh (2026-11-04)

The following dedicated REA workflows were inspected by exact commit SHA. All PRs listed remain open drafts; a passing dedicated workflow is not a merge, review approval or qualification outside its declared scope.

| Work item | Exact tested head | Dedicated workflow | Result | Scope actually tested |
|---|---|---|---|---|
| Rust Evidence core [#10347](https://github.com/Luminous-Dynamics/symthaea/pull/10347) | `8ff5b6cb87f1e4a4a9e5be19053d8fdefcf972d8` | [41784306259](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41784306259) | success | Pinned upstream-generated Evidence corpus and Rust core tests |
| EvidenceBundle verifier [#10353](https://github.com/Luminous-Dynamics/symthaea/pull/10353) | `a4b61fbe43c80cd4b85c2371e9dbede181a17bf3` | [41805927317](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41805927317) | success | Bundle/residual-unknown graph verifier and upstream-generated bundle comparisons |
| Offline CLI [#10348](https://github.com/Luminous-Dynamics/symthaea/pull/10348) | `59f9445fd1d714749f4d6edf5a37d7ce323bddd6` | [41785107146](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41785107146) | success | Isolated offline Evidence verifier CLI |
| Read-only MCP stdio adapter [#10395](https://github.com/Luminous-Dynamics/symthaea/pull/10395) | `7f6ec97a0cee92186e9a7deb5715875a5037dc85` | [41839127100](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41839127100) | success | Pinned upstream fixture generation, formatting, unit/protocol tests and standalone stdio binary build |
| Benchmark receipt validator [#10314](https://github.com/Luminous-Dynamics/symthaea/pull/10314) | `58dd14377870746b3b0fbe1e982391dd2cc918fa` | [41808798130](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41808798130) | success | Dedicated benchmark receipt validation workflow |

### Failure/claim accounting

The #10347 commit also had failures in two other repository workflows (Proof Insight Transfer Indexer V0 and Beverage Benchmark Contract); its dedicated REA Rust Evidence Core workflow passed. These unrelated failures are not hidden or re-labelled as successes, and the table above intentionally reports the dedicated REA lane rather than the aggregate repository status.

The MCP adapter required several exact-head fixes: enabling/declaring the `schemars` feature/dependency; importing EvidenceBundle functions from the public `bundle` module; aligning calls with rmcp 3.8.0's direct `CallToolResult` return type; disabling unintended harness auto-bin discovery; and applying rustfmt. The final exact-head workflow passed all substantive steps.

### Current status boundary

- The Evidence verifier, bundle verifier, CLI and read-only MCP adapter have successful dedicated exact-head workflow evidence on their respective draft branches.
- They are not yet merged together into one release or proven interoperable at every root feature/build combination.
- The MCP result verifies only two content-supplied, read-only tools. It does not qualify the general REA tool catalog, external provider dispatch, producer authenticity, observation truth or mobile-device behavior.
- The provider-dispatch gateway remains separately tracked in [#10398](https://github.com/Luminous-Dynamics/symthaea/issues/10398), with crash/reconciliation requirements composed from #4493, #7068, #7080 and #8769.
- The newly opened [#10410](https://github.com/Luminous-Dynamics/symthaea/issues/10410) adds reverse-engineering-specific provider lineage/common-mode provenance as a consumer of #6283, not a second evidence ledger or trust score.


### Provider-lineage benchmark family added

The machine-readable benchmark manifest now includes a 25th family,
`provider_lineage_and_common_mode_corroboration`, linked to
[REA-RUST-021 / #10410](https://github.com/Luminous-Dynamics/symthaea/issues/10410).
The manifest was fetched after the update and parsed successfully as JSON; it records
`as_of: 2026-11-04` and 25 distinct benchmark families.

This family evaluates seeded shared parser/engine/ruleset/model/acquisition roots,
derived-output relays, unknown ancestry, disagreement preservation and confidence
inflation. It composes with the general measurement-dependence work in #6283 and the
REA causal-investigation benchmark in #9911. It does not alter REA Evidence IDs and
does not produce a scalar provider-independence score.

## 18. 2026-11-04 — streamed ZIP content verification and cross-parser evidence

The metadata-only reader in [#9965](https://github.com/Luminous-Dynamics/symthaea/pull/9965) is being extended by the stacked draft [#10368](https://github.com/Luminous-Dynamics/symthaea/pull/10368). The new API targets a single central-directory occurrence and keeps the custom parser as the policy authority.

The draft uses `zipcore = 0.10.2` as a bounded Stored/Deflate streaming verifier and `sha2 = 0.10.9` for incremental SHA-256. Before consuming bytes, it compares archive count and central-directory location, ZIP64 status, central-directory occurrence index, raw filename bytes, raw local-header offset corrected by the known leading stub, general-purpose flags, method, declared compressed and uncompressed sizes, CRC, data-descriptor bit, encryption state and member kind. The decoder's own offset correction is therefore not allowed to silently substitute a different member.

A member qualifies only after the stream reaches EOF and the decoder accepts CRC and declared-size checks. The caller budget is capped by the already-admitted declared uncompressed size. Encrypted entries, directories, symlinks and methods outside Stored/Deflate are explicit unsupported cases. No extraction, output file, password, decryption or execution path is added. The held source descriptor and selected path are revalidated after the attempt. The reported compressed length remains declared metadata.

This is a per-member result, not archive completeness. Unlisted local-header scanning, APK Signing Block semantics, package-signature checks and remaining REA traversal semantics remain separate claims. The draft adds fixtures for duplicate names, CRC corruption, data descriptors, ZIP64 with a prefix, empty entries, Deflate, size-budget failure, oversized actual output, encrypted/unsupported methods and source mutation. These fixtures are authored; they are not considered passing until the exact-head workflow finishes.

### Paradigm shift: parser disagreement is evidence

The archive benchmark family in the machine-readable manifest now binds the implementation to #10368 and adds explicit cross-parser disagreement, source-race and output-bound controls. The custom reader, zipcore and pinned upstream REA implementation are separate adjudication arms, not a voting committee. A disagreement is preserved with exact source/version/feature/lockfile lineage and a minimized fixture. The zip-rs 9.3.0 candidate remains a separate possible oracle; shared dependency ancestry must be checked under [#10410](https://github.com/Luminous-Dynamics/symthaea/issues/10410) before calling two decoder results independent.

The benchmark should report correctness before speed:
- false accepts and false rejects by corruption/race family;
- occurrence mapping and exact metadata parity, including duplicate raw names;
- CRC and size failure recall, partial-stream false-accept rate, and source-mutation false-accept rate;
- output budget adherence, buffer size and measured peak RSS versus member size;
- only after these gates, throughput and energy per verified MiB, split by Stored and Deflate.

No throughput advantage, production suitability or cross-implementation parity is claimed yet. The generated dependency lock must be committed and the exact-head formatting, tests and Clippy workflow must pass before the draft can be promoted.

### Concurrency review addendum

A source-level review found that `File::try_clone()` may share an underlying file cursor on Unix. Sharing the metadata reader's descriptor with parallel content-verification calls would therefore introduce a seek/read race even when each individual stream is correct. The draft implementation now opens independent decoder and identity-witness descriptors per verification call, checks both against the admitted source identity before parsing, and revalidates the identity witness plus the currently selected path after the attempt. A concurrent same-reader, eight-worker fixture has been authored. This adds a distinct benchmark control for cross-call cursor interference; it is not yet a passing test result.

A deterministic 256-sample corruption smoke test has also been added to #10368: each sample applies three reproducible bit flips to a frozen minimal archive, then exercises inventory admission and content verification when admitted under panic capture. The manifest records the seed, mutation schedule, panic count and structured rejection rate as required reporting. This is a smoke test only; coverage-guided fuzzing, minimized regression fixtures and independent oracle comparisons remain promotion gates.

