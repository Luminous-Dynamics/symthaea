# Multilingual Coding Capability Roadmap: Modern Engineering and Legacy Mainframes

**Status:** research proposal; no new language capability or benchmark result is claimed by this document.  
**Date:** 2026-10-24

## Executive recommendation

Yes: Symthaea should expand beyond its current Rust-first, Rust/Python code-generation lane. The next step is not to add many language names to a prompt. It is to establish first-class language adapters that connect source understanding, edit planning, build/test execution, runtime isolation, and independently auditable evidence. A language is only “supported” at the level for which those parts have been demonstrated.

The most strategically important gaps are:

1. **SQL and schema migrations**, including transaction behavior, data constraints, query plans, and reversible migrations.
2. **TypeScript/JavaScript and the web/tooling formats around them**, for mixed-language applications.
3. **POSIX shell, YAML, Dockerfiles, TOML, Nix, and infrastructure definitions**, because production changes often fail in orchestration rather than the main source file.
4. **C/C++**, followed by Go and Java/Kotlin according to target repositories.
5. **COBOL plus the mainframe job/data environment** for legacy banking work.
6. **Assembly as architecture- and dialect-specific work**, including IBM HLASM for z/Architecture—not one generic “assembly” checkbox.

The repository's code-ability plan describes native generation patterns primarily for Rust and Python and identifies tree-sitter Rust/Python grammars for code generation; the root manifest also includes tree-sitter Nix. The inspected executor has real-execution allowlisting centered on Rust/Cargo, Python, and Nix evaluation. An LLM fallback may emit other languages, but that is not evidence of a fully verified language lane. Treat this as an initial repository inspection, not a claim that no other language-related code exists anywhere in the workspace.

## Capability tiers

| Priority | Language / artifact family | Why it matters | First acceptance gate |
|---|---|---|---|
| P0 | Rust | Current primary implementation and repair benchmark | Keep exact-head compile, unit, integration, mutation, and repository-repair evidence |
| P0 | Python | Existing generation and common automation/test tooling | Real interpreter tests; type/lint checks where configured; no “simulation = pass” |
| P1 | SQL: PostgreSQL and SQLite first; add target-specific dialects | Data integrity, application persistence, schema migration, reconciliation | Execute migrations against disposable databases; test upgrade, downgrade/rollback, constraints, and data invariants |
| P1 | TypeScript / JavaScript | Web UI, APIs, front-end and modern tooling ecosystems | Pinned package manager/runtime; typecheck, lint, unit tests, build, and browser/integration checks as applicable |
| P1 | POSIX shell / Bash, YAML, JSON/JSONL, TOML, Dockerfile, Make/CMake, Nix, HCL/Terraform | CI, deployment, build graphs, runtime configuration and infrastructure are common change surfaces | Parse/lint plus sandboxed validation of the actual configured toolchain; protect secrets and prohibit unapproved egress |
| P2 | C and C++ | Native libraries, systems software, embedded code and FFI | Pinned compiler, sanitizer-enabled tests where feasible, warnings-as-errors lane, reproducible build |
| P2 | Go and Java/Kotlin | Common backend, platform and enterprise repository ecosystems | Native formatter/compiler/test runner and repository-level repair tasks |
| P2 specialist | COBOL, JCL, copybooks, embedded DB2 SQL, CICS/IMS interfaces, PL/I | Mainframe applications are a program/data/job system, not isolated COBOL files | Dialect-aware parsing plus target-compatible builds and differential business-output tests |
| P2 specialist | IBM HLASM / z/Architecture; later x86-64, AArch64, and RISC-V assembly as needed | Mainframe assist routines and architecture-specific routines require instruction-level reasoning | Architecture/dialect-labelled disassembly, assembler diagnostics/listings, calling-convention analysis, emulator or target execution evidence |
| Demand-driven | C#, PHP, Ruby, Perl, Lua, Kotlin, Solidity, Fortran, Ada/SPARK, Verilog/VHDL, etc. | Valuable in their application domains; adding all at once dilutes validation effort | Prioritize from target-repository inventory and demand, then add a real adapter and held-out tasks |

This is a prioritization proposal, not a declaration that all listed languages have equal value. Use a versioned support manifest that distinguishes parse-only, edit-and-lint, buildable, tested, and repository-repair-evaluated. A language must not move up a tier on the basis of syntax highlighting or plausible-looking generated code alone.

## Banking migrations: COBOL is necessary, not sufficient

**COBOL should be a strategic specialization.** It is worth supporting both analysis/repair of existing programs and controlled modernization. However, translating COBOL line-by-line into Rust, Java, or C# is not a safe migration method. IBM's modernization guidance describes multiple approaches—including encapsulation, DevOps, refactoring, and migration—and stresses functional validation. See [IBM: What is COBOL modernization?](https://www.ibm.com/think/topics/cobol-modernization).

A credible mainframe adapter must understand the surrounding artifacts and runtime contracts:

- **COBOL dialect and source format:** fixed/free format, compiler options, version-specific language behavior, COPY/REPLACING and nested copybooks, compiler directives, condition names, and data declarations.
- **JCL:** job steps, EXEC/DD statements, procedures, conditions, datasets, return codes, restart/re-run behavior, and batch scheduling dependencies. IBM describes JCL as the control language that tells z/OS which work to execute and which inputs, outputs, and resources it needs ([IBM JCL overview](https://www.ibm.com/docs/en/zos-basic-skills?topic=sdsf-what-is-jcl)).
- **Database and transaction surfaces:** embedded Db2 SQL, SQLCA/error handling, commit/rollback boundaries, isolation, locks, cursors, CICS/IMS integration, and external interfaces.
- **Data representation:** EBCDIC and code-page conversions, record layouts, packed decimal (COMP-3), zoned decimal, binary fields, signedness, truncation, rounding, dates, collation, and exact numeric behavior.
- **Cross-language dependencies:** PL/I, HLASM, callable routines, macros, linkage conventions, copybooks, shared file layouts and batch jobs. IBM publishes distinct references for Enterprise COBOL, PL/I, HLASM, Db2, CICS and IMS because they are distinct technical surfaces ([IBM z/OS language references](https://www.ibm.com/docs/en/developer-for-zos/17.0.x?topic=reference-language-documents)).

### A safe migration workflow

1. **Inventory before editing.** Build a dependency graph across programs, copybooks, JCL, datasets/file layouts, SQL tables, transaction entry points, scheduler dependencies and external interfaces. Record versions, compiler flags, data encodings and batch control totals.
2. **Characterize the incumbent.** Capture legally authorized, de-identified test fixtures and reproducible outputs from the actual source system. Include edge cases for money, rounding, negative values, empty/invalid records, dates, duplicate transactions, restart points and failures.
3. **Establish a source-compatible baseline.** Use the target's actual compiler/runtime and DB environment where available. GnuCOBOL can provide an accessible development/test lane, but a green GnuCOBOL run must not be presented as proof of equivalence to a specific IBM Enterprise COBOL compiler/runtime or production environment. Record dialect and compiler identity.
4. **Translate incrementally and make assumptions explicit.** Preserve decimal semantics, record schemas, sorting/collation, error paths, transaction boundaries, ordering, and restart/idempotency behavior. Prefer wrapping or strangling one bounded function at a time when a complete rewrite is not justified.
5. **Differentially verify.** Run old and candidate implementations against the same sanitized inputs; compare every output record and field, database effects, side effects, return codes, balances and control totals. Produce an explicit mismatch report. Use property-based tests and generated boundary values in addition to golden examples.
6. **Rehearse production controls.** Shadow-run without live writes, reconcile outputs, verify rollback and restart paths, stage cutover, define accountable human sign-off, and retain an auditable recovery plan. Never test an agent-generated patch against live banking systems or real customer data.
7. **Publish an evidence package.** Pin the source revision, build image/compiler, copybooks and job inputs; hash inputs/outputs/logs; keep unresolved cases unresolved; and independently re-run the verifier. Do not let a syntactically valid conversion count as a behaviorally equivalent migration.

## What about assembly?

Absolutely include assembly—but **identify the ISA and assembler dialect**. For bank migrations the relevant specialist is often IBM High Level Assembler (HLASM) on z/Architecture, not x86 NASM or GNU assembler. IBM notes that z/OS assembler is used when byte/bit-level control or routines not conveniently provided by higher-level languages are required ([IBM: The assembler language on z/OS](https://www.ibm.com/docs/en/zos-basic-skills?topic=zos-assembler-language); [HLASM language reference](https://www.ibm.com/docs/en/hla-and-tf/1.6.0?topic=reference-abstract-language)).

Recommended order:

- **Phase A — read and trace:** parse source and listings; identify instructions, labels, registers, condition codes, storage layout, branch targets, macros, linkage conventions, save/restore behavior, and calls from COBOL/PL/I.
- **Phase B — cross-reference:** correlate source to assembler listings and available disassembly; make control/data-flow explanations evidence-backed, and label unknown macros or unresolved external calls.
- **Phase C — constrained repair:** handle only small, bounded changes with assembler/toolchain diagnostics, tests, and review by a mainframe specialist.
- **Phase D — conversion assistance:** translate selected routines only with differential tests covering register state, condition codes, byte layout, decimal arithmetic, calling conventions, and externally visible behavior.

Avoid claiming assembly support based only on an LLM's ability to emit plausible mnemonics. Assembly generation should fail closed if the target architecture, syntax dialect, assembler version, macro library, or link environment is unknown.

## Adapter architecture: make language support auditable

Implement a Rust-native LanguageAdapter contract rather than a growing language-name match scattered across generation and execution. Each registered adapter should provide, as applicable:

- language ID, version/dialect, file extensions, source-format rules and multi-file/project detection;
- parser/CST/AST and diagnostics, including explicit unsupported/parse-failed states;
- safe context extraction, symbol/call/type/data-flow references, and a planned edit scope;
- formatter/linter and build/test command definitions with pinned tool identities;
- an isolated execution profile (network, filesystem, secrets, time/resource limits, allowed binaries);
- native failure parsing and repair hints;
- evidence schema: source and patch hashes, toolchain/container IDs, command and exit status, test inventory/results, raw-log hashes and unresolved reasons;
- golden/negative fixtures, mutation tests and held-out repository repairs.

The common evaluator must distinguish at least pass, fail, and unresolved. Missing compilers, skipped tests, unknown dialects, simulated execution, nonzero unexplained exits and absent evidence are not passes. Configuration files should have schema/plan validation rather than being treated as general-purpose executable programs.

## Benchmark plan

Public leaderboard scores are useful context but not interchangeable across task sets. SWE-bench Multilingual provides 300 repository-repair tasks across nine languages (C, C++, Go, Java, JavaScript, TypeScript, PHP, Ruby and Rust); it is a useful breadth baseline but does not cover banking mainframes ([SWE-bench Multilingual](https://www.swebench.com/multilingual)). Factory's Legacy-Bench explicitly evaluates legacy engineering tasks across COBOL, Java 7, BASIC, C89, Fortran and Assembly ([Legacy-Bench methodology and current results](https://docs.factory.com/benchmarks/legacy-bench)). Treat all public tasks as potentially exposed and maintain a private held-out set.

Suggested staged gates (all are **proposed internal targets**, not achieved results or external standards):

1. **Inventory gate:** report language-by-language parser, generator, executor, test and evidence coverage. No language is called verified without an actual toolchain run.
2. **Adapter conformance gate:** for each language, require valid and invalid fixtures, parse/diagnostic tests, formatting/linting where appropriate, compile/run tests where the toolchain exists, timeout and sandbox tests, malformed-evidence tests, and negative controls.
3. **Repository repair baseline:** use pinned public multi-language/legacy tasks where their licenses and setup permit; report exact revision, attempt budget, task count, unresolved count, cost and latency. Never compare a small easy slice to a benchmark headline.
4. **Private generalization:** add held-out tasks drawn from independent repositories and organizations. Split by repository and task family, not random near-duplicate files. Pre-register a one-attempt primary score and preserve all failures.
5. **Migration-specific acceptance:** for each COBOL-to-target migration, require a target-compatible build plus differential equality of business outputs and data invariants on a representative, de-identified corpus. Report semantic mismatches, not only compilation rate.
6. **Causal Symthaea benefit:** matched randomized adapter-assisted vs baseline-agent experiments, with model, tools, budgets, environment and tasks held fixed. Report uncertainty intervals and cost per independently verified fix.

Initial work should be narrow: implement the support manifest and adapter contract, then prototype **SQL + TypeScript/shell configuration** as mixed-repository lanes; in parallel start a read-only **COBOL/JCL/copybook analyzer** and HLASM reader. Do not attempt automatic production migration as the first milestone.

## Research references

- IBM, [What is COBOL modernization?](https://www.ibm.com/think/topics/cobol-modernization) (updated 2026-04-06).
- IBM, [Enterprise COBOL for z/OS documentation library](https://www.ibm.com/support/pages/enterprise-cobol-zos-documentation-library).
- IBM, [JCL overview](https://www.ibm.com/docs/en/zos-basic-skills?topic=sdsf-what-is-jcl).
- IBM, [z/OS assembler language](https://www.ibm.com/docs/en/zos-basic-skills?topic=zos-assembler-language) and [HLASM Language Reference](https://www.ibm.com/docs/en/hla-and-tf/1.6.0?topic=reference-abstract-language).
- SWE-bench, [Multilingual benchmark](https://www.swebench.com/multilingual).
- Factory, [Legacy-Bench](https://docs.factory.com/benchmarks/legacy-bench).
- Snorkel, [Agentic Coding Benchmark](https://snorkel.ai/blog/introducing-the-snorkel-agentic-coding-benchmark/) (also includes Assembly and COBOL among its languages/artifacts).
