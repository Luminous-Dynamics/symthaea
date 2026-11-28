# NixWard typed Nix expression IR v0.1

**Status:** implementation prototype; exact-head CI pending  
**Related:** [#14241 — generator integrity](https://github.com/Luminous-Dynamics/symthaea/issues/14241), [#14235 — target option provenance](https://github.com/Luminous-Dynamics/symthaea/issues/14235)  
**Claim ceiling:** typed construction and rnix syntax validation only. No target option resolution, package existence, Nix evaluation, build, activation or runtime correctness is established.

## 1. Why this increment exists

The installer generator currently concatenates attribute paths, values and migrated package strings into Nix source. A better model is a small typed intermediate representation (IR) that cannot represent arbitrary Nix code.

This is deliberately not a general Nix AST editor. It is a restricted construction API for the subset of Nix expressions needed by configuration generation. The IR can represent:

- booleans, integers, strings and null;
- lists and static attribute sets;
- static validated attribute paths;
- package references below the module's pkgs argument.

It has no raw-source expression variant.

## 2. Trust boundaries

| Input | Allowed representation | Not allowed |
|---|---|---|
| Hostname, timezone, description, labels | Escaped Nix string literal | Interpolation or raw source |
| NixOS option path | Restricted static identifier segments | Dynamic attrs, shell text, semicolon or expression fragments |
| Package selection | Static package attribute below pkgs | Arbitrary expression or unvalidated package name |
| Boolean / integer / null | Typed literal | User-supplied serialized Nix |
| Lists / attribute sets | Recursively typed values and validated keys | Raw Nix snippets |

String escaping follows the Nix language's double-quoted string rules: quotes, backslashes, and the dollar-brace interpolation opener are escaped; newlines, carriage returns and tabs use named escapes. Other control characters are rejected rather than copied into the output.

The static attribute grammar is intentionally narrower than the full Nix grammar. This is a safety boundary, not a claim that every valid NixOS option name is supported by this builder.

## 3. Deterministic plan behavior

NixModulePlan stores assignments in a BTreeMap, so output order is stable across runs. It rejects:

- duplicate assignment paths;
- parent/child assignment overlap in one plan;
- invalid identifiers and malformed dotted paths;
- package paths that include the pkgs prefix twice;
- unsupported control characters in string values.

After rendering, the module is parsed by rnix. The parser is used as a syntax gate, not as a semantic evaluator.

## 4. Tests

Unit tests cover:
- quote, backslash, interpolation, newline and tab escaping;
- rejection of unsupported control characters;
- malicious attribute and package path inputs;
- deterministic output and rnix parseability;
- duplicate and overlapping assignments;
- typed nested values;
- property-based arbitrary string generation, requiring every accepted escaped literal to parse.

These tests do not establish that Nix evaluation returns the intended values. The next qualification increment should evaluate representative literals with a pinned Nix evaluator and compare their evaluated values against the original Rust strings.

## 5. What this prototype does not yet solve

- It does not establish that a NixOS option exists in the selected target configuration.
- It does not establish the option's type, read-only status, module namespace or release availability.
- It does not establish that a package attribute exists in the exact locked nixpkgs package set.
- It does not understand module merges, option definitions, mkIf, priorities, overrides or submodule composition.
- It does not evaluate a flake, build a system closure, boot a VM or prove runtime behavior.
- It is not yet wired into SovereignConfigGenerator.

Those gaps are intentional. The next layers must be separate contracts:

1. target-bound option and package resolver;
2. typed goal/constraint plan;
3. typed module construction using this IR;
4. exact target flake evaluation and build;
5. VM/runtime postconditions;
6. independent approval and existing transaction/effect-receipt path.

## 6. Benchmark plan

Do not compare only generated-source parse success. A syntax-only generator could emit a perfectly parseable configuration that is semantically wrong or unsafe.

The next benchmark should compare:
- current hard-coded string generator;
- safe typed IR with static package candidates;
- typed IR plus exact target package/option resolution;
- full planner plus evaluation, build and VM/runtime gates;
- native NixOS operator workflow and strong general coding-agent baselines.

Separate metrics:

| Metric | Meaning |
|---|---|
| Injection acceptance | Invalid user strings or paths become executable Nix — hard target: zero |
| Literal fidelity | Evaluated Nix string equals the original accepted input |
| Syntax validity | Rendered source parses under pinned rnix |
| Target resolution precision | Emitted options/packages exist and apply to the exact target |
| Evaluation/build success | Nix accepts the selected target configuration and builds required outputs |
| Runtime postconditions | Booted VM/hardware satisfies declared outcome |
| Constraint preservation | No forbidden mutation, exposure or recovery regression |
| Benign completion | Valid user goals are completed rather than refused |
| Cost | Latency, Nix evaluation/build time, resources and all-in cost per qualified outcome |

Use mutation/property-based tests for unbounded string and identifier inputs, but retain a frozen, independently graded benchmark with held-out target flakes and version changes. Report corpus digest, runner environment, Nixpkgs revision, exact commit, denominators, confidence intervals and quarantined evaluators.

## 7. External comparisons

- [NixBench 3.2](https://nixbench.com/) measures real Nix problem solving including silent runtime defects and three NixOS VM-graded tasks. Its 21-task corpus has five configurations with three trials each; its reported top configurations reached 52/63 (82.5%) on the 3.2 corpus. It also documents quarantined evaluator disputes, which is a reminder to audit evaluator validity before drawing SOTA conclusions.
- [Nixi architecture](https://docs.nixi.sh/architecture/) separates agent, LLM client, tool registry/executor and UI; it supports local/SSH executors and monitored read-only nodes. NixWard should learn from this operational UX while making exact target binding, typed change construction, evidence freshness and postcondition proof first-class.
- [Colmena](https://colmena.cli.rs/) provides parallel multi-host evaluation/build/deployment. NixWard should compose with established deployment mechanics rather than reinvent reliable closure transport and host orchestration.
- [deploy-rs](https://github.com/serokell/deploy-rs) provides profile activation and a probe/rollback mechanism. NixWard should treat this as a deployment comparator and still independently verify the actual application/service outcome.
- [nixos-anywhere](https://github.com/nix-community/nixos-anywhere) and [Disko](https://github.com/nix-community/disko) address remote installation and declarative disk layout. These are useful backend capabilities, but destructive installation requires exact device binding, independent backup/restore evidence and explicit user authorization.

## 8. Immediate follow-on

Wire the typed IR into a small, non-destructive path first: an approved development-shell or user-scoped package plan. Do not start with disk partitioning, encryption conversion, SSH exposure or bootloader changes. Add the target package resolver and target option resolver before any generated path can be called authoritative.

The existing NixWard action authority, one-shot permits and effect-receipt owners remain the only execution path.
