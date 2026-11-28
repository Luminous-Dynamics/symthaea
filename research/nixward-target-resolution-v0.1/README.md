# NixWard target-bound option/package resolution v0.1

**Status:** deterministic resolver prototype; exact-head Rust CI pending  
**Parent implementation:** [PR #14266 — typed Nix expression IR](https://github.com/Luminous-Dynamics/symthaea/pull/14266)  
**Related:** [#14235 — option catalog provenance](https://github.com/Luminous-Dynamics/symthaea/issues/14235), [#14241 — generator integrity](https://github.com/Luminous-Dynamics/symthaea/issues/14241), [#14268 — benchmark qualification](https://github.com/Luminous-Dynamics/symthaea/issues/14268)  
**Claim ceiling:** deterministic validation of a supplied catalog contract. It does not itself execute Nix evaluation, authenticate a catalog producer, prove target catalog correctness, build a derivation or qualify runtime behavior.

## 1. Why a second validation stage is necessary

Semantic retrieval and syntax-safe generation solve different problems:

- HDC retrieval can suggest likely option paths or package attributes from a user goal.
- Typed Nix construction ensures untrusted strings and values cannot be emitted as arbitrary source.
- A target resolver must establish that each path exists, is writable, has a compatible type and belongs to the exact host configuration and package universe.

A plausible path, a high semantic score and a parseable Nix expression do not establish target applicability.

## 2. Target identity contract

Each option/package catalog is bound to all of:

- host / installation identity;
- NixOS release;
- Nix system platform;
- flake source reference;
- exact nixpkgs revision;
- flake.lock content digest;
- selected nixosConfigurations attribute;
- evaluated module / overlay set digest.

A mismatch in any field rejects the catalog for the requested target. An incomplete field rejects catalog construction.

A release label alone is insufficient because overlays, third-party modules, flake follows and host-specific modules can change the available option and package universe.

## 3. Supported type resolution

The prototype models a deliberately restricted subset:

- bool;
- int;
- string;
- package;
- listOf package;
- listOf string;
- attribute set (without nested submodule schema checking).

Unknown and unsupported types fail closed. More types can be added only with explicit expression/value semantics and conformance tests.

A read_only value of true rejects a write. An unknown read_only value is also rejected because the catalog did not establish that the option is writable.

Package references are checked against a package-attribute set associated with the exact target catalog. Syntax-valid but absent attributes are rejected.

## 4. Catalog provenance and trust boundary

Catalog origins are represented separately:

- TargetEvaluated;
- PinnedReference;
- ControllerLocalLive;
- BundledReference.

Only TargetEvaluated may validate a mutation. The other origins can inform discovery but are not evidence of target membership.

**Important:** TargetEvaluated is a caller-supplied provenance assertion, not a cryptographic attestation. NixOptionCatalogV1::new does not run Nix or authenticate the adapter. The future adapter must:
1. acquire exact pinned sources;
2. evaluate the exact target flake and selected configuration in the untrusted-evaluation boundary from #14270;
3. derive option and package records from those results;
4. bind the source/lock/module digests to the same evaluation;
5. expose a catalog only after checking that the recorded identity matches the actual evaluation request.

This separation prevents the resolver from silently promoting a bundled or local catalog, but it does not replace trust in the adapter.

## 5. Resolver behavior

NixModulePlan::validate_against (through NixOptionCatalogV1::validate_plan) checks:

- requested target identity is complete;
- catalog origin is TargetEvaluated;
- target identities match exactly;
- every planned option exists;
- every option is explicitly writable;
- the typed expression matches the supported option type;
- every package reference is present in the target package set.

A successful result is a ValidatedNixModule. It can then be rendered and checked with rnix. The type name represents catalog-contract validation only; it does not imply a cryptographically verified catalog, Nix evaluation, buildability or runtime success.

HDC and language-model output should never receive a direct path to bypass this resolver.

## 6. Tests

Synthetic tests cover:
- exact target success;
- flake.lock mismatch;
- module-set digest mismatch;
- non-target catalog origin;
- incomplete target identity;
- missing option;
- read-only option;
- unknown writeability;
- type mismatch;
- package absence;
- successful package membership;
- duplicate catalog option registration, ensuring the original record is not overwritten.

## 7. Next steps

1. Implement a target evaluation adapter that runs in the isolated evaluator boundary of #14270.
2. Produce option metadata from the selected evaluated NixOS module set, preserving readOnly, type, declarations and location.
3. Produce package membership from the exact locked target pkgs set, not a controller or bundled package universe.
4. Add Home Manager as a separate namespace/evaluation identity rather than mixing it into NixOS option paths.
5. Expand type support only when submodule, enum, path, attrs, list and package semantics are represented.
6. Wire the resolver into a non-destructive path in SovereignConfigGenerator.
7. Add held-out multi-release/overlay/remote-host cases to #14268.

## 8. Industry comparison

- NixBench 3.2 demonstrates why evaluation-only success is insufficient: its 21-task benchmark includes silent runtime defects and three VM-graded tasks. Our resolver is a prerequisite layer, not a replacement for runtime qualification.
- Nixi exposes a broad conversational NixOS toolset with local/SSH executors and managed/monitored nodes. NixWard should provide stronger target-source identity and validation before a tool call can propose a mutation.
- Colmena and deploy-rs provide established deployment mechanics. NixWard should compose with them after a candidate has passed target resolution, evaluation/build and independent tests rather than inventing a deployment backend.
- nixos-anywhere and Disko can support remote install/disk orchestration, but destructive paths require a separately bound device identity and verified backup/restore evidence.

The differentiator is not merely finding the correct option. It is showing that the option exists in this exact target, is writable, accepts this value, belongs to this locked source universe and survives the next evaluation/runtime stages.
