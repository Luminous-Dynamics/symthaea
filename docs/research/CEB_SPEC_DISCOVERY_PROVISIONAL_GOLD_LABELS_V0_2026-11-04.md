# CEB specification-discovery provisional gold-label annotations V0

**Date:** 2026-11-04  
**Status:** public reviewer working notes; not a sealed gold set, not independently adjudicated, and not an executable evaluator.  
**Tasks:** `CEB-SPEC-001`, `CEB-SPEC-004`  
**Fixture:** `CEB-REPO-001-SYNTHETIC-V0` at source revision `f09e5d642ea5f13a4888553c94d799c62c7b1f83`  
**Catalog:** `benchmarks/ceb/catalog.v0.json`  
**Schema:** existing `benchmarks/ceb/task.schema.json`  
**Claim ceiling:** every annotation below is either directly supported by source text, inferred from source control flow, or explicitly underdetermined. None is compiler-resolved or runtime-observed.

---

## 1. Why these notes exist

The first task-bank tranche needs concrete, scoped claim sets before we can sensibly discuss graders or benchmark outcomes. These notes are the first independent authoring pass over two tasks.

This document is intentionally visible and provisional. It must **not** be used as a hidden answer key. For evaluation, rebuild the agent-visible workspace from a source-only allowlist, keep evaluator annotations in a separately provisioned read-only location, commission independent reviewers, and generate a fresh holdout template. The current fixture directory contains design/evaluator artifacts alongside source, so merely instructing an agent not to open a file is not access control.

## 2. Common evidence-authority vocabulary

Use these labels consistently:

- `source_declared`: a declaration, type, visibility, manifest entry or `cfg` expression is literally present in the pinned source.
- `syntax_inferred`: a relation such as a call, `impl`, conditional or `map` is inferred from source syntax / control flow.
- `compiler_resolved`: a pinned independent compiler or language-server run establishes name/type/trait/configuration resolution. **No such evidence exists for this fixture.**
- `runtime_observed`: a behavior was observed by an isolated, qualified run. **No such evidence exists for this fixture.**
- `externally_specified`: a normative requirement comes from an approved external/task specification. These two tasks do not currently provide one.
- `underdetermined`: the evidence set does not identify the proposition, or leaves multiple interpretations possible.

Do not elevate source comments into normative requirements. Do not count copied statements as independent corroboration. Every claim is scoped to the fixture revision and configuration described.

---

## 3. CEB-SPEC-001 — public trait and caller contracts across Rust crates

### 3.1 Provisional atomic claims

| Claim ID | Proposition | Evidence | Authority | Provisional status |
|---|---|---|---|---|
| S001-C01 | `UserId` is a public tuple struct containing one `u64` field. | `crates/domain/src/lib.rs:3-4` | `source_declared` | Supported |
| S001-C02 | `User` is a public struct with public `id: UserId` and `display_name: String` fields. | `crates/domain/src/lib.rs:6-10` | `source_declared` | Supported |
| S001-C03 | The public `Authorizer` trait declares `can_read(&self, user: &User) -> bool`. | `crates/domain/src/lib.rs:12-14` | `source_declared` | Supported |
| S001-C04 | The source declares `DefaultAuthorizer` as an implementation of `Authorizer`. | `crates/domain/src/lib.rs:16-23` | `syntax_inferred` | Supported from source; not compiler-resolved |
| S001-C05 | The body of `DefaultAuthorizer::can_read` returns the comparison `user.id.0 != 0`. | `crates/domain/src/lib.rs:19-23` | `source_declared` | Supported as source behavior |
| S001-C06 | The public `Store` trait declares `get_user(&self, id: UserId) -> Option<User>`. | `crates/store/src/lib.rs:6-8` | `source_declared` | Supported |
| S001-C07 | `MemoryStore` stores users in a `BTreeMap<UserId, User>`; its `insert` method indexes by `user.id`. | `crates/store/src/lib.rs:10-19` | `source_declared` | Supported |
| S001-C08 | The source declares `MemoryStore` as implementing `Store`; its method looks up an ID and clones the found user into an `Option<User>`. | `crates/store/src/lib.rs:21-25` | `syntax_inferred` | Supported from source; not compiler/runtime verified |
| S001-C09 | `UserService<S,A>` declares bounds `S: Store` and `A: Authorizer` on its struct and inherent impl. | `crates/service/src/lib.rs:6-18` | `source_declared` | Supported |
| S001-C10 | `UserService::get_user` first calls `store.get_user(id)?`; if that expression yields `None`, the function returns `None` before invoking the authorizer. | `crates/service/src/lib.rs:24-27` | `syntax_inferred` | Supported from control flow |
| S001-C11 | When a user is present, `UserService::get_user` evaluates `authorizer.can_read(&user)` and uses that bool as the condition for `then_some(user)`. | `crates/service/src/lib.rs:24-27` | `syntax_inferred` | Supported from control flow |
| S001-C12 | `Profile` is a public struct with a public `user: User` field. | `crates/api/src/profile_api.rs:7-10` | `source_declared` | Supported |
| S001-C13 | `get_profile` declares the same `Store` and `Authorizer` bounds for its generic service and maps a returned user into `Profile`. | `crates/api/src/profile_api.rs:12-18` | `source_declared` + `syntax_inferred` | Supported from source |
| S001-C14 | The API crate root declares `profile_api` private and publicly re-exports `Profile` and `get_profile`. | `crates/api/src/lib.rs:3-5` | `source_declared` | Supported |
| S001-C15 | The source-level service path returns `None` when storage yields `None` or when `can_read` returns false; the API maps that `None` to `None`. | `crates/service/src/lib.rs:24-27`; `crates/api/src/profile_api.rs:12-18` | `syntax_inferred` | Supported as a control-flow derivation, not a runtime observation |
| S001-C16 | The reason that ID zero is denied is that it represents an invalid, reserved, anonymous or privileged identity. | No evidence in the source set | `underdetermined` | Must not be asserted |
| S001-C17 | Every downstream implementation of `Authorizer` or `Store` must satisfy additional semantic requirements beyond the declared method signatures. | No external or normative contract supplied | `underdetermined` | Must not be invented |
| S001-C18 | The declared trait implementations and call relationships are accepted by the pinned Rust compiler. | No compiler receipt | `compiler_resolved` required | Not established |
| S001-C19 | The described return behavior was observed at runtime for all relevant inputs. | No runtime receipt | `runtime_observed` required | Not established |

### 3.2 Important scope distinctions

- C05 describes what this particular implementation's source body expresses. It does not establish why ID zero is treated differently.
- C10-C15 are source-control-flow deductions. They are not observed executions.
- C04 and C08 name `impl` declarations in source. They must not be relabelled compiler-resolved until a pinned compiler/LSP tool emits trustworthy evidence.
- A public trait signature states the callable type contract, not every behavioral obligation an implementation should satisfy.
- The source does not specify whether denial should be an error, a missing value, a distinct authorization result, or a logged audit event. The current code maps it to `None`; intent beyond that remains unknown.

### 3.3 Falsification and reviewer controls

- Replace the implementation with one that returns the opposite bool while preserving the method signature: signature-only extraction must not claim the same behavioral policy.
- Change `Store::get_user` to return a different user than the requested ID: a contract claim that it always returns the matching user must be challenged unless an independent invariant or requirement supports it.
- Make a candidate claim that `UserId(0)` is reserved: the grader must mark it unsupported / underdetermined.
- Remove or alter a trait implementation: do not leave the old semantic edge in the graph.
- Add a second valid `Authorizer` implementation with different semantics: the task must not generalize `DefaultAuthorizer` behavior to all implementors.
- Supply the same statement in a comment and a README copied from a common source: source-origin lineage must prevent counting them as two independent proofs.

### 3.4 Gold-label review questions

1. Are the method-level visibility and re-export claims stated with the exact Rust language semantics, without overclaiming compiler resolution?
2. Are C10-C15 expressed as source-derived deductions rather than observed behavior?
3. Are all intended, inferred and underdetermined claims separated correctly?
4. Does the rubric reward a correct uncertainty statement about ID-zero intent?

---

## 4. CEB-SPEC-004 — Cargo feature and target applicability

### 4.1 Provisional source facts

| Claim ID | Proposition | Evidence | Authority | Provisional status |
|---|---|---|---|---|
| S004-C01 | The API crate declares `default = ["audit"]`. | `crates/api/Cargo.toml:7-10` | `source_declared` | Supported |
| S004-C02 | The `audit` feature has no feature dependencies in its declaration. | `crates/api/Cargo.toml:7-10` | `source_declared` | Supported |
| S004-C03 | The `windows-native` feature enables the optional `windows-sys` dependency through `dep:windows-sys`. | `crates/api/Cargo.toml:7-10,16-17` | `source_declared` | Supported |
| S004-C04 | `windows-sys` is declared under the `cfg(windows)` target-specific dependency table and is optional. | `crates/api/Cargo.toml:16-17` | `source_declared` | Supported |
| S004-C05 | The audit module is gated by `feature = "audit"`. | `crates/api/src/lib.rs:11-12` | `source_declared` | Supported |
| S004-C06 | The Windows module is gated by both `target_os = "windows"` and `feature = "windows-native"`. | `crates/api/src/lib.rs:14-15` | `source_declared` | Supported |
| S004-C07 | Enabling `windows-native` on Linux activates the feature flag but does not satisfy the source module's Windows target predicate. | `crates/api/Cargo.toml:10,16-17`; `crates/api/src/lib.rs:14-15` | `syntax_inferred` | Supported as static predicate evaluation |
| S004-C08 | Under the declared default-feature set, `audit` is selected unless defaults are disabled or the package's feature selection changes the effective set. | `crates/api/Cargo.toml:7-10` | `source_declared` / `syntax_inferred` | Supported within the stated package configuration |
| S004-C09 | Under no-default-features, no explicit features, the audit feature predicate is false. | `crates/api/Cargo.toml:7-10`; `crates/api/src/lib.rs:11-12` | `syntax_inferred` | Supported as static predicate evaluation |
| S004-C10 | Under Windows plus `windows-native`, both conditions in the Windows module gate are true; the target-specific optional dependency is also eligible for activation through the feature declaration. | `crates/api/Cargo.toml:10,16-17`; `crates/api/src/lib.rs:14-15` | `syntax_inferred` | Supported as static declared-condition analysis; not compiler-resolved |

### 4.2 Static world matrix

The ten rows below are the provisional expected *predicate evaluation* for the fixture's explicit package-level worlds. “Feature enabled” and “module/dependency active” are separate columns.

| Target | Feature selection | Effective feature set | Audit feature/module | Windows-native feature | Windows module | `windows-sys` target dependency |
|---|---|---|---:|---:|---:|---:|
| Linux | Defaults | `audit` | Active | Inactive | Inactive | Inactive |
| Linux | No defaults | empty | Inactive | Inactive | Inactive | Inactive |
| Linux | No defaults + audit | `audit` | Active | Inactive | Inactive | Inactive |
| Linux | No defaults + windows-native | `windows-native` | Inactive | Active | Inactive | Inactive |
| Linux | No defaults + both | `audit, windows-native` | Active | Active | Inactive | Inactive |
| Windows | Defaults | `audit` | Active | Inactive | Inactive | Inactive |
| Windows | No defaults | empty | Inactive | Inactive | Inactive | Inactive |
| Windows | No defaults + audit | `audit` | Active | Inactive | Inactive | Inactive |
| Windows | No defaults + windows-native | `windows-native` | Inactive | Active | Active | Eligible/active |
| Windows | No defaults + both | `audit, windows-native` | Active | Active | Active | Eligible/active |

The Windows row for `windows-sys` is labelled “eligible/active” because this is a static derivation from the optional dependency alias and target table, not a Cargo metadata or compiler receipt. A future qualified evaluator should verify the effective dependency graph independently.

### 4.3 Explicit non-claims / unresolved conditions

- This is a package-level matrix for the task's declared feature selections. It is not a claim about arbitrary workspace-wide feature unification, transitive dependency activation or every downstream consumer graph.
- A manifest feature being selected does not prove that a symbol can be linked or that the crate compiles for that target.
- Source-level `cfg` conditions do not establish compiler-resolved module inclusion.
- No Cargo metadata, `cargo check`, target installation, dependency download or compiler was run.
- The `worlds.v0.json` expected sets are design labels, not compiler-produced results.

### 4.4 Falsification and reviewer controls

- Set `default = []`: the default-world audit expectation must change; stale active labels must be rejected.
- Explicitly enable `windows-native` on Linux: the feature must be distinguished from target-gated module/dependency activation.
- Remove `target_os = "windows"` from the module cfg: the Linux windows-native world should now be detected as a different source predicate.
- Move `windows-sys` out of the target-specific table: dependency availability must be recalculated separately from module gating.
- Duplicate a world row or omit the no-feature world: catalog validation should reject duplicates or missing coverage.
- Preserve each row's default-feature provenance; explicit `audit` and default-provided `audit` may produce the same effective feature set but are not the same request provenance.

### 4.5 Gold-label review questions

1. Does the task distinguish default selection, explicit selection, effective features, target conditions and actual dependency resolution?
2. Is “feature active on Linux while Windows module inactive” explained without conflating target dependency eligibility?
3. Are Cargo feature-unification assumptions stated?
4. Are the world matrix and task agent prompt scoped to the same package-level invocation semantics?

---

## 5. Readiness and separation checklist

Before either task can become `fixture_ready`:

- [ ] Materialize an agent-facing source-only tree that excludes `gold-graph.v0.json`, `worlds.v0.json`, `mutations.v0.json` and this annotation document.
- [ ] Store gold claims and expected world rows in evaluator-only custody, outside all candidate-writable paths.
- [ ] Obtain an independent second reviewer for every atomic claim and every authority assignment.
- [ ] Resolve reviewer disagreements explicitly; preserve unresolved claims rather than forcing consensus.
- [ ] Pin the exact fixture tree digest and environment/toolchain contract.
- [ ] Add known-good, known-bad, empty-answer and source-laundering controls.
- [ ] Add scope mutations and check invalidation of stale claims.
- [ ] Qualify the independent grader; do not let the candidate author or edit it.
- [ ] Keep both tasks `design_only` until the OS isolation and evidence/trust prerequisites are satisfied.
- [ ] Report all unsupported, underdetermined and not-assessed dispositions; do not score a polished answer as correct without claim-level support.

## 6. Status

- Provisional task annotations authored: 2.
- Atomic claims listed: 29.
- Feature/target rows described: 10.
- Independent reviewer approvals: 0.
- Sealed gold labels: 0.
- Executable tasks: 0.
- Qualified tasks: 0.
- Benchmark results: 0.

This is a static research artifact intended to accelerate independent review. It is not a gold-standard dataset release and does not establish compiler semantics, runtime behavior, or agent performance.
