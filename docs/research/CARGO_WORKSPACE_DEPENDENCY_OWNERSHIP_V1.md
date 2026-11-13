# Cargo workspace dependency ownership audit v1

## Purpose

Cargo 1.100's workspace dependency lint is useful, but its result is scoped to the workspace Cargo resolves. Symthaea contains nested workspaces, excluded packages, feature-gated dependencies and build/test targets. A root workspace warning is therefore a prompt for ownership analysis, not permission to delete a dependency.

The Rust xtask command:

```sh
cargo +1.100.0 run --locked -p xtask -- cargo-dependency-ownership
```

writes a deterministic JSON ownership report and a Markdown summary. Output paths are configurable with `--json` and `--markdown`.

## Classifications

- `USED_IN_WORKSPACE`: at least one member manifest inherits the workspace dependency entry.
- `USED_ONLY_UNDER_FEATURE`: every inheriting use is optional and excluded from default features.
- `USED_ONLY_BY_BUILD_OR_TEST_TARGET`: every inheriting use appears under build or development dependencies.
- `USED_BY_NESTED_WORKSPACE`: same-name use appears in a nested workspace, but it does not inherit the root workspace's version.
- `USED_BY_EXCLUDED_MEMBER`: same-name use appears under an explicit workspace exclusion; it is not counted as a root member.
- `UNUSED_IN_ROOT_WORKSPACE`: no inheriting member or same-name manifest use was observed for the root workspace entry.
- `UNUSED_IN_NESTED_WORKSPACE`: the corresponding nested workspace has no inheriting member or same-name use.
- `UNRESOLVED_OWNERSHIP`: same-name declarations or malformed manifests prevent a safe ownership conclusion.

Same-name declarations that do not use `workspace = true` are explicitly reported as non-inheriting candidates. They do not prove the dependency declaration is used or that their version matches the workspace entry.

## Discovery scope

The scanner walks the repository for Cargo.toml files, excluding build outputs and common vendored/tool directories. It reads:
- normal dependencies;
- development dependencies;
- build dependencies;
- target-specific dependency sections;
- optional dependencies and default-feature declarations;
- each workspace's members, excludes and workspace.dependencies table.

Nested workspace roots are resolved by nearest-ancestor workspace. Cargo member and exclude patterns support path-segment wildcards, including `*` and `**`.

The report records:
- exact Git commit;
- exact Cargo identity;
- root Cargo.toml SHA-256;
- every workspace-root manifest digest;
- each owning manifest path, package name, dependency section and manifest SHA-256;
- same-name non-inheriting candidates;
- parse failures and unresolved categories.

## Safety and claim ceiling

This audit is read-only. It does not update Cargo.toml, Cargo.lock, dependency versions, feature lists or source. It does not invoke Cargo to resolve an alternative dependency graph.

A report category of `UNUSED_IN_ROOT_WORKSPACE` means only that no workspace inheritance edge or same-name declaration was observed in the scanned manifests. It does not establish absence from generated code, external build contexts, unpublished tools or runtime-loaded plugins.

No dependency should be removed based on this report alone. Each future removal requires a separate change with:
1. exact package and feature consumer inventory;
2. root and nested workspace metadata checks;
3. locked build/test across affected workspaces;
4. explicit lockfile delta and review;
5. exact-head CI evidence.

The ownership audit is intended to turn Cargo 1.100's warning into an actionable, reproducible dependency graph rather than an automated pruning rule.
