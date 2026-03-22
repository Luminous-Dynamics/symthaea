# Dependency Security Policy

Last updated: 2026-03-22

This document defines how Luminous Dynamics manages third-party dependencies across all projects (Rust, Python, Nix, TypeScript/JavaScript).

---

## 1. Version Pinning Strategy

### Rust (Cargo)

- **Cargo.toml**: Use semver-compatible ranges (e.g., `serde = "1.0"`, `tokio = "1"`). Pin exact versions only when a specific release is required for correctness.
- **Cargo.lock**: Always committed. This is the source of truth for reproducible builds.
- **Workspace inheritance**: Shared dependencies declared in root `[workspace.dependencies]` to prevent version drift across crates.

### Python (Poetry)

- **pyproject.toml**: Caret ranges (`^1.0`) for flexibility within semver-major.
- **poetry.lock**: Always committed.

### Nix

- **flake.lock**: Always committed. Pin nixpkgs and all flake inputs to exact revisions.
- Update inputs deliberately via `nix flake update` with testing, not automatically.

### TypeScript/JavaScript (npm)

- **package.json**: Caret ranges (`^`) for dependencies, exact versions for critical security-sensitive packages.
- **package-lock.json**: Always committed.

### General Rule

Lockfiles are always committed. A build without a lockfile is not reproducible and must not ship.

---

## 2. New Dependency Approval Process

Before adding a new dependency, verify all of the following:

| Criterion | Threshold |
|-----------|-----------|
| **Active maintenance** | At least one commit within the last 12 months |
| **License compatibility** | Must appear in the `deny.toml` allowlist (see below) |
| **Security audit** | Must pass `cargo-audit` / `npm audit` / `pip-audit` with no unpatched CRITICAL or HIGH advisories |
| **Adoption** (Rust) | Prefer crates with >100K all-time downloads on crates.io |
| **Adoption** (npm/PyPI) | Prefer packages with >10K weekly downloads |
| **Transitive cost** | Check `cargo tree -i <crate>` or equivalent; avoid pulling large transitive trees for small features |

### Approved Licenses

From `symthaea/deny.toml`:

```
MIT, Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause,
0BSD, ISC, Unicode-3.0, Unicode-DFS-2016, Zlib, BSL-1.0, CC0-1.0, OpenSSL,
MPL-2.0, AGPL-3.0-or-later, AGPL-3.0-only, Unlicense, CDLA-Permissive-2.0,
OFL-1.1, Ubuntu-font-1.0, bzip2-1.0.6
```

Any dependency with a license not on this list requires explicit approval documented in a commit or PR.

### Commit Message Convention

When adding a new dependency, the commit message must include:

```
dep(<ecosystem>): add <package> <version>

Justification: <why this dep is needed and why alternatives were rejected>
License: <SPDX identifier>
Downloads: <approximate count>
```

---

## 3. CVE Response SLA

| Severity | Response Time | Action |
|----------|---------------|--------|
| **CRITICAL** (CVSS >= 9.0) | 24 hours | Patch, upgrade, or apply mitigation (e.g., feature-gate, remove usage path). If no fix exists, document risk and disable affected functionality. |
| **HIGH** (CVSS 7.0-8.9) | 7 days | Upgrade to patched version. If blocked, document in `deny.toml` ignore list with justification and expiry date. |
| **MEDIUM** (CVSS 4.0-6.9) | 30 days | Upgrade in next planned dependency update. |
| **LOW** (CVSS < 4.0) | Next release cycle | Bundle with routine dependency updates. |

### Ignored Advisories

Any advisory added to `deny.toml`'s `[advisories] ignore` list must include:

1. A comment explaining why it is safe to ignore (exposure analysis)
2. The advisory ID
3. Review at least quarterly to check if a fix is available

---

## 4. Tooling Requirements

### Rust

| Tool | Purpose | Enforcement |
|------|---------|-------------|
| **cargo-deny** | License, advisory, ban, and source checks | Required for all workspaces. Run: `cargo deny check` |
| **cargo-audit** | RUSTSEC advisory database | CI-blocking on CRITICAL and HIGH. Run: `cargo audit` |

`deny.toml` sections enforced:

- `[advisories]` -- known vulnerability check
- `[licenses]` -- allowlist enforcement (confidence >= 0.8)
- `[bans]` -- duplicate version warnings, wildcard policy
- `[sources]` -- registry restrictions (crates.io only for production)

### TypeScript/JavaScript

| Tool | Purpose | Enforcement |
|------|---------|-------------|
| **npm audit** | Known vulnerability check | CI-blocking at `--audit-level=high` |

### Python

| Tool | Purpose | Enforcement |
|------|---------|-------------|
| **pip-audit** | PyPI advisory check | CI-blocking on CRITICAL and HIGH |

### Automated Updates

| Tool | Scope | Configuration |
|------|-------|---------------|
| **Dependabot** | All public repos with dependencies | Enabled for Cargo, npm, pip, GitHub Actions |

Note: This monorepo is private. Dependabot runs on the public standalone repos (symthaea, mycelix) which receive changes via `symthaea/scripts/sync-to-standalone.sh`.

---

## 5. Supply Chain Security

### Registry Restrictions

| Ecosystem | Allowed Registries |
|-----------|--------------------|
| Rust | crates.io only (`https://github.com/rust-lang/crates.io-index`) |
| Node.js | npmjs.com only |
| Python | PyPI only |

**No git dependencies in production code.** Git dependencies are acceptable only in `[dev-dependencies]` for unreleased test utilities. Any exception requires a comment explaining when it will move to a registry release.

Enforced in `deny.toml`:

```toml
[sources]
unknown-registry = "warn"
unknown-git = "warn"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
allow-git = []
```

### SBOM Generation

Generate a Software Bill of Materials on each release:

```bash
cargo install cargo-cyclonedx
cargo cyclonedx --all
```

Output: CycloneDX JSON SBOM per crate, committed to the release tag or attached as a release artifact.

### Lockfile Hygiene

- Lockfiles (`Cargo.lock`, `poetry.lock`, `package-lock.json`, `flake.lock`) are always committed.
- PRs that modify lockfiles without a corresponding dependency change in the manifest file require explanation.
- Lockfile-only updates (e.g., `cargo update`) should be isolated commits for clear audit trail.

---

## 6. Routine Maintenance

### Monthly

- Run `cargo audit`, `npm audit`, `pip-audit` across all projects.
- Review `deny.toml` ignore list for stale entries.
- Run `cargo deny check` to catch new license or ban violations.

### Quarterly

- Evaluate major version bumps for key dependencies.
- Review and prune unused dependencies (`cargo machete` or manual `cargo tree` analysis).
- Verify Dependabot is active and PRs are being triaged on public repos.

### On Each Release

- Generate SBOM via `cargo-cyclonedx`.
- Confirm zero CRITICAL/HIGH advisories or document accepted risk.

---

## 7. Exceptions

Any deviation from this policy must be documented in a commit message or PR description with:

1. What rule is being bypassed
2. Why the bypass is necessary
3. When it will be resolved (target date or condition)

Exceptions are reviewed quarterly during routine maintenance.
