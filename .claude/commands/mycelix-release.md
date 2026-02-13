---
description: "Orchestrate a Mycelix release: version bump, changelog, tag, verification"
---

# Mycelix Release Automation

Orchestrate a coordinated release across the Mycelix ecosystem.

Arguments: $ARGUMENTS

Expected: version number (e.g., "0.2.0") or "patch"/"minor"/"major" for auto-bump.

## Pre-Flight Checks

### 1. Ensure clean working tree
!cd /srv/luminous-dynamics && git status --porcelain mycelix-* crates/mycelix-* mycelix-workspace/ | head -10

If there are uncommitted changes, STOP and ask the user to commit or stash first.

### 2. Ensure on correct branch
!cd /srv/luminous-dynamics && git branch --show-current

Release should typically be from `main` or a release branch. Warn if on a feature branch.

### 3. Ensure tests pass
Run a quick validation (equivalent to `/mycelix-ci-local`):
- `cargo test --lib` in bridge-common, commons, civic
- `npm test` in sdk-ts (if TS SDK is being released)
- `pytest` in sdk-python (if Python SDK is being released)

## Version Bump

### Determine New Version

If user passed "patch", "minor", or "major":
- Read current version from `mycelix-workspace/sdk/Cargo.toml`
- Bump accordingly

If user passed explicit version (e.g., "0.2.0"):
- Use that version directly

### Update Version in All Locations

These files need version updates:

**Rust SDK:**
- `mycelix-workspace/sdk/Cargo.toml` -> `version = "{new_version}"`

**TypeScript SDK:**
- `mycelix-workspace/sdk-ts/package.json` -> `"version": "{new_version}"`

**Python SDK:**
- `mycelix-workspace/sdk-python/pyproject.toml` -> `version = "{new_version}"`

**Cluster Workspaces** (if releasing clusters):
- `mycelix-commons/Cargo.toml` -> `[workspace.package] version`
- `mycelix-civic/Cargo.toml` -> `[workspace.package] version`

**Do NOT change:**
- Individual zome Cargo.toml files (they use `version.workspace = true`)
- `hdk`/`hdi` versions (those are Holochain versions, not ours)

## Changelog Generation

### Gather Changes Since Last Release

!cd /srv/luminous-dynamics && git log --oneline $(git describe --tags --abbrev=0 2>/dev/null || echo "HEAD~20")..HEAD -- mycelix-* crates/mycelix-* mycelix-workspace/ | head -30

### Categorize Changes

Group commits by type:
- **Features**: `feat(mycelix):` commits
- **Fixes**: `fix(mycelix):` commits
- **Breaking**: Any commit mentioning "breaking" or "BREAKING"
- **Other**: Refactors, docs, tests

### Generate Changelog Entry

Format as:
```markdown
## [{new_version}] - {date}

### Features
- {feature description} ({commit hash})

### Fixes
- {fix description} ({commit hash})

### Breaking Changes
- {breaking description} ({commit hash})
```

## Build Verification

### Verify WASM builds
```bash
cd mycelix-commons && cargo build --release --target wasm32-unknown-unknown
cd mycelix-civic && cargo build --release --target wasm32-unknown-unknown
```

### Verify SDK builds
```bash
cd mycelix-workspace/sdk && cargo build --release
cd mycelix-workspace/sdk-ts && npm run build
```

### Run full test suite
```bash
cd mycelix-workspace && just test
```

## Create Release

### 1. Stage version changes
Stage all modified files (Cargo.toml, package.json, pyproject.toml, CHANGELOG.md).

### 2. Create release commit
```
git commit -m "release: mycelix v{new_version}"
```

### 3. Create git tag
```
git tag -a v{new_version} -m "Mycelix v{new_version}"
```

### 4. Push (with user confirmation)
Ask the user before pushing:
```
git push origin {branch} --tags
```

## Post-Release

### Checklist
- [ ] Version bumped in all locations
- [ ] CHANGELOG.md updated
- [ ] All tests passing
- [ ] WASM builds verified
- [ ] Git tag created
- [ ] Pushed to remote
- [ ] GitHub Release created (via `gh release create`)
- [ ] NPM package published (if applicable)

### NPM Publishing (if TypeScript SDK)
```bash
cd mycelix-workspace/sdk-ts
npm publish --access public
```

### Crates.io Publishing (if Rust SDK)
```bash
cd mycelix-workspace/sdk
cargo publish
```

## Summary

Output:
- Previous version -> New version
- Files modified
- Changelog preview
- Release tag
- Next steps for publishing
