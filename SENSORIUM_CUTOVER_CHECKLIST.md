# Sensorium Cutover Checklist

This checklist governs when the legacy `mycelix-portal -> mycelix-sensorium`
symlink can be removed safely.

## Current State

- Primary repo path: `mycelix-sensorium/`
- Compatibility symlink: `mycelix-portal -> mycelix-sensorium`
- Top-level package: `mycelix-sensorium`
- Binary: `mycelix-sensorium`
- Legacy compatibility artifact intentionally retained:
  - `mycelix-portal-master-wrap-v1` salt in `sensorium-shell/src/identity/master_key.rs`

## Required Gates Before Symlink Removal

1. Repo-local build verification passes using the new path only.
   - Verified:
     - `cargo check --manifest-path /srv/luminous-dynamics/mycelix-sensorium/Cargo.toml`

2. Repo-local docs and active scripts point at `mycelix-sensorium`.
   - Verified for primary frontend planning docs and shell workspace.
   - Re-verify after any deployment or README edits.

3. Deployment/configuration references are updated or explicitly accepted.
   - Check:
     - reverse proxy / nginx configs
     - NixOS service definitions
     - docker-compose files
     - CI templates
     - local helper scripts

4. Hostname strategy is explicit.
   - Decide whether `portal.mycelix.net` remains:
     - a temporary compatibility hostname
     - a redirect to `sensorium.mycelix.net`
     - or is retired outright

5. Off-repo automation is audited.
   - Check:
     - personal shell aliases
     - deployment notebooks
     - external scripts
     - any runbooks not stored in this repository

## Safe Removal Procedure

1. Re-run repo-local search for `mycelix-portal` references.
2. Confirm only compatibility references remain.
3. Remove symlink:
   - `rm /srv/luminous-dynamics/mycelix-portal`
4. Re-run:
   - `cargo check --manifest-path /srv/luminous-dynamics/mycelix-sensorium/Cargo.toml`
5. Fix any breakage immediately rather than recreating mixed naming.

## Do Not Remove

- The legacy master-key salt should remain unchanged unless you are deliberately
  designing and migrating a new wrap format.
