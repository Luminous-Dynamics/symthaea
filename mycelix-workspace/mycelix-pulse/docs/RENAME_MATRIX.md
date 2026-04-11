# Mycelix Pulse Rename Matrix

This file separates branding work from compatibility-sensitive identifier
migration.

## Rename Now

These should use `Mycelix Pulse` consistently:

- repo-local directory names and docs
- UI titles, app shell labels, landing copy, and notification text
- README files and planning documents
- crate descriptions and package descriptions
- PWA metadata and product-facing desktop/mobile names

## Keep Stable For Now

These currently stay on legacy `mycelix-mail` / `mycelix_mail` identifiers:

- Holochain `app_id` values
- Holochain role names like `mycelix_mail`
- installed app ids in conductor configs
- network seeds
- localStorage keys used by existing clients
- database names such as `mycelix_mail`
- service names, namespaces, and deployment handles already used in ops

## Needs Migration Plan

These should only change with explicit migration and rollback steps:

- `holochain/conductor-config.yaml`
- `apps/leptos/src/holochain.rs`
- `deploy/nixos/`
- `deploy/kubernetes/`
- `deploy/terraform/`
- desktop/mobile bundle identifiers
- backend database connection defaults

## Recommended Migration Sequence

1. Finish user-facing branding cleanup.
2. Prove the current live vertical slice against the legacy runtime ids.
3. Inventory persisted identifiers in running environments.
4. Design a migration for conductor install ids, network seed, storage keys, and deployment names.
5. Execute the migration with compatibility shims where possible.
