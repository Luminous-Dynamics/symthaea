# State vs Bridge Policy

## Core Rule
Mycelix is **local-first** and **agent-centric**. Sovereign state **must live on the DHT or the local device**.

- **State**: agent identity, trust, ledger events, governance, reputation, mail metadata, contracts.
  - **Where it lives**: Holochain source chains + DHT.
  - **Cache**: in-process local memory only.
- **Bridge**: temporary interoperability layers that do **not** hold sovereign state.
  - **Allowed** only if clearly labeled and scoped.

## What the Cloud May Do (Bridge Only)
- **Static delivery**: Vercel/Netlify/CDN hosting of frontend assets.
- **Legacy blob storage**: temporary storage of large encrypted files (S3/IPFS pinning) until peer storage is fully available.
- **Monitoring**: observability for ops, no user state.

## What the Cloud Must NOT Do
- Databases for consensus, trust, reputations, or core records.
- Centralized caches that become sources of truth.

## Current Bridge Inventory (Explicit)
- **Vercel**: static frontend delivery only (no secrets, no state).
- **S3**: legacy blob storage for large media/backups; keys must be community-held; data encrypted.
- **Pinning services** (Pinata/Web3.Storage): transient distribution only; data must remain portable and encrypted.

## Decommission Targets
- Centralized SQL/Redis used for core logic must be removed or replaced with Holochain zomes.
- Any bridge that grows beyond temporary use must be scheduled for removal or replaced with peer storage.

## Enforcement Checklist
- New services must declare: `STATE` or `BRIDGE`.
- Bridges must include a removal plan and data portability.
- No client-side secrets (ever).
