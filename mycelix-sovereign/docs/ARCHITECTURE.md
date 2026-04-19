# Mycelix Sovereign — Architecture

Skeleton. Will be expanded as components land.

## Component map

```
┌─────────────────────────────────────────────────────────────────┐
│                   Mycelix Sovereign (AGPL-3.0)                  │
│                                                                 │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│   │ Admin Console  │  │  Ticket API    │  │  SMTP Bridge   │   │
│   │ (Leptos)       │  │  (Athena)      │  │  (Pulse)       │   │
│   └───────┬────────┘  └───────┬────────┘  └───────┬────────┘   │
│           │                   │                   │            │
│   ┌───────┴───────────────────┴───────────────────┴────────┐   │
│   │                 Unified DID Session                    │   │
│   │              (mycelix-bridge-common)                   │   │
│   └───────┬───────────────────┬───────────────────┬────────┘   │
│           │                   │                   │            │
│   ┌───────┴────────┐  ┌───────┴────────┐  ┌──────┴──────────┐  │
│   │ xenia-ledger   │  │ mycelix-       │  │ mycelix-        │  │
│   │ (AGPL)         │  │ knowledge      │  │ identity        │  │
│   └───────┬────────┘  └────────────────┘  └─────────────────┘  │
│           │                                                    │
└───────────┼────────────────────────────────────────────────────┘
            │
┌───────────┼────────────────────────────────────────────────────┐
│           │    Protocol / transport layer (MIT + Apache-2.0)   │
│   ┌───────┴────────┐  ┌────────────────┐  ┌────────────────┐   │
│   │ xenia-wire     │  │ xenia-peer     │  │ xenia-capture  │   │
│   │ (envelope,     │  │ (TCP transport │  │ (cross-plat    │   │
│   │  AEAD, consent)│  │  + Session)    │  │  screen cap)   │   │
│   └───────┬────────┘  └───────┬────────┘  └───────┬────────┘   │
│           └───────────────────┴───────────────────┘            │
│                            │                                   │
│                    ┌───────┴────────┐                          │
│                    │ xenia-handshake│                          │
│                    │ (ML-KEM + Ed25519 hybrid)                 │
│                    └────────────────┘                          │
└────────────────────────────────────────────────────────────────┘
```

## Key architectural decisions

- **Open-core licensing**: protocol + transport crates ship permissive (adoption-first); application layer ships AGPL (SaaS moat).
- **Single-tenant by construction**: no shared control plane, ever. Year-2 managed tier is dedicated VPS per customer.
- **Consent at the wire**: every privileged session produces a third-party-verifiable ledger entry; admin cannot rewrite the log.
- **Post-quantum by default**: ML-KEM + Ed25519 hybrid on Xenia wire; Pulse email envelopes PQC-sealed with epoch ratchet.

## Not yet documented

- Deployment topology (single-host vs split)
- Network diagram
- Data retention & backup strategy
- Disaster recovery playbook
- Key rotation cadence

All deferred to W3 when Suite integration closes.
