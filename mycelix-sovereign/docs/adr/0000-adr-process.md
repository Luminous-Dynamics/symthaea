# ADR 0000 — ADR process

**Status:** Accepted
**Date:** 2026-04-19

## Context

Mycelix Sovereign bundles four components with distinct maturity, licensing, and team ownership. Suite-level architectural decisions need a durable record that survives team rotation and external audit (SOC 2 readiness needs an auditable decision trail).

## Decision

We adopt a lightweight ADR practice.

- ADRs live in `docs/adr/NNNN-slug.md` in this meta-repo.
- ADRs cover **Suite-wide** decisions: cross-component integration choices, cross-cutting security properties, licensing, deployment topology. Per-component internal decisions live in that component's own ADR directory.
- ADRs are numbered sequentially starting from `0001`. `0000` is reserved for this meta-ADR.
- Each ADR uses the structure:
  - **Context** — what problem we're solving, what constraints apply
  - **Decision** — what we're doing
  - **Consequences** — what this commits us to; what gets worse; what followups it generates
  - **Alternatives** — what we considered and rejected, with one-line reasons
- ADRs are **immutable once committed**. Revisions come as new ADRs that **Supersede** the old, which is kept in place with a `Superseded by ADR-NNNN` header appended.
- An ADR is **Accepted** when merged. Draft ADRs live on branches / PRs.

## Consequences

- Audit trail for every substantive Suite decision is a `git log` and a directory walk away.
- External reviewers (SOC 2 auditors, design-partner CISOs) can read ADRs as a compact history of the Suite's threat-model and integration choices.
- Cost: ~15 minutes per decision to draft. We accept this.

## Alternatives

- **No ADRs, decisions captured only in commit messages.** Rejected: commit messages don't surface for audit without a grep campaign.
- **ADRs in the parent monorepo root.** Rejected: the monorepo has 100+ clusters and the Suite needs its own decision tree.
- **ADRs in each component repo.** Rejected for Suite-wide decisions; these cross multiple components and a single home is clearer. Per-component ADRs still live in their own repos.

## References

- Michael Nygard, "Documenting architecture decisions" (2011)
- ThoughtWorks Tech Radar ADR pattern
