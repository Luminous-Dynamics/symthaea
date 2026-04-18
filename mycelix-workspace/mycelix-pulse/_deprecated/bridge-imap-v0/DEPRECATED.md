# DEPRECATED: `bridge/` IMAP daemon v0

**Retired:** 2026-04-19 (Phase 0.4 of `PULSE_READINESS_PLAN.md`)
**Replacement:** Phase 5 `pulse-smtp-gateway` crate (see plan)

## Why retired

This was a 128-LOC skeleton that polled IMAP accounts and called a
`conductor.relay_inbound(&from, &subject, &body, &account_email)` method
that **did not exist as a real zome call**. The relay was a log-and-drop
stub: the daemon correctly fetched IMAP messages, then silently dropped
them instead of persisting to the DHT. Nothing in the zome surface
received them. The code audit in `PULSE_READINESS_PLAN.md` §2 documents
this as the "ghost bridge."

Keeping it around misleads future readers into thinking the external-mail
path works, and its presence in the top-level `mycelix-pulse/` tree gives
the impression of a working interop story when there is none.

## Why not deleted

Git history preserves the full thing, but keeping the code visible here
means:

- Nobody accidentally re-animates it thinking it's live (the `DEPRECATED.md`
  is unmissable)
- The IMAP-polling and OAuth patterns it demonstrated are reference material
  for the Phase 5 gateway's own inbound path (just without the drop-on-floor
  at the end)
- `lettre`/`imap`/`native-tls` version choices here inform Phase 5's
  dependency picks

## What replaces it

Phase 5 of `PULSE_READINESS_PLAN.md`: `pulse-smtp-gateway` — a proper SMTP
receiver on port 25 of a real VPS (Hetzner/OVH, unblocked 25, provider-set
PTR), built on `mailin-embedded` + `mail-auth` + `mail-send` + `mail-parser`
+ `hickory-resolver`. See §3.2 of the plan and the "Phase 5 — SMTP gateway
daemon" section for the full architecture.

The Phase 5 gateway writes directly to Holochain via a real `receive_external`
zome extern, closing the gap this daemon never filled.
