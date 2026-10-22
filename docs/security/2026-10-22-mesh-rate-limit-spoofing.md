# Mesh source-ID spoofing and rate-limit isolation

**Date:** 2026-10-22  
**Status:** Candidate fix; implementation tests have not been run locally.

## Finding

The mesh receive path previously called the per-peer rate limiter before verifying the packet MAC. The limiter indexes state by the eight-byte source ID carried inside the untrusted packet. An attacker could therefore send invalid-MAC telemetry using an established peer's claimed source ID and consume that peer's 100-packets-per-10-second quota. When the legitimate peer later sent a correctly authenticated packet, it could be dropped as rate limited.

This is a protocol-level availability issue: authenticity must be established before a packet can consume *trusted peer* quota. It is distinct from global resource protection against unauthenticated ingress.

## Candidate remediation

- Move the tracked-peer rate-limit check after successful MAC verification and after the authenticated replay-window check.
- Leave invalid-MAC telemetry subject to the existing bounded inbox and best-effort handling path, but do not let it mutate the authenticated peer's rate counter.
- Add a regression that establishes a peer with a valid MAC, sends more than 100 packets with the same claimed source ID and invalid MACs, then confirms a fresh valid packet is still accepted.

## Remaining boundary

A packet-controlled source ID is not a trustworthy network endpoint. This change does **not** solve unauthenticated flood attacks at the radio/socket edge; those should be limited using the real transport ingress/connection identifier before expensive processing. Such a limiter must be separate from authenticated per-peer policy and bounded in state. When a configured shared group MAC key is used, key holders may still forge another member's source ID; a group MAC does not provide individual sender attribution.

## Validation

Candidate tests: `spoofed_source_id_cannot_exhaust_authenticated_peer_rate_limit`, plus existing mesh feature tests. They have not been run locally and no pass is claimed. Exact-head CI should include the mesh feature and the existing HMAC/replay tests.
