# Nixward CROSS-070 — Governed systemd Lifecycle Transaction Waist V1

## Purpose

CROSS-070 turns the safe ordering of the native systemd lifecycle primitives into a reusable API.

The individual hardening layers already existed:

- CROSS-061 pre-arms JobRemoved;
- CROSS-063 binds systemd manager incarnation;
- CROSS-065 dispatches to the captured unique manager owner;
- CROSS-064 provides typed native lifecycle mutation.

The remaining risk was caller-side composition: a consumer could manually combine those capabilities in the wrong order.

## Canonical sequence

`watch -> dispatch -> capture -> await`

The orchestration API is `NixSystemdLifecycleTransactionV1::dispatch_and_observe()`.

It performs:

1. canonical typed service-operation validation;
2. JobRemoved watcher arming;
3. capture of the watcher's systemd manager unique owner;
4. native lifecycle dispatch to that exact unique owner;
5. live Job capture from the returned object path;
6. live JobType correlation against the typed operation;
7. consumption of the pre-armed JobRemoved watcher;
8. manager-incarnation verification on the resulting evidence.

## Why the order matters

Arming after dispatch permits a sufficiently fast systemd job to emit JobRemoved before the observer starts listening.

Dispatching to the well-known name after capturing one manager incarnation permits a manager restart to retarget the mutation to a replacement incarnation.

Capturing Job metadata only after completion loses JobType provenance because JobRemoved itself does not carry JobType.

The transaction waist prevents these sequencing errors from being convention-only.

## Boundary

The transaction module is transport/orchestration only.

It does not:

- decide whether the action is authorized;
- mint a live authorization capability;
- inspect arbitrary filesystem paths;
- expose a D-Bus proxy;
- parse shell commands;
- grant a broader mutation scope.

Authorization and currentness remain upstream.

Post-state semantic proof remains downstream.

## Failure semantics

All of the following are fail-closed:

- invalid service operation;
- unsupported operation such as Enable/Disable in the lifecycle transport;
- watcher-arm failure;
- manager-owner mismatch;
- mutation transport failure;
- Job capture failure;
- JobType mismatch;
- malformed/mismatched JobRemoved evidence;
- timeout.

A caller never receives a positive lifecycle evidence object from a partially completed sequence.

## Relationship to receipts

The returned `NixSystemdLifecycleEvidenceV1` is intentionally narrower than a post-state receipt.

It proves only the correlated transport-level lifecycle event. It does not prove final ActiveState, UnitFileState, definition identity, stability, or authorization.

The existing post-state observer and receipt verifier consume these narrower facts to establish their stronger claims.

## Claim ceiling

CROSS-070 does not provide:

- an atomic multi-call D-Bus/CAS transaction;
- proof that the unit state did not change between JobRemoved and post-state observation;
- cryptographic attestation of systemd;
- authorization by itself.

Its guarantee is procedural and structural: governed lifecycle callers have one narrow API whose implementation fixes the required ordering of watcher arming, exact-owner dispatch, Job capture, and JobRemoved correlation.

## Qualification

Exact-head GitHub Actions are the qualification authority. Queued, pending, cancelled, mergeable, or static-only states are not PASS evidence.