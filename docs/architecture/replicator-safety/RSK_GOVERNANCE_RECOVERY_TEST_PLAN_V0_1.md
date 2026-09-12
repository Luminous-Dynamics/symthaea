# Replicator Safety Kernel — Governance and Recovery Test Plan v0.1

Status: **normative verification plan; authored design evidence only**

This document defines the minimum negative, property, fault, and recovery evidence required before governance/recovery APIs can participate in production RSK authority.

It contains no physical replication mechanism.

---

## 1. Test-family goals

The implementation must prove both:

1. ordinary governance roles cannot escape their exact negative/policy scope;
2. recovery authority is structurally distinct, externally governed, and unable to silently resurrect old positive authority.

---

## 2. Governance command vectors — GR-CMD

Required cases:

- valid command with exact role/kind/target/epoch/head accepted;
- wrong command kind rejected;
- wrong target kind rejected;
- wrong target identifier rejected;
- wrong epoch rejected;
- stale expected sequence rejected;
- stale/wrong expected head digest rejected;
- expired/not-yet-valid command rejected;
- duplicate command/mutation ID does not apply twice;
- signature/digest mutation rejected;
- wrong issuer role rejected;
- retired/revoked signer rejected;
- stale/rolled-back trust snapshot rejected;
- wrong policy digest rejected;
- noncanonical/oversized command rejected before mutation.

For every rejection assert authoritative ledger state is unchanged.

---

## 3. Role-separation vectors — GR-ROLE

### Runtime safety monitor

Prove monitor may perform only configured negative operations and cannot:

- mint a grant;
- widen capability/budget ceilings;
- clear quarantine/revocation;
- recover/create a new epoch;
- choose a fork winner.

### Quarantine authority

Prove quarantine command cannot be reinterpreted as:

- revocation if not separately authorized;
- unquarantine;
- grant issue;
- policy widening;
- recovery.

### Revocation authority

Prove scoped revocation cannot:

- revoke a different target/generation;
- clear another revocation;
- mint replacement grant;
- recover/new epoch.

### Policy authority

Prove policy authority cannot:

- lower constitutional floors;
- widen immutable ancestor ceilings;
- clear committed negative state;
- choose fork winner;
- perform recovery without recovery role.

### Ordinary operator

Prove possession of operator credentials alone cannot invoke recovery-sensitive operations.

Threat coverage: T16, T24, T25.

---

## 4. Negative-state tests — GR-NEG

For subject, lineage, and grant-generation scopes:

- authenticated quarantine takes effect at exact next state;
- authenticated revocation takes effect at exact next state;
- later positive grant/quorum evidence cannot override committed negative state;
- descendant/ancestry propagation remains consistent with constitutional rules;
- unknown/ambiguous target fails closed;
- uncertainty/freshness loss causes freeze rather than unbounded negative mutation unless policy explicitly authorizes a negative command.

---

## 5. Command replay/property tests — GR-REPLAY

Generate sequences of:

- governance command authorization;
- intervening ledger mutation;
- stale command delivery;
- duplicate delivery;
- command-target substitution;
- policy/trust rotation;
- key revocation;
- epoch transition.

Properties:

1. one command ID applies at most once;
2. command valid for cursor/head H cannot apply to H' unless exact semantics explicitly allow it;
3. command valid in epoch E cannot apply in E+1;
4. target/kind substitution never succeeds;
5. failed command leaves state unchanged;
6. revoked/wrong-role authority never creates a successful governance mutation.

---

## 6. Fork tests — GR-FORK

Required cases:

- two valid conflicting successors for one predecessor -> explicit fork/non-operational state;
- both branches retained;
- last-write-wins prohibited;
- highest timestamp prohibited;
- longest branch prohibited unless an explicitly reviewed recovery policy ever defines such a rule;
- valid transparency receipt for one branch does not choose it;
- valid receipts for both branches surface contradiction/fork;
- ordinary operator/policy/revoker cannot choose winner;
- new positive authority denied until recovery.

Threat coverage: T20, T22, T25.

---

## 7. Recovery proposal vectors — GR-REC-PROP

Required cases:

- canonical valid recovery proposal accepted structurally;
- old epoch mismatch rejected;
- old terminal head mismatch rejected;
- missing known fork branch head rejected where required;
- incident/evidence root mismatch rejected;
- wrong recovery policy digest rejected;
- invalid new epoch ID/genesis commitment rejected;
- invalid/widening preservation ceiling rejected;
- malformed/oversized preservation plan rejected;
- unknown critical field/discriminant rejected.

---

## 8. Recovery quorum vectors — GR-REC-QUORUM

Required cases:

- exact valid proposal + valid independent recovery quorum produces opaque verified recovery plan;
- signatures over different proposal digests cannot combine;
- duplicate signer identity cannot inflate recovery quorum;
- same required failure domain cannot satisfy distinct-domain threshold;
- ordinary grant/revocation/operator key cannot count unless explicitly recovery-authorized;
- retired/revoked/expired recovery signer rejected;
- stale trust snapshot rejected;
- approval expiry intersection enforced;
- compromised/forbidden subject-controlled signer rejected according to recovery policy.

Recovery threshold compromise itself remains a named catastrophic trust-root assumption, not something these tests can "prove impossible."

---

## 9. Fresh-epoch transition tests — GR-EPOCH

After a valid recovery plan:

- new epoch ID differs from old;
- new genesis binds old terminal/fork/recovery evidence;
- old history remains immutable/readable;
- preserved identity/lineage mapping exactly follows recovery plan;
- new hard ceilings do not silently widen beyond allowed recovery/constitutional constraints;
- **no active grant carries over automatically**;
- old bounded authorization token fails in new epoch;
- old verified quorum capability cannot authorize new-epoch action;
- old runtime witness cannot become current new-epoch witness;
- old cursor/head token invalid;
- new positive authority requires new/reverified grant/evidence.

Threat coverage: T25, T26.

---

## 10. Key/trust compromise recovery tests — GR-TRUST-REC

Required cases:

- recovery plan explicitly removes/revokes compromised key/root;
- new trust epoch/root bound into new genesis;
- old compromised key cannot sign valid new positive authority;
- historical old signature remains auditable but non-authorizing;
- recovery quorum cannot consist solely of the compromised domain unless that is explicitly outside the guarantee;
- trust snapshot sequence/epoch cannot be rolled back after recovery.

Threat coverage: T11, T18, T30.

---

## 11. Crash/transaction tests — GR-FAULT

Once durable storage exists, inject crashes at:

- before governance command append;
- after append before state projection;
- after state projection before acknowledgement;
- during fork declaration;
- during recovery evidence append;
- after new genesis append before acknowledgement;
- during checkpoint update.

Required properties:

- replay produces one semantic mutation or explicit non-operational corruption state;
- no partially applied recovery results in active positive authority;
- retry with same command/recovery ID is idempotent;
- stale competing command cannot apply after a successful durable successor;
- old epoch cannot become current after new-epoch durable commit.

---

## 12. Property/state-machine suite — GR-PROP

Generate bounded histories containing:

- ordinary descendant authority decisions;
- quarantine/revocation commands;
- policy changes;
- grant/key revocation;
- trust rotation;
- stale command replay;
- command substitution;
- fork creation/detection;
- recovery proposals/quorums;
- fresh epoch transition.

Global properties:

1. replication authorization cannot invoke governance without verified governance capability;
2. negative-governance capability cannot mint positive grant;
3. ordinary governance capability cannot recover;
4. committed negative state blocks positive authority until explicitly governed recovery semantics apply;
5. forked state cannot authorize;
6. one governance command cannot apply twice;
7. stale/wrong-target command leaves state unchanged;
8. recovery always creates fresh epoch;
9. no positive grant/token crosses epoch implicitly;
10. old incident/fork evidence remains bound and auditable.

---

## 13. API compile/review tests — GR-API

Establish where practical:

- direct production governance entry point requires `VerifiedGovernanceCommand` or equivalent;
- direct reference ledger methods are not exposed as untrusted production API;
- verified governance command cannot be constructed/deserialized directly;
- recovery entry point accepts only verified recovery capability, not ordinary governance command;
- grant/bounded-replication token types do not implement recovery-authority conversion;
- monitor capability has no grant-mint/widen/recovery methods;
- role capability interfaces are least-privilege.

---

## 14. Transparency tests — GR-TRANS

- publish valid governance/recovery statement and receipt;
- receipt verifies inclusion but cannot execute command by itself;
- forged receipt rejected;
- valid receipt + invalid role/signature/head still rejects governance action;
- contradictory recovery/fork statements surface incident evidence;
- transparency outage does not relax recovery threshold or grant carry-over rule.

---

## 15. Evidence subject

Every executed Class A governance/recovery test intended as production evidence binds:

- exact Git commit/tree;
- Cargo.lock/toolchain/Nix environment;
- runtime/config/features;
- trust/policy/recovery fixture digests;
- durable-store profile;
- test seeds/fault schedule;
- artifact/runtime digest where applicable.

---

## 16. Promotion rule

No production governance/recovery admission until green executed evidence exists for applicable:

- GR-CMD;
- GR-ROLE;
- GR-NEG;
- GR-REPLAY;
- GR-FORK;
- GR-REC-PROP;
- GR-REC-QUORUM;
- GR-EPOCH;
- GR-TRUST-REC;
- GR-FAULT;
- GR-PROP;
- GR-API;
- GR-TRANS.

Authored tests/specs are not executed evidence.

**Production admission remains DENIED / NOT YET ELIGIBLE.**
