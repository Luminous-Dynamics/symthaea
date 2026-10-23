-------------------- MODULE MESH_REPLAY_EFFECT_LIFECYCLE --------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
  PacketSeq,
  Mode

ASSUME
  /\ PacketSeq \in Nat
  /\ Mode \in {"effect-first", "commit-first", "atomic"}

VARIABLES
  durableSeen,
  effectCount,
  authenticated,
  volatileEffectDone,
  crashed

vars == <<durableSeen, effectCount, authenticated, volatileEffectDone, crashed>>

Init ==
  /\ durableSeen = {}
  /\ effectCount = 0
  /\ authenticated = FALSE
  /\ volatileEffectDone = FALSE
  /\ crashed = FALSE

Authenticate ==
  /\ ~crashed
  /\ ~authenticated
  /\ PacketSeq \notin durableSeen
  /\ authenticated' = TRUE
  /\ UNCHANGED <<durableSeen, effectCount, volatileEffectDone, crashed>>

\* Unsafe: the business effect is committed before the replay record.
EffectBeforeCommit ==
  /\ Mode = "effect-first"
  /\ ~crashed
  /\ authenticated
  /\ ~volatileEffectDone
  /\ effectCount < 2
  /\ effectCount' = effectCount + 1
  /\ volatileEffectDone' = TRUE
  /\ UNCHANGED <<durableSeen, authenticated, crashed>>

CommitAfterEffect ==
  /\ Mode = "effect-first"
  /\ ~crashed
  /\ authenticated
  /\ volatileEffectDone
  /\ PacketSeq \notin durableSeen
  /\ durableSeen' = durableSeen \cup {PacketSeq}
  /\ UNCHANGED <<effectCount, authenticated, volatileEffectDone, crashed>>

\* Unsafe: replay state is committed before the business effect.
CommitBeforeEffect ==
  /\ Mode = "commit-first"
  /\ ~crashed
  /\ authenticated
  /\ PacketSeq \notin durableSeen
  /\ durableSeen' = durableSeen \cup {PacketSeq}
  /\ UNCHANGED <<effectCount, authenticated, volatileEffectDone, crashed>>

EffectAfterCommit ==
  /\ Mode = "commit-first"
  /\ ~crashed
  /\ authenticated
  /\ PacketSeq \in durableSeen
  /\ ~volatileEffectDone
  /\ effectCount < 2
  /\ effectCount' = effectCount + 1
  /\ volatileEffectDone' = TRUE
  /\ UNCHANGED <<durableSeen, authenticated, crashed>>

\* Reference design: replay commit and idempotent/durable effect are one
\* atomic state transition. This is an abstraction; real implementations need
\* a transaction, idempotency key, or an outbox/inbox protocol.
AtomicCommitAndEffect ==
  /\ Mode = "atomic"
  /\ ~crashed
  /\ authenticated
  /\ PacketSeq \notin durableSeen
  /\ effectCount < 2
  /\ durableSeen' = durableSeen \cup {PacketSeq}
  /\ effectCount' = effectCount + 1
  /\ volatileEffectDone' = TRUE
  /\ UNCHANGED <<authenticated, crashed>>

Crash ==
  /\ ~crashed
  /\ crashed' = TRUE
  /\ authenticated' = FALSE
  /\ volatileEffectDone' = FALSE
  /\ UNCHANGED <<durableSeen, effectCount>>

Restart ==
  /\ crashed
  /\ crashed' = FALSE
  /\ authenticated' = FALSE
  /\ volatileEffectDone' = FALSE
  /\ UNCHANGED <<durableSeen, effectCount>>

Next ==
  \/ Authenticate
  \/ EffectBeforeCommit
  \/ CommitAfterEffect
  \/ CommitBeforeEffect
  \/ EffectAfterCommit
  \/ AtomicCommitAndEffect
  \/ Crash
  \/ Restart

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ durableSeen \subseteq {PacketSeq}
  /\ effectCount \in 0..2
  /\ authenticated \in BOOLEAN
  /\ volatileEffectDone \in BOOLEAN
  /\ crashed \in BOOLEAN

NoDuplicateEffect ==
  effectCount <= 1

NoLostCommittedEffect ==
  PacketSeq \notin durableSeen \/ effectCount > 0

=============================================================================
