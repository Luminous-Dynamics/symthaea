-------------------- MODULE MESH_CLONE_NONCE_LIFECYCLE --------------------
EXTENDS Naturals, Sequences, TLC

CONSTANTS
  MaxCounter,
  MaxGeneration,
  UseFencing,
  InstanceA,
  InstanceB,
  InitialEpoch

ASSUME
  /\ MaxCounter \in Nat
  /\ MaxCounter >= 1
  /\ MaxGeneration \in Nat
  /\ MaxGeneration >= 2
  /\ UseFencing \in BOOLEAN
  /\ InstanceA # InstanceB
  /\ InitialEpoch \in Nat

VARIABLES
  nextA,
  nextB,
  fence,
  owner,
  tokenA,
  tokenB,
  keyGenA,
  keyGenB,
  history

vars == <<nextA, nextB, fence, owner, tokenA, tokenB, keyGenA, keyGenB, history>>

\* keyGen is an abstract identifier for the effective traffic key, not a
\* wire field. Distinct generations are safe only if the implementation
\* actually derives distinct keys or allocates provably disjoint nonce space.
NonceA == <<keyGenA, InitialEpoch, nextA>>
NonceB == <<keyGenB, InitialEpoch, nextB>>

FreshA ==
  ~(\E i \in 1..Len(history) : history[i] = NonceA)

FreshB ==
  ~(\E i \in 1..Len(history) : history[i] = NonceB)

CanSendA ==
  ~UseFencing \/
  (owner = InstanceA /\ tokenA = fence /\ FreshA)

CanSendB ==
  ~UseFencing \/
  (owner = InstanceB /\ tokenB = fence /\ FreshB)

Init ==
  /\ nextA = 0
  /\ nextB = 0
  /\ fence = 1
  /\ owner = InstanceA
  /\ tokenA = 1
  /\ tokenB = 0
  /\ keyGenA = 1
  /\ keyGenB = 0
  /\ history = <<>>

\* Snapshot/clone B from A. The copy includes the counter, identity-bound
\* traffic-key generation, and local fencing token. In the legacy profile,
\* nothing outside the clone prevents both copies from emitting.
CloneSnapshot ==
  /\ nextB' = nextA
  /\ tokenB' = tokenA
  /\ keyGenB' = keyGenA
  /\ UNCHANGED <<nextA, fence, owner, tokenA, keyGenA, history>>

\* An authority that owns durable monotonic state grants a fresh fencing
\* generation and corresponding fresh effective traffic key. A stale clone
\* is unable to emit once another instance has acquired the lease.
AcquireA ==
  /\ UseFencing
  /\ fence < MaxGeneration
  /\ fence' = fence + 1
  /\ owner' = InstanceA
  /\ tokenA' = fence + 1
  /\ keyGenA' = fence + 1
  /\ nextA' = 0
  /\ UNCHANGED <<nextB, tokenB, keyGenB, history>>

AcquireB ==
  /\ UseFencing
  /\ fence < MaxGeneration
  /\ fence' = fence + 1
  /\ owner' = InstanceB
  /\ tokenB' = fence + 1
  /\ keyGenB' = fence + 1
  /\ nextB' = 0
  /\ UNCHANGED <<nextA, tokenA, keyGenA, history>>

SendA ==
  /\ CanSendA
  /\ history' = Append(history, NonceA)
  /\ nextA' = IF nextA = MaxCounter THEN 0 ELSE nextA + 1
  /\ UNCHANGED <<nextB, fence, owner, tokenA, tokenB, keyGenA, keyGenB>>

SendB ==
  /\ CanSendB
  /\ history' = Append(history, NonceB)
  /\ nextB' = IF nextB = MaxCounter THEN 0 ELSE nextB + 1
  /\ UNCHANGED <<nextA, fence, owner, tokenA, tokenB, keyGenA, keyGenB>>

Next ==
  \/ CloneSnapshot
  \/ AcquireA
  \/ AcquireB
  \/ SendA
  \/ SendB

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ nextA \in 0..MaxCounter
  /\ nextB \in 0..MaxCounter
  /\ fence \in 1..MaxGeneration
  /\ owner \in {InstanceA, InstanceB}
  /\ tokenA \in 0..MaxGeneration
  /\ tokenB \in 0..MaxGeneration
  /\ keyGenA \in 0..MaxGeneration
  /\ keyGenB \in 0..MaxGeneration
  /\ history \in Seq((0..MaxGeneration) \X {InitialEpoch} \X (0..MaxCounter))

\* Every emitted (effective-key generation, epoch, counter) tuple must be
\* unique. This is a bounded nonce-lifecycle property, not a cryptographic
\* proof that key generations map to different key bytes.
NoNonceReuse ==
  \A i \in 1..Len(history):
    \A j \in 1..Len(history):
      i < j => history[i] # history[j]

=============================================================================
