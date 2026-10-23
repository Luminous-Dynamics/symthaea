-------------------- MODULE MESH_NONCE_LIFECYCLE --------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
  SenderIDs,
  Prefixes,
  EffectiveKeys,
  PayloadTypes,
  FragmentIndices,
  EpochValues,
  DefaultEpoch,
  MaxSeq,
  FragmentModulus,
  Variant,
  StrictNoReuse

ASSUME
  /\ SenderIDs # {}
  /\ Prefixes # {}
  /\ EffectiveKeys # {}
  /\ PayloadTypes # {}
  /\ FragmentIndices # {}
  /\ EpochValues # {}
  /\ DefaultEpoch \in EpochValues
  /\ MaxSeq \in Nat
  /\ MaxSeq > 0
  /\ FragmentModulus \in Nat
  /\ FragmentModulus > 0
  /\ Variant \in {"packet", "fragment"}
  /\ StrictNoReuse \in BOOLEAN

VARIABLES nextSeq, epochs, usedPairs, reuseDetected

vars == <<nextSeq, epochs, usedPairs, reuseDetected>>

\* This suite deliberately configures a singleton prefix/key namespace: all
\* modeled senders share the same truncated packet prefix and effective key.
\* Per-sender unique key domains require a separate explicit configuration.
PrefixFor(sender) == CHOOSE prefix \in Prefixes : TRUE
KeyFor(sender) == CHOOSE key \in EffectiveKeys : TRUE

\* In packet mode, PrefixFor models the identity prefix retained by the nonce.
\* In fragment mode, FragmentModulus is deliberately small so TLC can
\* explore low-bit wrap quickly; it abstracts the actual 16-bit thought ID.
Nonce(sender, payloadType, fragmentIndex) ==
  IF Variant = "packet"
  THEN <<PrefixFor(sender), payloadType, epochs[sender], nextSeq[sender]>>
  ELSE <<sender, nextSeq[sender] % FragmentModulus, fragmentIndex, 0>>

Init ==
  /\ nextSeq = [sender \in SenderIDs |-> 0]
  /\ epochs = [sender \in SenderIDs |-> DefaultEpoch]
  /\ usedPairs = {}
  /\ reuseDetected = FALSE

Send(sender, payloadType, fragmentIndex) ==
  LET pair == <<KeyFor(sender),
                Nonce(sender, payloadType, fragmentIndex)>>
  IN
    /\ (~StrictNoReuse \/ pair \notin usedPairs)
    /\ usedPairs' = usedPairs \cup {pair}
    /\ reuseDetected' = (reuseDetected \/ (pair \in usedPairs))
    /\ nextSeq' =
         [nextSeq EXCEPT
           ![sender] = IF @ = MaxSeq THEN 0 ELSE @ + 1]
    /\ UNCHANGED epochs

\* This models the observed deterministic restart default. It does not model
\* operating-system randomness, durable counter reservation, or key custody.
Restart(sender) ==
  /\ nextSeq' = [nextSeq EXCEPT ![sender] = 0]
  /\ epochs' = [epochs EXCEPT ![sender] = DefaultEpoch]
  /\ UNCHANGED <<usedPairs, reuseDetected>>

\* Model an externally selected/random restart epoch as nondeterministic. TLC
\* must explore collisions too; this does not estimate their probability.
RestartWithEpoch(sender, epoch) ==
  /\ epoch \in EpochValues
  /\ nextSeq' = [nextSeq EXCEPT ![sender] = 0]
  /\ epochs' = [epochs EXCEPT ![sender] = epoch]
  /\ UNCHANGED <<usedPairs, reuseDetected>>

\* Rotation may choose any configured epoch, including a previously used one.
\* Epoch reuse alone is not necessarily a collision: the strict property is
\* uniqueness of each complete (effective-key, nonce) pair. This model keeps
\* the effective key fixed so reset/wrap counterexamples remain observable.
RotateEpoch(sender, epoch) ==
  /\ epoch \in EpochValues
  /\ epochs' = [epochs EXCEPT ![sender] = epoch]
  /\ UNCHANGED <<nextSeq, usedPairs, reuseDetected>>

Next ==
  \/ \E sender \in SenderIDs :
       \E payloadType \in PayloadTypes :
         \E fragmentIndex \in FragmentIndices :
           Send(sender, payloadType, fragmentIndex)
  \/ \E sender \in SenderIDs : Restart(sender)
  \/ \E sender \in SenderIDs :
       \E epoch \in EpochValues : RestartWithEpoch(sender, epoch)
  \/ \E sender \in SenderIDs :
       \E epoch \in EpochValues : RotateEpoch(sender, epoch)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ nextSeq \in [SenderIDs -> 0..MaxSeq]
  /\ epochs \in [SenderIDs -> EpochValues]
  /\ reuseDetected \in BOOLEAN

NoNonceReuse == reuseDetected = FALSE

=============================================================================
