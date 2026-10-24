---------------- MODULE MESH_CROSS_API_NONCE_LIFECYCLE ----------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
  PacketKey,
  FragmentKey,
  S1, S2, S3, S4, S5, S6, S7, S8,
  PacketType,
  Epoch,
  SeqB0, SeqB1, SeqB2, SeqB3,
  ThoughtLo, ThoughtHi,
  FragmentIndex,
  Reserved

VARIABLES sentChannels, sentPairs, reuseDetected

vars == <<sentChannels, sentPairs, reuseDetected>>

\* These two vectors represent the actual 12-byte layouts of the legacy
\* packet and fragment helpers. The configuration supplies matching byte
\* values; keys are separate abstract effective-key identifiers.
PacketNonce ==
  <<S1, S2, S3, S4, S5, S6,
    PacketType, Epoch, SeqB0, SeqB1, SeqB2, SeqB3>>

FragmentNonce ==
  <<S1, S2, S3, S4, S5, S6, S7, S8,
    ThoughtLo, ThoughtHi, FragmentIndex, Reserved>>

PacketPair == <<PacketKey, PacketNonce>>
FragmentPair == <<FragmentKey, FragmentNonce>>

\* These predicates guard against accidentally weakening the test setup.
NonceLayoutsOverlap == PacketNonce = FragmentNonce
PurposeKeysDistinct == PacketKey # FragmentKey

Init ==
  /\ sentChannels = {}
  /\ sentPairs = {}
  /\ reuseDetected = FALSE

SendPacket ==
  /\ "packet" \notin sentChannels
  /\ sentPairs' = sentPairs \cup {PacketPair}
  /\ reuseDetected' = reuseDetected \/ (PacketPair \in sentPairs)
  /\ sentChannels' = sentChannels \cup {"packet"}

SendFragment ==
  /\ "fragment" \notin sentChannels
  /\ sentPairs' = sentPairs \cup {FragmentPair}
  /\ reuseDetected' = reuseDetected \/ (FragmentPair \in sentPairs)
  /\ sentChannels' = sentChannels \cup {"fragment"}

\* Both helpers can be invoked once in either order. Once both have been
\* exercised, stuttering is a valid terminal state rather than a deadlock.
Terminal ==
  /\ sentChannels = {"packet", "fragment"}
  /\ UNCHANGED vars

Next == SendPacket \/ SendFragment \/ Terminal

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ sentChannels \subseteq {"packet", "fragment"}
  /\ sentPairs \subseteq {PacketPair, FragmentPair}
  /\ reuseDetected \in BOOLEAN

NoNonceReuse == reuseDetected = FALSE

=============================================================================
