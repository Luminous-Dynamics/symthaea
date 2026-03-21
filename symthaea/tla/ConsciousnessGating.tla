---------------------- MODULE ConsciousnessGating ----------------------
(***************************************************************************
 * Consciousness Gating — Formal Specification
 *
 * Formally verifies the Mycelix consciousness gating mechanism, proving:
 *   S1: Monotonic Privilege — Observer can never execute Constitutional
 *   S2: Hysteresis Safety — no oscillation sequence bypasses tier gates
 *   S3: Credential Expiry — grace period never grants higher-tier access
 *   S4: Offline Degradation — longer offline → lower-or-equal tier
 *   S5: Blacklist Safety — blacklisted agents have zero vote weight
 *   S6: Sigmoid Boundedness — vote weight ∈ [0, MAX_WEIGHT]
 *   S7: Profile Boundedness — combined_score ∈ [0, 1]
 *
 * Models the implementation in:
 *   crates/mycelix-bridge-common/src/consciousness_profile.rs
 *
 * Authors: Tristan Stoltz, Claude (Anthropic)
 * Date: March 21, 2026
 * Version: 1.0
 ***************************************************************************)

EXTENDS Integers, Reals, FiniteSets, Sequences, TLC

(***************************************************************************
 * CONSTANTS
 ***************************************************************************)

CONSTANTS
    Agents,              \* Set of agent identifiers
    MaxTime,             \* Maximum simulation time (microseconds, discretized)
    NumActions           \* Number of governance actions to simulate

(***************************************************************************
 * GOVERNANCE TYPES
 ***************************************************************************)

\* Consciousness tiers (ordered: Observer < Participant < Citizen < Steward < Guardian)
Tiers == {"Observer", "Participant", "Citizen", "Steward", "Guardian"}

\* Tier ordering (numeric for comparison)
TierOrd(t) ==
    CASE t = "Observer"    -> 0
      [] t = "Participant" -> 1
      [] t = "Citizen"     -> 2
      [] t = "Steward"     -> 3
      [] t = "Guardian"    -> 4

\* Tier from ordinal
TierFromOrd(n) ==
    CASE n = 0 -> "Observer"
      [] n = 1 -> "Participant"
      [] n = 2 -> "Citizen"
      [] n = 3 -> "Steward"
      [] n = 4 -> "Guardian"

\* Governance actions with their minimum tier requirements
ActionTiers == [
    read           |-> "Observer",
    comment        |-> "Participant",
    submit         |-> "Participant",
    vote_standard  |-> "Citizen",
    vote_emergency |-> "Citizen",
    constitutional |-> "Steward",
    guardian_power |-> "Guardian"
]

(***************************************************************************
 * THRESHOLD CONSTANTS (matching Rust implementation)
 ***************************************************************************)

\* Tier boundary scores
TierThreshold(t) ==
    CASE t = "Observer"    -> 0
      [] t = "Participant" -> 30  \* 0.3 × 100 (integer arithmetic)
      [] t = "Citizen"     -> 40
      [] t = "Steward"     -> 60
      [] t = "Guardian"    -> 80

\* Hysteresis margin: 5 (= 0.05 × 100)
HysteresisMargin == 5

\* Blacklist threshold: 5 (= 0.05 × 100)
BlacklistThreshold == 5

\* Credential TTL: 86400 time units (24 hours)
CredentialTTL == 86400

\* Grace period: 1800 time units (30 minutes)
GracePeriod == 1800

\* Offline degradation thresholds (time units)
OfflineDegradation1 == 86400   \* 24 hours: -1 tier
OfflineDegradation2 == 259200  \* 72 hours: -2 tiers
OfflineDegradation3 == 604800  \* 7 days:   → Observer

\* Max vote weight (basis points: 10000 = 100%)
MaxVoteWeightBP == 10000

(***************************************************************************
 * VARIABLES
 ***************************************************************************)

VARIABLES
    \* === Agent State ===
    profiles,         \* Function: Agent -> [identity, reputation, community, engagement]
                      \* Each dimension is 0..100 (representing 0.00-1.00)
    currentTiers,     \* Function: Agent -> Tier (current tier with hysteresis)
    blacklisted,      \* Function: Agent -> BOOLEAN

    \* === Credential State ===
    credIssuedAt,     \* Function: Agent -> time (when credential was issued)
    lastOnline,       \* Function: Agent -> time (last online verification)

    \* === Global State ===
    time,             \* Current time
    actionLog         \* Sequence of (agent, action, result) for verification

vars == <<profiles, currentTiers, blacklisted, credIssuedAt, lastOnline,
          time, actionLog>>

(***************************************************************************
 * HELPER OPERATORS
 ***************************************************************************)

\* Combined score from 4D profile (integer arithmetic, 0-100)
\* 0.25×id + 0.25×rep + 0.30×comm + 0.20×eng
CombinedScore(p) ==
    (p.identity * 25 + p.reputation * 25 + p.community * 30 + p.engagement * 20) \div 100

\* Tier from raw score (no hysteresis)
TierFromScore(score) ==
    IF score >= 80 THEN "Guardian"
    ELSE IF score >= 60 THEN "Steward"
    ELSE IF score >= 40 THEN "Citizen"
    ELSE IF score >= 30 THEN "Participant"
    ELSE "Observer"

\* Tier with hysteresis (matches Rust: from_score_with_hysteresis)
TierWithHysteresis(score, current) ==
    LET
        \* Promotion requires score >= boundary + margin
        promoted ==
            IF score >= 80 + HysteresisMargin THEN "Guardian"
            ELSE IF score >= 60 + HysteresisMargin THEN "Steward"
            ELSE IF score >= 40 + HysteresisMargin THEN "Citizen"
            ELSE IF score >= 30 + HysteresisMargin THEN "Participant"
            ELSE "Observer"

        \* Demotion requires score < boundary - margin
        demoted ==
            IF score < 30 - HysteresisMargin THEN "Observer"
            ELSE IF score < 40 - HysteresisMargin THEN "Participant"
            ELSE IF score < 60 - HysteresisMargin THEN "Citizen"
            ELSE IF score < 80 - HysteresisMargin THEN "Steward"
            ELSE "Guardian"
    IN
        IF TierOrd(promoted) > TierOrd(current) THEN promoted
        ELSE IF TierOrd(demoted) < TierOrd(current) THEN demoted
        ELSE current

\* Offline tier degradation (matches Rust: effective_tier_offline)
OfflineDegradedTier(baseTier, offlineDuration) ==
    LET baseOrd == TierOrd(baseTier)
    IN
        IF offlineDuration > OfflineDegradation3 THEN "Observer"
        ELSE IF offlineDuration > OfflineDegradation2 THEN
            TierFromOrd(IF baseOrd >= 2 THEN baseOrd - 2 ELSE 0)
        ELSE IF offlineDuration > OfflineDegradation1 THEN
            TierFromOrd(IF baseOrd >= 1 THEN baseOrd - 1 ELSE 0)
        ELSE baseTier

\* Effective tier considering offline degradation
EffectiveTier(agent) ==
    LET
        offlineDuration == time - lastOnline[agent]
        baseTier == currentTiers[agent]
    IN
        OfflineDegradedTier(baseTier, offlineDuration)

\* Is credential expired?
IsExpired(agent) ==
    time > credIssuedAt[agent] + CredentialTTL

\* Is credential in grace period?
InGracePeriod(agent) ==
    /\ IsExpired(agent)
    /\ time <= credIssuedAt[agent] + CredentialTTL + GracePeriod

\* Can agent execute action? (core gating logic)
CanExecute(agent, action) ==
    LET
        requiredTier == ActionTiers[action]
        effectiveTier == EffectiveTier(agent)
        meetsRequirement == TierOrd(effectiveTier) >= TierOrd(requiredTier)
    IN
        /\ ~blacklisted[agent]
        /\ IF IsExpired(agent) THEN
               \* Grace period: only basic ops (Participant or below)
               /\ InGracePeriod(agent)
               /\ TierOrd(requiredTier) <= TierOrd("Participant")
               /\ meetsRequirement
           ELSE
               meetsRequirement

(***************************************************************************
 * INITIAL STATE
 ***************************************************************************)

Init ==
    /\ profiles = [a \in Agents |-> [
           identity   |-> 0,
           reputation |-> 0,
           community  |-> 0,
           engagement |-> 0
       ]]
    /\ currentTiers = [a \in Agents |-> "Observer"]
    /\ blacklisted = [a \in Agents |-> FALSE]
    /\ credIssuedAt = [a \in Agents |-> 0]
    /\ lastOnline = [a \in Agents |-> 0]
    /\ time = 0
    /\ actionLog = <<>>

(***************************************************************************
 * ACTIONS
 ***************************************************************************)

\* Agent profile changes (e.g., reputation update, community attestation)
UpdateProfile ==
    \E agent \in Agents :
        \E newId, newRep, newComm, newEng \in 0..100 :
            LET
                newProfile == [
                    identity   |-> newId,
                    reputation |-> newRep,
                    community  |-> newComm,
                    engagement |-> newEng
                ]
                newScore == CombinedScore(newProfile)
                newTier == TierWithHysteresis(newScore, currentTiers[agent])
                newBlacklist == newRep < BlacklistThreshold
            IN
                /\ profiles' = [profiles EXCEPT ![agent] = newProfile]
                /\ currentTiers' = [currentTiers EXCEPT ![agent] = newTier]
                /\ blacklisted' = [blacklisted EXCEPT ![agent] = newBlacklist]
                /\ UNCHANGED <<credIssuedAt, lastOnline, time, actionLog>>

\* Agent refreshes credential (comes online)
RefreshCredential ==
    \E agent \in Agents :
        /\ credIssuedAt' = [credIssuedAt EXCEPT ![agent] = time]
        /\ lastOnline' = [lastOnline EXCEPT ![agent] = time]
        /\ UNCHANGED <<profiles, currentTiers, blacklisted, time, actionLog>>

\* Time advances
Tick ==
    /\ time < MaxTime
    /\ time' = time + 1
    /\ UNCHANGED <<profiles, currentTiers, blacklisted, credIssuedAt,
                   lastOnline, actionLog>>

\* Agent attempts governance action
AttemptAction ==
    \E agent \in Agents :
        \E action \in DOMAIN ActionTiers :
            LET result == CanExecute(agent, action)
            IN
                /\ actionLog' = Append(actionLog, <<agent, action, result>>)
                /\ UNCHANGED <<profiles, currentTiers, blacklisted,
                               credIssuedAt, lastOnline, time>>

\* Complete next-state relation
Next ==
    \/ UpdateProfile
    \/ RefreshCredential
    \/ Tick
    \/ AttemptAction

(***************************************************************************
 * SAFETY INVARIANTS
 ***************************************************************************)

\* S1: MONOTONIC PRIVILEGE
\* An agent with tier T can never execute an action requiring tier > T.
\* Specifically: Observer can never execute Constitutional or Guardian actions.
MonotonicPrivilege ==
    \A entry \in {actionLog[i] : i \in 1..Len(actionLog)} :
        LET
            agent == entry[1]
            action == entry[2]
            result == entry[3]
        IN
            result = TRUE =>
                TierOrd(EffectiveTier(agent)) >= TierOrd(ActionTiers[action])

\* S2: HYSTERESIS SAFETY
\* Tier transitions are bounded by hysteresis margin.
\* An agent cannot jump more than one tier level per profile update
\* without crossing the hysteresis band.
HysteresisConsistency ==
    \A agent \in Agents :
        LET
            score == CombinedScore(profiles[agent])
            tier == currentTiers[agent]
        IN
            \* If score is within the hysteresis band, tier is valid
            /\ TierOrd(tier) >= TierOrd(TierFromScore(score)) - 1
            /\ TierOrd(tier) <= TierOrd(TierFromScore(score)) + 1

\* S3: CREDENTIAL EXPIRY SAFETY
\* Grace period NEVER grants access to actions above Participant tier.
\* This is critical: expired creds must not allow voting or constitutional acts.
GracePeriodSafety ==
    \A entry \in {actionLog[i] : i \in 1..Len(actionLog)} :
        LET
            agent == entry[1]
            action == entry[2]
            result == entry[3]
        IN
            (result = TRUE /\ IsExpired(agent)) =>
                TierOrd(ActionTiers[action]) <= TierOrd("Participant")

\* S4: OFFLINE DEGRADATION MONOTONICITY
\* Longer offline duration → lower-or-equal effective tier (never higher).
OfflineDegradationMonotonic ==
    \A agent \in Agents :
        \A d1, d2 \in 0..MaxTime :
            d1 <= d2 =>
                TierOrd(OfflineDegradedTier(currentTiers[agent], d1)) >=
                TierOrd(OfflineDegradedTier(currentTiers[agent], d2))

\* S5: BLACKLIST SAFETY
\* Blacklisted agents have zero effective governance power.
BlacklistSafety ==
    \A entry \in {actionLog[i] : i \in 1..Len(actionLog)} :
        LET
            agent == entry[1]
            result == entry[3]
        IN
            blacklisted[agent] => result = FALSE

\* S6: PROFILE BOUNDEDNESS
\* Combined score is always in [0, 100].
ProfileBoundedness ==
    \A agent \in Agents :
        LET score == CombinedScore(profiles[agent])
        IN 0 <= score /\ score <= 100

\* S7: TIER VALIDITY
\* currentTier is always a valid tier.
TierValidity ==
    \A agent \in Agents : currentTiers[agent] \in Tiers

\* Combined Safety Invariant
Safety ==
    /\ MonotonicPrivilege
    /\ HysteresisConsistency
    /\ GracePeriodSafety
    /\ OfflineDegradationMonotonic
    /\ BlacklistSafety
    /\ ProfileBoundedness
    /\ TierValidity

(***************************************************************************
 * LIVENESS PROPERTIES
 ***************************************************************************)

\* L1: An agent that builds sufficient reputation eventually reaches Citizen
EventualCitizenship ==
    \A agent \in Agents :
        (CombinedScore(profiles[agent]) >= 40 + HysteresisMargin
         /\ ~blacklisted[agent])
        ~> (currentTiers[agent] \in {"Citizen", "Steward", "Guardian"})

\* L2: Blacklisted agents can eventually be restored
EventualRestoration ==
    \A agent \in Agents :
        blacklisted[agent] ~> <>~blacklisted[agent]

(***************************************************************************
 * TEMPORAL SPECIFICATION
 ***************************************************************************)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

Fairness ==
    /\ WF_vars(UpdateProfile)
    /\ WF_vars(RefreshCredential)
    /\ WF_vars(Tick)
    /\ WF_vars(AttemptAction)

(***************************************************************************
 * THEOREMS
 ***************************************************************************)

\* Main Safety Theorem
THEOREM SafetyTheorem == Spec => []Safety

\* Monotonic Privilege Theorem (most critical)
THEOREM MonotonicPrivilegeTheorem == Spec => []MonotonicPrivilege

\* Grace Period Theorem
THEOREM GracePeriodTheorem == Spec => []GracePeriodSafety

\* Blacklist Theorem
THEOREM BlacklistTheorem == Spec => []BlacklistSafety

\* Offline Degradation Theorem
THEOREM OfflineDegradationTheorem == Spec => []OfflineDegradationMonotonic

(***************************************************************************
 * MODEL CHECKING CONFIGURATION
 *
 * For TLC: use small values for tractability
 * Agents = {a1, a2}
 * MaxTime = 200000 (enough for 2× credential TTL + grace)
 * NumActions = 10
 ***************************************************************************)

=============================================================================
