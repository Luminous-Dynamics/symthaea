---------------------- MODULE ConsciousnessGating_MC ----------------------
(***************************************************************************
 * Consciousness Gating — Model Checking Version
 *
 * Reduced state space for TLC tractability:
 * - Profile dimensions: 0..10 (instead of 0..100)
 * - Tier boundaries scaled: 3, 4, 6, 8 (instead of 30, 40, 60, 80)
 * - Hysteresis margin: 1 (instead of 5)
 * - Time: 0..20 (instead of 200000)
 *
 * All safety invariants preserved under scaling.
 *
 * Authors: Tristan Stoltz, Claude (Anthropic)
 * Date: March 21, 2026
 ***************************************************************************)

EXTENDS Integers, FiniteSets, Sequences, TLC

CONSTANTS
    Agents,
    MaxTime

\* Scaled constants
HysteresisMargin == 1
BlacklistThreshold == 1
CredentialTTL == 10
GracePeriod == 3
OfflineDeg1 == 10
OfflineDeg2 == 15
OfflineDeg3 == 20
MaxDim == 10

\* Tier boundaries (scaled from 0.3/0.4/0.6/0.8 to 3/4/6/8 out of 10)
TierFromScore(score) ==
    IF score >= 8 THEN 4      \* Guardian
    ELSE IF score >= 6 THEN 3  \* Steward
    ELSE IF score >= 4 THEN 2  \* Citizen
    ELSE IF score >= 3 THEN 1  \* Participant
    ELSE 0                     \* Observer

\* Action tier requirements
ActionReq(action) ==
    CASE action = "read"     -> 0
      [] action = "comment"  -> 1
      [] action = "vote"     -> 2
      [] action = "constit"  -> 3
      [] action = "guardian" -> 4

\* Hysteresis tier computation
TierWithHyst(score, current) ==
    LET
        promoted ==
            IF score >= 8 + HysteresisMargin THEN 4
            ELSE IF score >= 6 + HysteresisMargin THEN 3
            ELSE IF score >= 4 + HysteresisMargin THEN 2
            ELSE IF score >= 3 + HysteresisMargin THEN 1
            ELSE 0
        demoted ==
            IF score < 3 - HysteresisMargin THEN 0
            ELSE IF score < 4 - HysteresisMargin THEN 1
            ELSE IF score < 6 - HysteresisMargin THEN 2
            ELSE IF score < 8 - HysteresisMargin THEN 3
            ELSE 4
    IN
        IF promoted > current THEN promoted
        ELSE IF demoted < current THEN demoted
        ELSE current

\* Offline tier degradation
OfflineTier(baseTier, offDur) ==
    IF offDur > OfflineDeg3 THEN 0
    ELSE IF offDur > OfflineDeg2 THEN
        IF baseTier >= 2 THEN baseTier - 2 ELSE 0
    ELSE IF offDur > OfflineDeg1 THEN
        IF baseTier >= 1 THEN baseTier - 1 ELSE 0
    ELSE baseTier

\* Combined score: (id*25 + rep*25 + comm*30 + eng*20) / 100
\* Scaled: with 0..10 dimensions, max = 10*25+10*25+10*30+10*20 = 1000
\* Divide by 100 → 0..10
CombScore(p) == (p[1]*25 + p[2]*25 + p[3]*30 + p[4]*20) \div 100

VARIABLES
    profiles,      \* Agent -> <<id, rep, comm, eng>> each 0..MaxDim
    tiers,         \* Agent -> 0..4
    blacklisted,   \* Agent -> BOOLEAN
    credAt,        \* Agent -> time
    lastOn,        \* Agent -> time
    time,
    log            \* Seq of <<agent, action, result>>

vars == <<profiles, tiers, blacklisted, credAt, lastOn, time, log>>

EffTier(a) == OfflineTier(tiers[a], time - lastOn[a])

IsExpired(a) == time > credAt[a] + CredentialTTL
InGrace(a) == IsExpired(a) /\ time <= credAt[a] + CredentialTTL + GracePeriod

CanExec(a, action) ==
    LET req == ActionReq(action)
        eff == EffTier(a)
    IN
        /\ ~blacklisted[a]
        /\ IF IsExpired(a) THEN
               InGrace(a) /\ req <= 1 /\ eff >= req
           ELSE
               eff >= req

Init ==
    /\ profiles = [a \in Agents |-> <<0, 0, 0, 0>>]
    /\ tiers = [a \in Agents |-> 0]
    /\ blacklisted = [a \in Agents |-> FALSE]
    /\ credAt = [a \in Agents |-> 0]
    /\ lastOn = [a \in Agents |-> 0]
    /\ time = 0
    /\ log = <<>>

UpdateProfile ==
    \E a \in Agents :
        \E id, rep, comm, eng \in 0..MaxDim :
            LET
                p == <<id, rep, comm, eng>>
                s == CombScore(p)
                t == TierWithHyst(s, tiers[a])
                bl == rep < BlacklistThreshold
            IN
                /\ profiles' = [profiles EXCEPT ![a] = p]
                /\ tiers' = [tiers EXCEPT ![a] = t]
                /\ blacklisted' = [blacklisted EXCEPT ![a] = bl]
                /\ UNCHANGED <<credAt, lastOn, time, log>>

Refresh ==
    \E a \in Agents :
        /\ credAt' = [credAt EXCEPT ![a] = time]
        /\ lastOn' = [lastOn EXCEPT ![a] = time]
        /\ UNCHANGED <<profiles, tiers, blacklisted, time, log>>

Tick ==
    /\ time < MaxTime
    /\ time' = time + 1
    /\ UNCHANGED <<profiles, tiers, blacklisted, credAt, lastOn, log>>

Actions == {"read", "comment", "vote", "constit", "guardian"}

Attempt ==
    \E a \in Agents :
        \E action \in Actions :
            /\ log' = Append(log, <<a, action, CanExec(a, action)>>)
            /\ UNCHANGED <<profiles, tiers, blacklisted, credAt, lastOn, time>>

Next == UpdateProfile \/ Refresh \/ Tick \/ Attempt

\* ========== SAFETY INVARIANTS ==========

\* S1: Monotonic Privilege — approved actions always have sufficient tier
MonotonicPrivilege ==
    \A i \in 1..Len(log) :
        log[i][3] = TRUE => EffTier(log[i][1]) >= ActionReq(log[i][2])

\* S3: Grace Period Safety — expired creds never grant above Participant
GracePeriodSafety ==
    \A i \in 1..Len(log) :
        (log[i][3] = TRUE /\ IsExpired(log[i][1])) =>
            ActionReq(log[i][2]) <= 1

\* S5: Blacklist Safety
BlacklistSafety ==
    \A i \in 1..Len(log) :
        blacklisted[log[i][1]] => log[i][3] = FALSE

\* S4: Offline Degradation Monotonicity
OfflineMono ==
    \A a \in Agents :
        \A d1, d2 \in 0..MaxTime :
            d1 <= d2 => OfflineTier(tiers[a], d1) >= OfflineTier(tiers[a], d2)

\* S6: Profile Boundedness
ProfileBound ==
    \A a \in Agents :
        LET s == CombScore(profiles[a])
        IN 0 <= s /\ s <= MaxDim

\* S7: Tier Validity
TierValid ==
    \A a \in Agents : tiers[a] \in 0..4

Safety ==
    /\ MonotonicPrivilege
    /\ GracePeriodSafety
    /\ BlacklistSafety
    /\ OfflineMono
    /\ ProfileBound
    /\ TierValid

Spec == Init /\ [][Next]_vars

=============================================================================
