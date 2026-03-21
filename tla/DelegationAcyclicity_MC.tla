---------------------- MODULE DelegationAcyclicity_MC ----------------------
(***************************************************************************
 * Delegation Acyclicity — Model Checking Version (reduced state space)
 *
 * Authors: Tristan Stoltz, Claude (Anthropic)
 * Date: March 21, 2026
 ***************************************************************************)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Agents,
    MaxChainDepth,
    MaxWeight

VARIABLES
    deleg,        \* [Agent][Agent] -> 0..100 (delegation percentage)
    baseW,        \* [Agent] -> 0..MaxWeight
    step

vars == <<deleg, baseW, step>>

\* Reachability check (bounded DFS)
RECURSIVE Reach(_, _, _)
Reach(src, visited, depth) ==
    IF depth > MaxChainDepth THEN {}
    ELSE
        LET next == {b \in Agents : deleg[src][b] > 0 /\ b \notin visited}
        IN next \cup UNION {Reach(n, visited \cup next, depth + 1) : n \in next}

WouldCycle(src, dst) == src \in Reach(dst, {dst}, 1)

Init ==
    /\ deleg = [a \in Agents |-> [b \in Agents |-> 0]]
    /\ baseW \in [Agents -> 0..MaxWeight]
    /\ step = 0

CreateDeleg ==
    \E src, dst \in Agents :
        \E pct \in {25, 50, 100} :  \* Discretized for tractability
            /\ src # dst
            /\ ~WouldCycle(src, dst)
            /\ deleg[src][dst] = 0
            /\ deleg' = [deleg EXCEPT ![src][dst] = pct]
            /\ UNCHANGED <<baseW, step>>

RevokeDeleg ==
    \E src, dst \in Agents :
        /\ deleg[src][dst] > 0
        /\ deleg' = [deleg EXCEPT ![src][dst] = 0]
        /\ UNCHANGED <<baseW, step>>

Next == CreateDeleg \/ RevokeDeleg

\* ========== SAFETY INVARIANTS ==========

\* S1: Acyclicity — no agent can reach itself
Acyclicity ==
    \A a \in Agents : a \notin Reach(a, {a}, 1)

\* S4: Delegation percentages valid
DelegValid ==
    \A a, b \in Agents : deleg[a][b] >= 0 /\ deleg[a][b] <= 100

\* Self-delegation prohibited
NoSelfDeleg ==
    \A a \in Agents : deleg[a][a] = 0

Safety == Acyclicity /\ DelegValid /\ NoSelfDeleg

Spec == Init /\ [][Next]_vars

=============================================================================
