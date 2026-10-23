----------------------------- MODULE ReplicatorSafety -----------------------------
EXTENDS Naturals, FiniteSets, TLC

(***************************************************************************
RSK v0.1 abstract constitutional model.

This model deliberately contains no molecular, biological, or physical
replication mechanism. It models only the authority boundary around creation,
external grants, bounded budgets, monitoring, containment, and quarantine.
***************************************************************************)

CONSTANTS Units, ExternalIssuers, InitialUnits, MaxBudget, QuorumThreshold

ASSUME InitialUnits \subseteq Units
ASSUME Units \cap ExternalIssuers = {}
ASSUME MaxBudget \in Nat
ASSUME QuorumThreshold \in Nat
ASSUME QuorumThreshold > 0

VARIABLES
    existing,
    parentRel,
    authority,
    quarantined,
    lineageKnown,
    monitorHealthy,
    policyCurrent,
    environmentPermitted,
    budget,
    approvals,
    lastAction,
    lastCreated,
    lastGranted,
    previousAuthorityCount

vars == <<
    existing,
    parentRel,
    authority,
    quarantined,
    lineageKnown,
    monitorHealthy,
    policyCurrent,
    environmentPermitted,
    budget,
    approvals,
    lastAction,
    lastCreated,
    lastGranted,
    previousAuthorityCount
>>

TypeOK ==
    /\ existing \subseteq Units
    /\ parentRel \subseteq (Units \X Units)
    /\ authority \subseteq existing
    /\ quarantined \subseteq existing
    /\ lineageKnown \subseteq existing
    /\ monitorHealthy \subseteq Units
    /\ policyCurrent \subseteq Units
    /\ environmentPermitted \subseteq Units
    /\ budget \in [Units -> 0..MaxBudget]
    /\ approvals \in [Units -> SUBSET ExternalIssuers]
    /\ lastAction \in {"Init", "Approve", "Grant", "Create", "Quarantine",
                       "LoseMonitor", "InvalidateLineage", "ExpirePolicy",
                       "EnvironmentDrift"}
    /\ lastCreated \in Units
    /\ lastGranted \in Units
    /\ previousAuthorityCount \in Nat

Init ==
    /\ existing = InitialUnits
    /\ parentRel = {}
    /\ authority = {}
    /\ quarantined = {}
    /\ lineageKnown = InitialUnits
    /\ monitorHealthy = Units
    /\ policyCurrent = Units
    /\ environmentPermitted = Units
    /\ budget = [u \in Units |-> MaxBudget]
    /\ approvals = [u \in Units |-> {}]
    /\ lastAction = "Init"
    /\ lastCreated \in Units
    /\ lastGranted \in Units
    /\ previousAuthorityCount = 0

ExternalApprove(issuer, subject) ==
    /\ issuer \in ExternalIssuers
    /\ subject \in existing
    /\ approvals' = [approvals EXCEPT ![subject] = @ \cup {issuer}]
    /\ previousAuthorityCount' = Cardinality(authority)
    /\ lastAction' = "Approve"
    /\ UNCHANGED << existing, parentRel, authority, quarantined, lineageKnown,
                    monitorHealthy, policyCurrent, environmentPermitted, budget,
                    lastCreated, lastGranted >>

CanGrant(subject) ==
    /\ subject \in existing
    /\ subject \in lineageKnown
    /\ subject \in monitorHealthy
    /\ subject \in policyCurrent
    /\ subject \in environmentPermitted
    /\ subject \notin quarantined
    /\ budget[subject] > 0
    /\ Cardinality(approvals[subject]) >= QuorumThreshold

ExternalGrant(issuer, subject) ==
    /\ issuer \in ExternalIssuers
    /\ CanGrant(subject)
    /\ authority' = authority \cup {subject}
    /\ previousAuthorityCount' = Cardinality(authority)
    /\ lastAction' = "Grant"
    /\ lastGranted' = subject
    /\ UNCHANGED << existing, parentRel, quarantined, lineageKnown,
                    monitorHealthy, policyCurrent, environmentPermitted, budget,
                    approvals, lastCreated >>

CreateDescendant(parent, child) ==
    /\ parent \in authority
    /\ parent \in lineageKnown
    /\ parent \in monitorHealthy
    /\ parent \in policyCurrent
    /\ parent \in environmentPermitted
    /\ parent \notin quarantined
    /\ budget[parent] > 0
    /\ child \in Units \ existing
    /\ child # parent
    /\ existing' = existing \cup {child}
    /\ parentRel' = parentRel \cup {<<parent, child>>}
    /\ lineageKnown' = lineageKnown \cup {child}
    /\ budget' = [budget EXCEPT ![parent] = @ - 1]
    /\ previousAuthorityCount' = Cardinality(authority)
    /\ lastAction' = "Create"
    /\ lastCreated' = child
    /\ UNCHANGED << authority, quarantined, monitorHealthy, policyCurrent,
                    environmentPermitted, approvals, lastGranted >>

Quarantine(subject) ==
    /\ subject \in existing
    /\ quarantined' = quarantined \cup {subject}
    /\ authority' = authority \ {subject}
    /\ previousAuthorityCount' = Cardinality(authority)
    /\ lastAction' = "Quarantine"
    /\ UNCHANGED << existing, parentRel, lineageKnown, monitorHealthy,
                    policyCurrent, environmentPermitted, budget, approvals,
                    lastCreated, lastGranted >>

LoseMonitor(subject) ==
    /\ subject \in existing
    /\ monitorHealthy' = monitorHealthy \ {subject}
    /\ authority' = authority \ {subject}
    /\ previousAuthorityCount' = Cardinality(authority)
    /\ lastAction' = "LoseMonitor"
    /\ UNCHANGED << existing, parentRel, quarantined, lineageKnown,
                    policyCurrent, environmentPermitted, budget, approvals,
                    lastCreated, lastGranted >>

InvalidateLineage(subject) ==
    /\ subject \in existing
    /\ lineageKnown' = lineageKnown \ {subject}
    /\ authority' = authority \ {subject}
    /\ previousAuthorityCount' = Cardinality(authority)
    /\ lastAction' = "InvalidateLineage"
    /\ UNCHANGED << existing, parentRel, quarantined, monitorHealthy,
                    policyCurrent, environmentPermitted, budget, approvals,
                    lastCreated, lastGranted >>

ExpirePolicy(subject) ==
    /\ subject \in existing
    /\ policyCurrent' = policyCurrent \ {subject}
    /\ authority' = authority \ {subject}
    /\ previousAuthorityCount' = Cardinality(authority)
    /\ lastAction' = "ExpirePolicy"
    /\ UNCHANGED << existing, parentRel, quarantined, lineageKnown,
                    monitorHealthy, environmentPermitted, budget, approvals,
                    lastCreated, lastGranted >>

EnvironmentDrift(subject) ==
    /\ subject \in existing
    /\ environmentPermitted' = environmentPermitted \ {subject}
    /\ authority' = authority \ {subject}
    /\ previousAuthorityCount' = Cardinality(authority)
    /\ lastAction' = "EnvironmentDrift"
    /\ UNCHANGED << existing, parentRel, quarantined, lineageKnown,
                    monitorHealthy, policyCurrent, budget, approvals,
                    lastCreated, lastGranted >>

Next ==
    \/ \E issuer \in ExternalIssuers, subject \in Units:
         ExternalApprove(issuer, subject)
    \/ \E issuer \in ExternalIssuers, subject \in Units:
         ExternalGrant(issuer, subject)
    \/ \E parent \in Units, child \in Units:
         CreateDescendant(parent, child)
    \/ \E subject \in Units: Quarantine(subject)
    \/ \E subject \in Units: LoseMonitor(subject)
    \/ \E subject \in Units: InvalidateLineage(subject)
    \/ \E subject \in Units: ExpirePolicy(subject)
    \/ \E subject \in Units: EnvironmentDrift(subject)

Spec == Init /\ [][Next]_vars

(***************************************************************************
Constitutional invariants
***************************************************************************)

CreationNeverGrantsAuthority ==
    lastAction = "Create" => lastCreated \notin authority

CreationNeverAmplifiesAuthority ==
    lastAction = "Create" => Cardinality(authority) = previousAuthorityCount

QuarantineDominates ==
    quarantined \cap authority = {}

UnknownLineageCannotAuthorize ==
    authority \subseteq lineageKnown

MonitorFailureCannotAuthorize ==
    authority \subseteq monitorHealthy

ExpiredPolicyCannotAuthorize ==
    authority \subseteq policyCurrent

EnvironmentDriftCannotAuthorize ==
    authority \subseteq environmentPermitted

BudgetNeverNegative ==
    \A u \in Units: budget[u] >= 0

GrantSubjectExists ==
    lastAction = "Grant" => lastGranted \in existing

GrantRequiresQuorum ==
    lastAction = "Grant" => Cardinality(approvals[lastGranted]) >= QuorumThreshold

=============================================================================
