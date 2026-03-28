---- MODULE DelegationAcyclicity_TTrace_1774058958 ----
EXTENDS Sequences, TLCExt, DelegationAcyclicity_TEConstants, Toolbox, Naturals, TLC, DelegationAcyclicity

_expression ==
    LET DelegationAcyclicity_TEExpression == INSTANCE DelegationAcyclicity_TEExpression
    IN DelegationAcyclicity_TEExpression!expression
----

_trace ==
    LET DelegationAcyclicity_TETrace == INSTANCE DelegationAcyclicity_TETrace
    IN DelegationAcyclicity_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        effectiveWeights = ((a1 :> 3 @@ a2 :> 30 @@ a3 :> 77))
        /\
        step = (0)
        /\
        decayRate = (10)
        /\
        delegations = ((a1 :> (a1 :> 0 @@ a2 :> 0 @@ a3 :> 0) @@ a2 :> (a1 :> 43 @@ a2 :> 0 @@ a3 :> 0) @@ a3 :> (a1 :> 0 @@ a2 :> 0 @@ a3 :> 0)))
        /\
        baseWeights = ((a1 :> 3 @@ a2 :> 30 @@ a3 :> 77))
    )
----

_init ==
    /\ effectiveWeights = _TETrace[1].effectiveWeights
    /\ step = _TETrace[1].step
    /\ decayRate = _TETrace[1].decayRate
    /\ delegations = _TETrace[1].delegations
    /\ baseWeights = _TETrace[1].baseWeights
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ effectiveWeights  = _TETrace[i].effectiveWeights
        /\ effectiveWeights' = _TETrace[j].effectiveWeights
        /\ step  = _TETrace[i].step
        /\ step' = _TETrace[j].step
        /\ decayRate  = _TETrace[i].decayRate
        /\ decayRate' = _TETrace[j].decayRate
        /\ delegations  = _TETrace[i].delegations
        /\ delegations' = _TETrace[j].delegations
        /\ baseWeights  = _TETrace[i].baseWeights
        /\ baseWeights' = _TETrace[j].baseWeights

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("DelegationAcyclicity_TTrace_1774058958.json", _TETrace)

=============================================================================

 Note that you can extract this module `DelegationAcyclicity_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `DelegationAcyclicity_TEExpression.tla` file takes precedence 
  over the module `DelegationAcyclicity_TEExpression` below).

---- MODULE DelegationAcyclicity_TEExpression ----
EXTENDS Sequences, TLCExt, DelegationAcyclicity_TEConstants, Toolbox, Naturals, TLC, DelegationAcyclicity

expression == 
    [
        \* To hide variables of the `DelegationAcyclicity` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        effectiveWeights |-> effectiveWeights
        ,step |-> step
        ,decayRate |-> decayRate
        ,delegations |-> delegations
        ,baseWeights |-> baseWeights
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_effectiveWeightsUnchanged |-> effectiveWeights = effectiveWeights'
        
        \* Format the `effectiveWeights` variable as Json value.
        \* ,_effectiveWeightsJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(effectiveWeights)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_effectiveWeightsModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].effectiveWeights # _TETrace[s-1].effectiveWeights
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE DelegationAcyclicity_TETrace ----
\*EXTENDS IOUtils, DelegationAcyclicity_TEConstants, TLC, DelegationAcyclicity
\*
\*trace == IODeserialize("DelegationAcyclicity_TTrace_1774058958.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE DelegationAcyclicity_TETrace ----
EXTENDS DelegationAcyclicity_TEConstants, TLC, DelegationAcyclicity

trace == 
    <<
    ([effectiveWeights |-> (a1 :> 3 @@ a2 :> 30 @@ a3 :> 77),step |-> 0,decayRate |-> 10,delegations |-> (a1 :> (a1 :> 0 @@ a2 :> 0 @@ a3 :> 0) @@ a2 :> (a1 :> 0 @@ a2 :> 0 @@ a3 :> 0) @@ a3 :> (a1 :> 0 @@ a2 :> 0 @@ a3 :> 0)),baseWeights |-> (a1 :> 3 @@ a2 :> 30 @@ a3 :> 77)]),
    ([effectiveWeights |-> (a1 :> 3 @@ a2 :> 30 @@ a3 :> 77),step |-> 0,decayRate |-> 10,delegations |-> (a1 :> (a1 :> 0 @@ a2 :> 0 @@ a3 :> 0) @@ a2 :> (a1 :> 43 @@ a2 :> 0 @@ a3 :> 0) @@ a3 :> (a1 :> 0 @@ a2 :> 0 @@ a3 :> 0)),baseWeights |-> (a1 :> 3 @@ a2 :> 30 @@ a3 :> 77)])
    >>
----


=============================================================================

---- MODULE DelegationAcyclicity_TEConstants ----
EXTENDS DelegationAcyclicity

CONSTANTS a1, a2, a3

=============================================================================

---- CONFIG DelegationAcyclicity_TTrace_1774058958 ----
CONSTANTS
    Agents = { a1 , a2 , a3 }
    MaxChainDepth = 3
    MaxVotingWeight = 150
    a1 = a1
    a2 = a2
    a3 = a3

INVARIANT
    _inv

CHECK_DEADLOCK
    \* CHECK_DEADLOCK off because of PROPERTY or INVARIANT above.
    FALSE

INIT
    _init

NEXT
    _next

CONSTANT
    _TETrace <- _trace

ALIAS
    _expression
=============================================================================
\* Generated on Sat Mar 21 05:03:10 SAST 2026