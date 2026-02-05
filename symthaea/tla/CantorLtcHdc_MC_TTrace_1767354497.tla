---- MODULE CantorLtcHdc_MC_TTrace_1767354497 ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, CantorLtcHdc_MC

_expression ==
    LET CantorLtcHdc_MC_TEExpression == INSTANCE CantorLtcHdc_MC_TEExpression
    IN CantorLtcHdc_MC_TEExpression!expression
----

_trace ==
    LET CantorLtcHdc_MC_TETrace == INSTANCE CantorLtcHdc_MC_TETrace
    IN CantorLtcHdc_MC_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        globalPhi = (10)
        /\
        elasticNodes = ({<<4, 0>>, <<4, 1>>})
        /\
        children = ((<<0, 0>> :> [left |-> <<1, 0>>, right |-> <<1, 1>>] @@ <<1, 0>> :> [left |-> <<2, 0>>, right |-> <<2, 1>>] @@ <<1, 1>> :> [left |-> <<2, 2>>, right |-> <<2, 3>>] @@ <<2, 0>> :> [left |-> <<3, 0>>, right |-> <<3, 1>>] @@ <<2, 1>> :> [left |-> <<3, 2>>, right |-> <<3, 3>>] @@ <<2, 2>> :> [left |-> <<3, 4>>, right |-> <<3, 5>>] @@ <<2, 3>> :> [left |-> <<3, 6>>, right |-> <<3, 7>>] @@ <<3, 0>> :> [left |-> <<4, 0>>, right |-> <<4, 1>>] @@ <<3, 1>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 2>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 3>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 4>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 5>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 6>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 7>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>]))
        /\
        time = (1)
        /\
        lateralLinks = ((<<0, 0>> :> {} @@ <<1, 0>> :> {} @@ <<1, 1>> :> {} @@ <<2, 0>> :> {} @@ <<2, 1>> :> {} @@ <<2, 2>> :> {} @@ <<2, 3>> :> {} @@ <<3, 0>> :> {} @@ <<3, 1>> :> {} @@ <<3, 2>> :> {} @@ <<3, 3>> :> {} @@ <<3, 4>> :> {} @@ <<3, 5>> :> {} @@ <<3, 6>> :> {} @@ <<3, 7>> :> {} @@ <<4, 0>> :> {} @@ <<4, 1>> :> {}))
        /\
        localPhi = ((<<0, 0>> :> 10 @@ <<1, 0>> :> 10 @@ <<1, 1>> :> 10 @@ <<2, 0>> :> 10 @@ <<2, 1>> :> 10 @@ <<2, 2>> :> 10 @@ <<2, 3>> :> 10 @@ <<3, 0>> :> 10 @@ <<3, 1>> :> 10 @@ <<3, 2>> :> 10 @@ <<3, 3>> :> 10 @@ <<3, 4>> :> 10 @@ <<3, 5>> :> 10 @@ <<3, 6>> :> 10 @@ <<3, 7>> :> 10 @@ <<4, 0>> :> 0 @@ <<4, 1>> :> 0))
        /\
        clusters = ({})
        /\
        activeNodes = ({<<0, 0>>, <<1, 0>>, <<1, 1>>, <<2, 0>>, <<2, 1>>, <<2, 2>>, <<2, 3>>, <<3, 0>>, <<3, 1>>, <<3, 2>>, <<3, 3>>, <<3, 4>>, <<3, 5>>, <<3, 6>>, <<3, 7>>, <<4, 0>>, <<4, 1>>})
        /\
        states = ((<<0, 0>> :> 0 @@ <<1, 0>> :> 0 @@ <<1, 1>> :> 0 @@ <<2, 0>> :> 0 @@ <<2, 1>> :> 0 @@ <<2, 2>> :> 0 @@ <<2, 3>> :> 0 @@ <<3, 0>> :> 0 @@ <<3, 1>> :> 0 @@ <<3, 2>> :> 0 @@ <<3, 3>> :> 0 @@ <<3, 4>> :> 0 @@ <<3, 5>> :> 0 @@ <<3, 6>> :> 0 @@ <<3, 7>> :> 0 @@ <<4, 0>> :> 0 @@ <<4, 1>> :> 0))
    )
----

_init ==
    /\ activeNodes = _TETrace[1].activeNodes
    /\ time = _TETrace[1].time
    /\ states = _TETrace[1].states
    /\ globalPhi = _TETrace[1].globalPhi
    /\ elasticNodes = _TETrace[1].elasticNodes
    /\ clusters = _TETrace[1].clusters
    /\ children = _TETrace[1].children
    /\ localPhi = _TETrace[1].localPhi
    /\ lateralLinks = _TETrace[1].lateralLinks
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ activeNodes  = _TETrace[i].activeNodes
        /\ activeNodes' = _TETrace[j].activeNodes
        /\ time  = _TETrace[i].time
        /\ time' = _TETrace[j].time
        /\ states  = _TETrace[i].states
        /\ states' = _TETrace[j].states
        /\ globalPhi  = _TETrace[i].globalPhi
        /\ globalPhi' = _TETrace[j].globalPhi
        /\ elasticNodes  = _TETrace[i].elasticNodes
        /\ elasticNodes' = _TETrace[j].elasticNodes
        /\ clusters  = _TETrace[i].clusters
        /\ clusters' = _TETrace[j].clusters
        /\ children  = _TETrace[i].children
        /\ children' = _TETrace[j].children
        /\ localPhi  = _TETrace[i].localPhi
        /\ localPhi' = _TETrace[j].localPhi
        /\ lateralLinks  = _TETrace[i].lateralLinks
        /\ lateralLinks' = _TETrace[j].lateralLinks

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("CantorLtcHdc_MC_TTrace_1767354497.json", _TETrace)

=============================================================================

 Note that you can extract this module `CantorLtcHdc_MC_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `CantorLtcHdc_MC_TEExpression.tla` file takes precedence 
  over the module `CantorLtcHdc_MC_TEExpression` below).

---- MODULE CantorLtcHdc_MC_TEExpression ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, CantorLtcHdc_MC

expression == 
    [
        \* To hide variables of the `CantorLtcHdc_MC` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        activeNodes |-> activeNodes
        ,time |-> time
        ,states |-> states
        ,globalPhi |-> globalPhi
        ,elasticNodes |-> elasticNodes
        ,clusters |-> clusters
        ,children |-> children
        ,localPhi |-> localPhi
        ,lateralLinks |-> lateralLinks
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_activeNodesUnchanged |-> activeNodes = activeNodes'
        
        \* Format the `activeNodes` variable as Json value.
        \* ,_activeNodesJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(activeNodes)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_activeNodesModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].activeNodes # _TETrace[s-1].activeNodes
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE CantorLtcHdc_MC_TETrace ----
\*EXTENDS IOUtils, TLC, CantorLtcHdc_MC
\*
\*trace == IODeserialize("CantorLtcHdc_MC_TTrace_1767354497.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE CantorLtcHdc_MC_TETrace ----
EXTENDS TLC, CantorLtcHdc_MC

trace == 
    <<
    ([globalPhi |-> 0,elasticNodes |-> {},children |-> (<<0, 0>> :> [left |-> <<1, 0>>, right |-> <<1, 1>>] @@ <<1, 0>> :> [left |-> <<2, 0>>, right |-> <<2, 1>>] @@ <<1, 1>> :> [left |-> <<2, 2>>, right |-> <<2, 3>>] @@ <<2, 0>> :> [left |-> <<3, 0>>, right |-> <<3, 1>>] @@ <<2, 1>> :> [left |-> <<3, 2>>, right |-> <<3, 3>>] @@ <<2, 2>> :> [left |-> <<3, 4>>, right |-> <<3, 5>>] @@ <<2, 3>> :> [left |-> <<3, 6>>, right |-> <<3, 7>>] @@ <<3, 0>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 1>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 2>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 3>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 4>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 5>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 6>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 7>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>]),time |-> 0,lateralLinks |-> (<<0, 0>> :> {} @@ <<1, 0>> :> {} @@ <<1, 1>> :> {} @@ <<2, 0>> :> {} @@ <<2, 1>> :> {} @@ <<2, 2>> :> {} @@ <<2, 3>> :> {} @@ <<3, 0>> :> {} @@ <<3, 1>> :> {} @@ <<3, 2>> :> {} @@ <<3, 3>> :> {} @@ <<3, 4>> :> {} @@ <<3, 5>> :> {} @@ <<3, 6>> :> {} @@ <<3, 7>> :> {}),localPhi |-> (<<0, 0>> :> 0 @@ <<1, 0>> :> 0 @@ <<1, 1>> :> 0 @@ <<2, 0>> :> 0 @@ <<2, 1>> :> 0 @@ <<2, 2>> :> 0 @@ <<2, 3>> :> 0 @@ <<3, 0>> :> 0 @@ <<3, 1>> :> 0 @@ <<3, 2>> :> 0 @@ <<3, 3>> :> 0 @@ <<3, 4>> :> 0 @@ <<3, 5>> :> 0 @@ <<3, 6>> :> 0 @@ <<3, 7>> :> 0),clusters |-> {},activeNodes |-> {<<0, 0>>, <<1, 0>>, <<1, 1>>, <<2, 0>>, <<2, 1>>, <<2, 2>>, <<2, 3>>, <<3, 0>>, <<3, 1>>, <<3, 2>>, <<3, 3>>, <<3, 4>>, <<3, 5>>, <<3, 6>>, <<3, 7>>},states |-> (<<0, 0>> :> 0 @@ <<1, 0>> :> 0 @@ <<1, 1>> :> 0 @@ <<2, 0>> :> 0 @@ <<2, 1>> :> 0 @@ <<2, 2>> :> 0 @@ <<2, 3>> :> 0 @@ <<3, 0>> :> 0 @@ <<3, 1>> :> 0 @@ <<3, 2>> :> 0 @@ <<3, 3>> :> 0 @@ <<3, 4>> :> 0 @@ <<3, 5>> :> 0 @@ <<3, 6>> :> 0 @@ <<3, 7>> :> 0)]),
    ([globalPhi |-> 0,elasticNodes |-> {},children |-> (<<0, 0>> :> [left |-> <<1, 0>>, right |-> <<1, 1>>] @@ <<1, 0>> :> [left |-> <<2, 0>>, right |-> <<2, 1>>] @@ <<1, 1>> :> [left |-> <<2, 2>>, right |-> <<2, 3>>] @@ <<2, 0>> :> [left |-> <<3, 0>>, right |-> <<3, 1>>] @@ <<2, 1>> :> [left |-> <<3, 2>>, right |-> <<3, 3>>] @@ <<2, 2>> :> [left |-> <<3, 4>>, right |-> <<3, 5>>] @@ <<2, 3>> :> [left |-> <<3, 6>>, right |-> <<3, 7>>] @@ <<3, 0>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 1>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 2>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 3>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 4>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 5>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 6>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 7>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>]),time |-> 1,lateralLinks |-> (<<0, 0>> :> {} @@ <<1, 0>> :> {} @@ <<1, 1>> :> {} @@ <<2, 0>> :> {} @@ <<2, 1>> :> {} @@ <<2, 2>> :> {} @@ <<2, 3>> :> {} @@ <<3, 0>> :> {} @@ <<3, 1>> :> {} @@ <<3, 2>> :> {} @@ <<3, 3>> :> {} @@ <<3, 4>> :> {} @@ <<3, 5>> :> {} @@ <<3, 6>> :> {} @@ <<3, 7>> :> {}),localPhi |-> (<<0, 0>> :> 0 @@ <<1, 0>> :> 0 @@ <<1, 1>> :> 0 @@ <<2, 0>> :> 0 @@ <<2, 1>> :> 0 @@ <<2, 2>> :> 0 @@ <<2, 3>> :> 0 @@ <<3, 0>> :> 0 @@ <<3, 1>> :> 0 @@ <<3, 2>> :> 0 @@ <<3, 3>> :> 0 @@ <<3, 4>> :> 0 @@ <<3, 5>> :> 0 @@ <<3, 6>> :> 0 @@ <<3, 7>> :> 0),clusters |-> {},activeNodes |-> {<<0, 0>>, <<1, 0>>, <<1, 1>>, <<2, 0>>, <<2, 1>>, <<2, 2>>, <<2, 3>>, <<3, 0>>, <<3, 1>>, <<3, 2>>, <<3, 3>>, <<3, 4>>, <<3, 5>>, <<3, 6>>, <<3, 7>>},states |-> (<<0, 0>> :> 0 @@ <<1, 0>> :> 0 @@ <<1, 1>> :> 0 @@ <<2, 0>> :> 0 @@ <<2, 1>> :> 0 @@ <<2, 2>> :> 0 @@ <<2, 3>> :> 0 @@ <<3, 0>> :> 0 @@ <<3, 1>> :> 0 @@ <<3, 2>> :> 0 @@ <<3, 3>> :> 0 @@ <<3, 4>> :> 0 @@ <<3, 5>> :> 0 @@ <<3, 6>> :> 0 @@ <<3, 7>> :> 0)]),
    ([globalPhi |-> 10,elasticNodes |-> {},children |-> (<<0, 0>> :> [left |-> <<1, 0>>, right |-> <<1, 1>>] @@ <<1, 0>> :> [left |-> <<2, 0>>, right |-> <<2, 1>>] @@ <<1, 1>> :> [left |-> <<2, 2>>, right |-> <<2, 3>>] @@ <<2, 0>> :> [left |-> <<3, 0>>, right |-> <<3, 1>>] @@ <<2, 1>> :> [left |-> <<3, 2>>, right |-> <<3, 3>>] @@ <<2, 2>> :> [left |-> <<3, 4>>, right |-> <<3, 5>>] @@ <<2, 3>> :> [left |-> <<3, 6>>, right |-> <<3, 7>>] @@ <<3, 0>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 1>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 2>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 3>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 4>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 5>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 6>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 7>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>]),time |-> 1,lateralLinks |-> (<<0, 0>> :> {} @@ <<1, 0>> :> {} @@ <<1, 1>> :> {} @@ <<2, 0>> :> {} @@ <<2, 1>> :> {} @@ <<2, 2>> :> {} @@ <<2, 3>> :> {} @@ <<3, 0>> :> {} @@ <<3, 1>> :> {} @@ <<3, 2>> :> {} @@ <<3, 3>> :> {} @@ <<3, 4>> :> {} @@ <<3, 5>> :> {} @@ <<3, 6>> :> {} @@ <<3, 7>> :> {}),localPhi |-> (<<0, 0>> :> 10 @@ <<1, 0>> :> 10 @@ <<1, 1>> :> 10 @@ <<2, 0>> :> 10 @@ <<2, 1>> :> 10 @@ <<2, 2>> :> 10 @@ <<2, 3>> :> 10 @@ <<3, 0>> :> 10 @@ <<3, 1>> :> 10 @@ <<3, 2>> :> 10 @@ <<3, 3>> :> 10 @@ <<3, 4>> :> 10 @@ <<3, 5>> :> 10 @@ <<3, 6>> :> 10 @@ <<3, 7>> :> 10),clusters |-> {},activeNodes |-> {<<0, 0>>, <<1, 0>>, <<1, 1>>, <<2, 0>>, <<2, 1>>, <<2, 2>>, <<2, 3>>, <<3, 0>>, <<3, 1>>, <<3, 2>>, <<3, 3>>, <<3, 4>>, <<3, 5>>, <<3, 6>>, <<3, 7>>},states |-> (<<0, 0>> :> 0 @@ <<1, 0>> :> 0 @@ <<1, 1>> :> 0 @@ <<2, 0>> :> 0 @@ <<2, 1>> :> 0 @@ <<2, 2>> :> 0 @@ <<2, 3>> :> 0 @@ <<3, 0>> :> 0 @@ <<3, 1>> :> 0 @@ <<3, 2>> :> 0 @@ <<3, 3>> :> 0 @@ <<3, 4>> :> 0 @@ <<3, 5>> :> 0 @@ <<3, 6>> :> 0 @@ <<3, 7>> :> 0)]),
    ([globalPhi |-> 10,elasticNodes |-> {<<4, 0>>, <<4, 1>>},children |-> (<<0, 0>> :> [left |-> <<1, 0>>, right |-> <<1, 1>>] @@ <<1, 0>> :> [left |-> <<2, 0>>, right |-> <<2, 1>>] @@ <<1, 1>> :> [left |-> <<2, 2>>, right |-> <<2, 3>>] @@ <<2, 0>> :> [left |-> <<3, 0>>, right |-> <<3, 1>>] @@ <<2, 1>> :> [left |-> <<3, 2>>, right |-> <<3, 3>>] @@ <<2, 2>> :> [left |-> <<3, 4>>, right |-> <<3, 5>>] @@ <<2, 3>> :> [left |-> <<3, 6>>, right |-> <<3, 7>>] @@ <<3, 0>> :> [left |-> <<4, 0>>, right |-> <<4, 1>>] @@ <<3, 1>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 2>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 3>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 4>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 5>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 6>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>] @@ <<3, 7>> :> [left |-> <<-1, -1>>, right |-> <<-1, -1>>]),time |-> 1,lateralLinks |-> (<<0, 0>> :> {} @@ <<1, 0>> :> {} @@ <<1, 1>> :> {} @@ <<2, 0>> :> {} @@ <<2, 1>> :> {} @@ <<2, 2>> :> {} @@ <<2, 3>> :> {} @@ <<3, 0>> :> {} @@ <<3, 1>> :> {} @@ <<3, 2>> :> {} @@ <<3, 3>> :> {} @@ <<3, 4>> :> {} @@ <<3, 5>> :> {} @@ <<3, 6>> :> {} @@ <<3, 7>> :> {} @@ <<4, 0>> :> {} @@ <<4, 1>> :> {}),localPhi |-> (<<0, 0>> :> 10 @@ <<1, 0>> :> 10 @@ <<1, 1>> :> 10 @@ <<2, 0>> :> 10 @@ <<2, 1>> :> 10 @@ <<2, 2>> :> 10 @@ <<2, 3>> :> 10 @@ <<3, 0>> :> 10 @@ <<3, 1>> :> 10 @@ <<3, 2>> :> 10 @@ <<3, 3>> :> 10 @@ <<3, 4>> :> 10 @@ <<3, 5>> :> 10 @@ <<3, 6>> :> 10 @@ <<3, 7>> :> 10 @@ <<4, 0>> :> 0 @@ <<4, 1>> :> 0),clusters |-> {},activeNodes |-> {<<0, 0>>, <<1, 0>>, <<1, 1>>, <<2, 0>>, <<2, 1>>, <<2, 2>>, <<2, 3>>, <<3, 0>>, <<3, 1>>, <<3, 2>>, <<3, 3>>, <<3, 4>>, <<3, 5>>, <<3, 6>>, <<3, 7>>, <<4, 0>>, <<4, 1>>},states |-> (<<0, 0>> :> 0 @@ <<1, 0>> :> 0 @@ <<1, 1>> :> 0 @@ <<2, 0>> :> 0 @@ <<2, 1>> :> 0 @@ <<2, 2>> :> 0 @@ <<2, 3>> :> 0 @@ <<3, 0>> :> 0 @@ <<3, 1>> :> 0 @@ <<3, 2>> :> 0 @@ <<3, 3>> :> 0 @@ <<3, 4>> :> 0 @@ <<3, 5>> :> 0 @@ <<3, 6>> :> 0 @@ <<3, 7>> :> 0 @@ <<4, 0>> :> 0 @@ <<4, 1>> :> 0)])
    >>
----


=============================================================================

---- CONFIG CantorLtcHdc_MC_TTrace_1767354497 ----
CONSTANTS
    HdcDim = 4
    MaxFixedDepth = 3
    MaxElasticDepth = 5
    CantorRatio = 1
    BaseTau = 1
    DeltaT = 1
    MaxBound = 100
    LateralThreshold = 1
    BuddingThreshold = 1
    PruningPhiThreshold = 1

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
\* Generated on Fri Jan 02 13:48:58 SAST 2026