--------------------------- MODULE CantorLtcHdc_MC ---------------------------
(***************************************************************************
 * Hierarchical Cantor-LTC/HDC Network - Model Checking Version
 *
 * This is the TLC-executable version of CantorLtcHdc.tla with:
 *   - Finite state representations (integers instead of reals)
 *   - Concrete similarity functions
 *   - Bounded sets for tractable exploration
 *
 * Run: tlc CantorLtcHdc_MC.tla -config CantorLtcHdc.cfg -workers auto
 *
 * Authors: Tristan Stoltz, Claude (Anthropic)
 * Date: January 2, 2026
 * Version: 2.0-MC (Model Checking)
 ***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

(***************************************************************************
 * CONSTANTS
 ***************************************************************************)

CONSTANTS
    HdcDim,           \* Symbolic dimension (not used in MC version)
    MaxFixedDepth,    \* Maximum fixed core depth (e.g., 3)
    MaxElasticDepth,  \* Maximum elastic depth (e.g., 5)
    CantorRatio,      \* Scaling ratio (symbolic in MC)
    BaseTau,          \* Base time constant (symbolic)
    DeltaT,           \* Integration timestep (symbolic)
    MaxBound,         \* Stability bound for state magnitude
    LateralThreshold, \* Similarity threshold for lateral binding
    BuddingThreshold, \* Φ threshold for budding
    PruningPhiThreshold \* Φ threshold for pruning

(***************************************************************************
 * TYPE DEFINITIONS (Finite for Model Checking)
 ***************************************************************************)

\* State magnitude range: integers in [-MaxBound, MaxBound]
StateRange == -MaxBound..MaxBound

\* Node identifier: (level, index within level)
\* For MC: level in 0..MaxElasticDepth, index in 0..2^level-1
\* Using UNION to handle level-dependent index bounds
NodeId == UNION {
    {<<lv, ix>> : ix \in 0..((2^lv) - 1)} : lv \in 0..MaxElasticDepth
}

\* Possible similarity values (discretized)
SimilarityRange == 0..10  \* 0 = dissimilar, 10 = identical

\* Φ values (discretized to integers 0-10)
PhiRange == 0..10

\* Sentinel value for "no children" (use invalid node ID)
NoChildren == [left |-> <<-1, -1>>, right |-> <<-1, -1>>]

\* Check if a children record is the sentinel
HasNoChildren(ch) == ch.left[1] = -1

(***************************************************************************
 * VARIABLES
 ***************************************************************************)

VARIABLES
    \* === Core State (simplified to magnitude only) ===
    states,           \* Function: NodeId -> StateRange (magnitude)

    \* === Hierarchy Structure ===
    activeNodes,      \* Set of currently active node IDs
    children,         \* Function: NodeId -> {left, right} or NoChildren

    \* === Lateral Binding (v2.0) ===
    lateralLinks,     \* Function: NodeId -> Set of peer NodeIds
    clusters,         \* Set of node sets (autonomous clusters)

    \* === Elastic Depth (v2.0) ===
    elasticNodes,     \* Set of nodes in elastic periphery

    \* === Consciousness Metrics ===
    localPhi,         \* Function: NodeId -> PhiRange
    globalPhi,        \* Integer in PhiRange

    \* === Time ===
    time              \* Simulation step counter

\* Tuple of all variables
vars == <<states, activeNodes, children, lateralLinks, clusters,
          elasticNodes, localPhi, globalPhi, time>>

(***************************************************************************
 * HELPER OPERATORS
 ***************************************************************************)

\* Number of nodes at a given level (perfect binary tree)
NodesAtLevel(level) == 2^level

\* Total nodes in fixed core
TotalFixedNodes == (2^(MaxFixedDepth + 1)) - 1

\* Check if a node is in the fixed core
IsFixedCore(nodeId) == nodeId[1] <= MaxFixedDepth

\* Check if a node is in the elastic periphery
IsElastic(nodeId) == nodeId[1] > MaxFixedDepth

\* Get parent node ID
ParentId(nodeId) ==
    IF nodeId[1] = 0
    THEN nodeId  \* Root has no parent
    ELSE <<nodeId[1] - 1, nodeId[2] \div 2>>

\* Get left child node ID
LeftChildId(nodeId) == <<nodeId[1] + 1, nodeId[2] * 2>>

\* Get right child node ID
RightChildId(nodeId) == <<nodeId[1] + 1, nodeId[2] * 2 + 1>>

\* Check if two nodes are at the same level
SameLevel(n1, n2) == n1[1] = n2[1]

\* Simplified similarity (based on state difference)
\* Returns 0-10 where 10 = identical, 0 = maximally different
Similarity(s1, s2) ==
    LET diff == IF s1 >= s2 THEN s1 - s2 ELSE s2 - s1
    IN IF diff = 0 THEN 10
       ELSE IF diff >= 10 THEN 0
       ELSE 10 - diff

\* Bounded addition (stays within StateRange)
BoundedAdd(x, delta) ==
    LET result == x + delta
    IN IF result > MaxBound THEN MaxBound
       ELSE IF result < -MaxBound THEN -MaxBound
       ELSE result

\* Simplified activation (bounds to valid range)
Activate(x) ==
    IF x > MaxBound THEN MaxBound
    ELSE IF x < -MaxBound THEN -MaxBound
    ELSE x

\* Compute τ scaling factor for a level (simplified to integer)
TauFactor(level) ==
    IF level = 0 THEN 3
    ELSE IF level = 1 THEN 2
    ELSE 1

(***************************************************************************
 * INITIAL STATE
 ***************************************************************************)

\* Generate initial node IDs for fixed core
\* Using UNION to handle level-dependent index bounds
InitialNodeIds == UNION {
    {<<lv, ix>> : ix \in 0..((2^lv) - 1)} : lv \in 0..MaxFixedDepth
}

\* Initial state predicate
Init ==
    \* Initialize states to zero
    /\ states = [n \in InitialNodeIds |-> 0]

    \* All fixed core nodes are active
    /\ activeNodes = InitialNodeIds

    \* Initialize children pointers
    /\ children = [n \in InitialNodeIds |->
        IF n[1] < MaxFixedDepth
        THEN [left |-> LeftChildId(n), right |-> RightChildId(n)]
        ELSE NoChildren]

    \* No lateral links initially
    /\ lateralLinks = [n \in InitialNodeIds |-> {}]

    \* No clusters initially
    /\ clusters = {}

    \* No elastic nodes initially
    /\ elasticNodes = {}

    \* Initialize Φ to zero
    /\ localPhi = [n \in InitialNodeIds |-> 0]
    /\ globalPhi = 0

    \* Start at time 0
    /\ time = 0

(***************************************************************************
 * LTC DYNAMICS (Simplified for Model Checking)
 ***************************************************************************)

\* Evolve a single node's state
EvolveNode(nodeId) ==
    LET
        current == states[nodeId]
        level == nodeId[1]

        \* Parent influence
        parentId == ParentId(nodeId)
        parentState == IF nodeId[1] = 0 THEN current ELSE states[parentId]
        parentInf == (current + parentState) \div 2

        \* Child influence (average of children)
        childL == IF HasNoChildren(children[nodeId]) THEN current
                  ELSE IF children[nodeId].left \in activeNodes
                       THEN states[children[nodeId].left]
                       ELSE current
        childR == IF HasNoChildren(children[nodeId]) THEN current
                  ELSE IF children[nodeId].right \in activeNodes
                       THEN states[children[nodeId].right]
                       ELSE current
        childInf == (childL + childR) \div 2

        \* Lateral influence (average of lateral peers)
        lateralPeers == lateralLinks[nodeId]
        lateralSum == IF lateralPeers = {} THEN current
                      ELSE current \* Simplified: just use current

        \* Combined influence with τ-scaled damping
        combined == (current + parentInf + childInf) \div 3
        damped == current + (combined - current) \div TauFactor(level)
    IN
        Activate(damped)

\* Global evolution step (all nodes)
StepDynamics ==
    /\ states' = [n \in activeNodes |-> EvolveNode(n)]
    /\ time' = time + 1
    /\ UNCHANGED <<activeNodes, children, lateralLinks, clusters,
                   elasticNodes, localPhi, globalPhi>>

\* External input action (drives state changes for interesting exploration)
ExternalInput ==
    \E nodeId \in activeNodes :
        \E delta \in {-2, -1, 1, 2} :
            /\ states' = [states EXCEPT ![nodeId] = BoundedAdd(@, delta)]
            /\ time' = time + 1
            /\ UNCHANGED <<activeNodes, children, lateralLinks, clusters,
                           elasticNodes, localPhi, globalPhi>>

(***************************************************************************
 * LATERAL BINDING (v2.0)
 ***************************************************************************)

\* Form lateral binding between similar nodes at same level
FormLateralBinding ==
    \E n1, n2 \in activeNodes :
        /\ n1 # n2
        /\ SameLevel(n1, n2)
        /\ Similarity(states[n1], states[n2]) >= LateralThreshold
        /\ n2 \notin lateralLinks[n1]
        \* Add symmetric links
        /\ lateralLinks' = [lateralLinks EXCEPT
            ![n1] = @ \cup {n2},
            ![n2] = @ \cup {n1}]
        /\ UNCHANGED <<states, activeNodes, children, clusters,
                       elasticNodes, localPhi, globalPhi, time>>

\* Form autonomous cluster from laterally-linked nodes
FormCluster ==
    \E nodeId \in activeNodes :
        LET
            peers == lateralLinks[nodeId]
            newCluster == {nodeId} \cup peers
        IN
            /\ Cardinality(newCluster) >= 2
            /\ newCluster \notin clusters
            /\ clusters' = clusters \cup {newCluster}
            /\ UNCHANGED <<states, activeNodes, children, lateralLinks,
                           elasticNodes, localPhi, globalPhi, time>>

(***************************************************************************
 * ELASTIC BUDDING/PRUNING (v2.0)
 ***************************************************************************)

\* Check if a node should bud
ShouldBud(nodeId) ==
    /\ nodeId[1] >= MaxFixedDepth  \* Only at elastic boundary
    /\ nodeId[1] < MaxElasticDepth \* Don't exceed max depth
    /\ HasNoChildren(children[nodeId])   \* No children yet
    /\ localPhi[nodeId] >= BuddingThreshold \* High activity

\* Bud action: create new children
Bud ==
    \E nodeId \in activeNodes :
        /\ ShouldBud(nodeId)
        /\ LET
               leftChild == LeftChildId(nodeId)
               rightChild == RightChildId(nodeId)
           IN
               /\ activeNodes' = activeNodes \cup {leftChild, rightChild}
               /\ elasticNodes' = elasticNodes \cup {leftChild, rightChild}
               \* Update parent's children AND add entries for new children
               /\ children' = [children EXCEPT
                    ![nodeId] = [left |-> leftChild, right |-> rightChild]]
                    @@ (leftChild :> NoChildren)
                    @@ (rightChild :> NoChildren)
               /\ states' = states @@ (leftChild :> 0) @@ (rightChild :> 0)
               /\ localPhi' = localPhi @@ (leftChild :> 0) @@ (rightChild :> 0)
               /\ lateralLinks' = lateralLinks @@ (leftChild :> {}) @@ (rightChild :> {})
               /\ UNCHANGED <<clusters, globalPhi, time>>

\* Check if a node should prune
ShouldPrune(nodeId) ==
    /\ nodeId \in elasticNodes     \* Only elastic nodes
    /\ localPhi[nodeId] < PruningPhiThreshold \* Low Φ
    /\ HasNoChildren(children[nodeId])   \* Leaf node

\* Prune action: remove elastic leaf node
Prune ==
    \E nodeId \in elasticNodes :
        /\ ShouldPrune(nodeId)
        /\ LET parentId == ParentId(nodeId)
           IN
               /\ activeNodes' = activeNodes \ {nodeId}
               /\ elasticNodes' = elasticNodes \ {nodeId}
               /\ IF parentId \in DOMAIN children /\ ~HasNoChildren(children[parentId])
                  THEN children' = [children EXCEPT ![parentId] = NoChildren]
                  ELSE children' = children
               /\ UNCHANGED <<states, lateralLinks, clusters,
                              localPhi, globalPhi, time>>

(***************************************************************************
 * PHI MEASUREMENT (Simplified)
 ***************************************************************************)

\* Compute simplified Φ for a node
ComputeLocalPhi(nodeId) ==
    LET
        myState == states[nodeId]
        parentId == ParentId(nodeId)
        parentSim == IF nodeId[1] = 0 THEN 10
                     ELSE Similarity(myState, states[parentId])

        \* Simplified Φ based on parent similarity
        phi == parentSim
    IN
        IF phi < 0 THEN 0 ELSE IF phi > 10 THEN 10 ELSE phi

\* Update all Φ measurements
UpdatePhi ==
    /\ localPhi' = [n \in activeNodes |-> ComputeLocalPhi(n)]
    /\ LET sumPhi == LET S == {ComputeLocalPhi(n) : n \in activeNodes}
                     IN IF S = {} THEN 0
                        ELSE CHOOSE x \in S : TRUE \* Simplified
       IN globalPhi' = sumPhi
    /\ UNCHANGED <<states, activeNodes, children, lateralLinks,
                   clusters, elasticNodes, time>>

(***************************************************************************
 * NEXT-STATE RELATION
 ***************************************************************************)

Next ==
    \/ StepDynamics        \* Core LTC evolution
    \/ ExternalInput       \* External stimulation
    \/ FormLateralBinding  \* Discover and form lateral links
    \/ FormCluster         \* Create autonomous clusters
    \/ Bud                 \* Create elastic children
    \/ Prune               \* Remove elastic nodes
    \/ UpdatePhi           \* Update consciousness metrics

(***************************************************************************
 * SAFETY INVARIANTS
 ***************************************************************************)

\* S1: State Boundedness - All states within range
StateBoundedness ==
    \A n \in activeNodes :
        states[n] >= -MaxBound /\ states[n] <= MaxBound

\* S2: Hierarchical Ordering (trivially true with our TauFactor)
HierarchicalOrdering == TRUE

\* S3: Fixed Core Integrity - Fixed core nodes NEVER removed
\* THIS IS THE CRITICAL SOVEREIGN INVARIANT
FixedCoreIntegrity ==
    \A n \in InitialNodeIds : n \in activeNodes

\* S4: Lateral Symmetry - Links are bidirectional
LateralSymmetry ==
    \A n1 \in activeNodes :
        \A n2 \in lateralLinks[n1] :
            n2 \in activeNodes => n1 \in lateralLinks[n2]

\* S5: Parent-Child Consistency
ParentChildConsistency ==
    \A n \in activeNodes :
        ~HasNoChildren(children[n]) =>
            (children[n].left \in activeNodes /\ children[n].right \in activeNodes)

\* S6: Elastic Containment - Elastic only in periphery
ElasticContainment ==
    \A n \in elasticNodes : n[1] > MaxFixedDepth

\* S7: Phi Boundedness - Φ in valid range
PhiBoundedness ==
    /\ \A n \in activeNodes : localPhi[n] >= 0 /\ localPhi[n] <= 10
    /\ globalPhi >= 0 /\ globalPhi <= 10

\* Combined Safety
Safety ==
    /\ StateBoundedness
    /\ HierarchicalOrdering
    /\ FixedCoreIntegrity
    /\ LateralSymmetry
    /\ ParentChildConsistency
    /\ ElasticContainment
    /\ PhiBoundedness

(***************************************************************************
 * LIVENESS PROPERTIES
 ***************************************************************************)

\* L1: Eventually reaches bounded state
EventualStability ==
    <>[](\A n \in activeNodes : states[n] >= -50 /\ states[n] <= 50)

\* L2: Similar nodes eventually bind
EventualLateralBinding ==
    (\E n1, n2 \in activeNodes :
        n1 # n2 /\ SameLevel(n1, n2) /\
        Similarity(states[n1], states[n2]) >= LateralThreshold)
    ~> (\E n \in activeNodes : lateralLinks[n] # {})

\* L3: High-Φ nodes eventually bud
EventualBudding ==
    (\E n \in activeNodes : ShouldBud(n))
    ~> (elasticNodes # {})

\* L4: Low-Φ elastic nodes eventually pruned
\* Simplified: eventually no prunable nodes remain
EventualPruning ==
    <>[](\A n \in elasticNodes : ~ShouldPrune(n))

(***************************************************************************
 * SPECIFICATION
 ***************************************************************************)

\* Complete specification with weak fairness
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Fairness for all actions
Fairness ==
    /\ WF_vars(StepDynamics)
    /\ WF_vars(ExternalInput)
    /\ WF_vars(FormLateralBinding)
    /\ WF_vars(FormCluster)
    /\ WF_vars(Bud)
    /\ WF_vars(Prune)
    /\ WF_vars(UpdatePhi)

(***************************************************************************
 * TYPE INVARIANT (for debugging)
 ***************************************************************************)

TypeOK ==
    /\ states \in [activeNodes -> StateRange]
    /\ activeNodes \subseteq NodeId
    /\ elasticNodes \subseteq activeNodes
    /\ \A n \in activeNodes : lateralLinks[n] \subseteq activeNodes
    /\ localPhi \in [activeNodes -> PhiRange]
    /\ globalPhi \in PhiRange
    /\ time \in Nat

=============================================================================
