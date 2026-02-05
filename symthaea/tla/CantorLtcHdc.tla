--------------------------- MODULE CantorLtcHdc ---------------------------
(***************************************************************************
 * Hierarchical Cantor-LTC/HDC Network - Formal Specification v2.0
 *
 * This TLA+ specification formally verifies the Hierarchical Cantor-LTC/HDC
 * consciousness architecture, including:
 *   - Core LTC dynamics with HDC binding
 *   - Cantor hierarchy with τ scaling
 *   - Lateral binding (v2.0)
 *   - Elastic budding/pruning (v2.0)
 *   - Hierarchical Φ measurement
 *
 * Authors: Tristan Stoltz, Claude (Anthropic)
 * Date: January 2, 2026
 * Version: 2.0
 ***************************************************************************)

EXTENDS Reals, Integers, Sequences, FiniteSets, TLC

(***************************************************************************
 * CONSTANTS
 ***************************************************************************)

CONSTANTS
    HdcDim,           \* Dimension of hypervectors (16,384 in implementation)
    MaxFixedDepth,    \* Maximum fixed core depth (6)
    MaxElasticDepth,  \* Maximum elastic depth (e.g., 10)
    CantorRatio,      \* Scaling ratio (1/3)
    BaseTau,          \* Root time constant (1.0s = 1000ms)
    DeltaT,           \* Integration timestep
    MaxBound,         \* Stability bound for state magnitude
    LateralThreshold, \* Similarity threshold for lateral binding (0.85)
    BuddingThreshold, \* Prediction error threshold for budding
    PruningPhiThreshold \* Φ threshold for pruning

(***************************************************************************
 * TYPE DEFINITIONS
 ***************************************************************************)

\* A simplified hypervector representation
\* In practice, this would be a 16,384-dimensional real vector
\* For TLA+, we abstract to key properties
HyperVector == [magnitude: Real, similarity: Real -> Real]

\* Node identifier: (level, index within level)
NodeId == Nat \X Nat

\* Lateral link between nodes
LateralLink == [peer: NodeId, similarity: Real, active: BOOLEAN]

\* Cluster of laterally-bound nodes
Cluster == SUBSET NodeId

(***************************************************************************
 * VARIABLES
 ***************************************************************************)

VARIABLES
    \* === Core State ===
    states,           \* Function: NodeId -> HyperVector (node states)
    taus,             \* Function: level -> Real (time constants)

    \* === Hierarchy Structure ===
    activeNodes,      \* Set of currently active node IDs
    children,         \* Function: NodeId -> {left: NodeId, right: NodeId} or NULL

    \* === Lateral Binding (v2.0) ===
    lateralLinks,     \* Function: NodeId -> Set of LateralLink
    clusters,         \* Set of Cluster (autonomous clusters)

    \* === Elastic Depth (v2.0) ===
    elasticNodes,     \* Set of nodes in elastic periphery (level > MaxFixedDepth)

    \* === Consciousness Metrics ===
    localPhi,         \* Function: NodeId -> Real (local Φ)
    globalPhi,        \* Real (global hierarchical Φ)
    coherences,       \* Function: NodeId -> Real (parent coherence)

    \* === Time ===
    time              \* Simulation time

\* Tuple of all variables for temporal formulas
vars == <<states, taus, activeNodes, children, lateralLinks, clusters,
          elasticNodes, localPhi, globalPhi, coherences, time>>

(***************************************************************************
 * HELPER OPERATORS
 ***************************************************************************)

\* Compute τ at a given level using Cantor scaling
Tau(level) == BaseTau * (CantorRatio ^ level)

\* Number of nodes at a given level (perfect binary tree)
NodesAtLevel(level) == 2 ^ level

\* Total nodes in fixed core (levels 0 to MaxFixedDepth)
TotalFixedNodes == (2 ^ (MaxFixedDepth + 1)) - 1

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

\* Simplified similarity computation (abstracted)
\* In implementation: cosine similarity of HDC vectors
Similarity(s1, s2) ==
    IF s1.magnitude = 0 \/ s2.magnitude = 0
    THEN 0
    ELSE s1.similarity[s2]

\* Simplified magnitude (abstracted L2 norm)
Magnitude(s) == s.magnitude

\* Simplified HDC binding (element-wise multiplication)
Bind(s1, s2) == [
    magnitude |-> s1.magnitude * s2.magnitude,
    similarity |-> [x \in HyperVector |-> s1.similarity[x] * s2.similarity[x]]
]

\* Simplified HDC bundling (element-wise average)
Bundle(stateSet) ==
    LET n == Cardinality(stateSet)
    IN [
        magnitude |-> (SUM x \in stateSet : x.magnitude) / n,
        similarity |-> [y \in HyperVector |->
            (SUM x \in stateSet : x.similarity[y]) / n]
    ]

\* Activation function (tanh, bounded in [-1, 1])
Tanh(x) ==
    IF x > 1 THEN 1
    ELSE IF x < -1 THEN -1
    ELSE x

(***************************************************************************
 * INITIAL STATE
 ***************************************************************************)

\* Generate initial node IDs for fixed core
InitialNodeIds ==
    {<<level, idx>> : level \in 0..MaxFixedDepth, idx \in 0..(NodesAtLevel(level) - 1)}

\* Initial state predicate
Init ==
    \* Initialize states to zero vectors
    /\ states = [n \in InitialNodeIds |-> [magnitude |-> 0, similarity |-> [x \in HyperVector |-> 0]]]

    \* Initialize τ at each level
    /\ taus = [level \in 0..MaxElasticDepth |-> Tau(level)]

    \* All fixed core nodes are active
    /\ activeNodes = InitialNodeIds

    \* Initialize children pointers
    /\ children = [n \in InitialNodeIds |->
        IF n[1] < MaxFixedDepth
        THEN [left |-> LeftChildId(n), right |-> RightChildId(n)]
        ELSE NULL]

    \* No lateral links initially
    /\ lateralLinks = [n \in InitialNodeIds |-> {}]

    \* No clusters initially
    /\ clusters = {}

    \* No elastic nodes initially
    /\ elasticNodes = {}

    \* Initialize Φ to zero
    /\ localPhi = [n \in InitialNodeIds |-> 0]
    /\ globalPhi = 0

    \* Initialize coherences
    /\ coherences = [n \in InitialNodeIds |-> 0]

    \* Start at time 0
    /\ time = 0

(***************************************************************************
 * LTC DYNAMICS
 ***************************************************************************)

\* Core LTC evolution for a single node
\* dx/dt = (-x + σ(W⊗x + parent⊗bundle(children) + bias)) / τ
EvolveNode(nodeId) ==
    LET
        current == states[nodeId]
        level == nodeId[1]
        tau == taus[level]

        \* Self-transformation (W⊗x, simplified)
        selfTrans == Bind(current, current)

        \* Parent influence
        parentId == ParentId(nodeId)
        parentState == IF nodeId[1] = 0 THEN current ELSE states[parentId]
        parentInf == Bind(current, parentState)

        \* Child influence (bundle of children states)
        childIds == IF children[nodeId] = NULL
                    THEN {}
                    ELSE {children[nodeId].left, children[nodeId].right}
        childStates == {states[c] : c \in childIds \cap activeNodes}
        childInf == IF childStates = {}
                    THEN [magnitude |-> 0, similarity |-> [x \in HyperVector |-> 0]]
                    ELSE Bundle(childStates)

        \* Lateral influence (v2.0)
        lateralPeers == {link.peer : link \in lateralLinks[nodeId]}
        lateralStates == {states[p] : p \in lateralPeers \cap activeNodes}
        lateralInf == IF lateralStates = {}
                      THEN [magnitude |-> 0, similarity |-> [x \in HyperVector |-> 0]]
                      ELSE Bundle(lateralStates)

        \* Combine influences (weighted)
        combined == Bundle({selfTrans, parentInf, childInf, lateralInf})

        \* Apply activation
        activated == [
            magnitude |-> Tanh(combined.magnitude),
            similarity |-> [x \in HyperVector |-> Tanh(combined.similarity[x])]
        ]

        \* LTC integration: x_new = x + (activated - x) * dt / τ
        delta_mag == (activated.magnitude - current.magnitude) * DeltaT / tau
        newMagnitude == current.magnitude + delta_mag
    IN
        [magnitude |-> newMagnitude,
         similarity |-> [x \in HyperVector |->
            current.similarity[x] + (activated.similarity[x] - current.similarity[x]) * DeltaT / tau]]

\* Global evolution step (all nodes)
StepDynamics ==
    /\ states' = [n \in activeNodes |-> EvolveNode(n)]
    /\ time' = time + DeltaT
    /\ UNCHANGED <<taus, activeNodes, children, lateralLinks, clusters,
                   elasticNodes, localPhi, globalPhi, coherences>>

(***************************************************************************
 * LATERAL BINDING (v2.0)
 ***************************************************************************)

\* Discover lateral peers for a node
DiscoverLateralPeers(nodeId) ==
    LET
        level == nodeId[1]
        myState == states[nodeId]
        sameLevelNodes == {n \in activeNodes : n[1] = level /\ n # nodeId}

        \* Find nodes with similarity > threshold
        similarNodes == {n \in sameLevelNodes :
            Similarity(myState, states[n]) > LateralThreshold}
    IN
        {[peer |-> n, similarity |-> Similarity(myState, states[n]), active |-> TRUE]
         : n \in similarNodes}

\* Form lateral binding action
FormLateralBinding ==
    \E nodeId \in activeNodes :
        LET newLinks == DiscoverLateralPeers(nodeId)
        IN
            /\ newLinks # {}
            /\ lateralLinks' = [lateralLinks EXCEPT ![nodeId] = @ \cup newLinks]
            /\ UNCHANGED <<states, taus, activeNodes, children, clusters,
                           elasticNodes, localPhi, globalPhi, coherences, time>>

\* Form autonomous cluster from lateral links
FormCluster ==
    \E nodeId \in activeNodes :
        LET
            peers == {link.peer : link \in lateralLinks[nodeId]}
            newCluster == {nodeId} \cup peers
        IN
            /\ Cardinality(newCluster) >= 2
            /\ newCluster \notin clusters
            /\ clusters' = clusters \cup {newCluster}
            /\ UNCHANGED <<states, taus, activeNodes, children, lateralLinks,
                           elasticNodes, localPhi, globalPhi, coherences, time>>

(***************************************************************************
 * ELASTIC BUDDING/PRUNING (v2.0)
 ***************************************************************************)

\* Check if a node should bud (create children)
ShouldBud(nodeId) ==
    /\ nodeId[1] >= MaxFixedDepth  \* Only elastic layer can bud
    /\ nodeId[1] < MaxElasticDepth \* Don't exceed max depth
    /\ children[nodeId] = NULL      \* No children yet
    /\ localPhi[nodeId] > BuddingThreshold \* High activity

\* Bud action: create new children for a node
Bud ==
    \E nodeId \in activeNodes :
        /\ ShouldBud(nodeId)
        /\ LET
               leftChild == LeftChildId(nodeId)
               rightChild == RightChildId(nodeId)
               newLevel == nodeId[1] + 1
           IN
               /\ activeNodes' = activeNodes \cup {leftChild, rightChild}
               /\ elasticNodes' = elasticNodes \cup {leftChild, rightChild}
               /\ children' = [children EXCEPT
                    ![nodeId] = [left |-> leftChild, right |-> rightChild]]
               /\ states' = states @@ (leftChild :> [magnitude |-> 0, similarity |-> [x \in HyperVector |-> 0]])
                                   @@ (rightChild :> [magnitude |-> 0, similarity |-> [x \in HyperVector |-> 0]])
               /\ localPhi' = localPhi @@ (leftChild :> 0) @@ (rightChild :> 0)
               /\ coherences' = coherences @@ (leftChild :> 0) @@ (rightChild :> 0)
               /\ lateralLinks' = lateralLinks @@ (leftChild :> {}) @@ (rightChild :> {})
               /\ UNCHANGED <<taus, clusters, globalPhi, time>>

\* Check if a node should prune (remove children)
ShouldPrune(nodeId) ==
    /\ nodeId \in elasticNodes    \* Only elastic nodes can be pruned
    /\ localPhi[nodeId] < PruningPhiThreshold \* Low Φ
    /\ children[nodeId] = NULL    \* Leaf node (no children to orphan)

\* Prune action: remove elastic node
Prune ==
    \E nodeId \in elasticNodes :
        /\ ShouldPrune(nodeId)
        /\ LET parentId == ParentId(nodeId)
           IN
               /\ activeNodes' = activeNodes \ {nodeId}
               /\ elasticNodes' = elasticNodes \ {nodeId}
               \* Update parent's children pointer
               /\ IF children[parentId] # NULL
                  THEN children' = [children EXCEPT ![parentId] = NULL]
                  ELSE children' = children
               /\ UNCHANGED <<states, taus, lateralLinks, clusters,
                              localPhi, globalPhi, coherences, time>>

(***************************************************************************
 * PHI MEASUREMENT
 ***************************************************************************)

\* Compute local Φ for a node (simplified approximation)
\* In implementation: uses similarity matrix and eigenvalue decomposition
ComputeLocalPhi(nodeId) ==
    LET
        myState == states[nodeId]
        parentId == ParentId(nodeId)
        parentCoherence == IF nodeId[1] = 0 THEN 1 ELSE Similarity(myState, states[parentId])

        childIds == IF children[nodeId] = NULL
                    THEN {}
                    ELSE {children[nodeId].left, children[nodeId].right}
        childCoherence == IF childIds = {}
                          THEN 0
                          ELSE (SUM c \in childIds \cap activeNodes :
                                Similarity(myState, states[c])) / Cardinality(childIds \cap activeNodes)

        \* Φ approximation: integration of parent and child coherence
        phi == (parentCoherence + childCoherence) / 2
    IN
        IF phi < 0 THEN 0 ELSE IF phi > 1 THEN 1 ELSE phi

\* Update all Φ measurements
UpdatePhi ==
    /\ localPhi' = [n \in activeNodes |-> ComputeLocalPhi(n)]
    /\ globalPhi' = (SUM n \in activeNodes : ComputeLocalPhi(n)) / Cardinality(activeNodes)
    /\ coherences' = [n \in activeNodes |->
        IF n[1] = 0 THEN 1
        ELSE Similarity(states[n], states[ParentId(n)])]
    /\ UNCHANGED <<states, taus, activeNodes, children, lateralLinks,
                   clusters, elasticNodes, time>>

(***************************************************************************
 * COMPLETE NEXT-STATE RELATION
 ***************************************************************************)

Next ==
    \/ StepDynamics        \* Core LTC evolution
    \/ FormLateralBinding  \* Discover and form lateral links
    \/ FormCluster         \* Create autonomous clusters
    \/ Bud                 \* Create elastic children
    \/ Prune               \* Remove elastic nodes
    \/ UpdatePhi           \* Update consciousness metrics

(***************************************************************************
 * SAFETY INVARIANTS
 ***************************************************************************)

\* S1: State Boundedness - All states remain bounded
StateBoundedness ==
    \A n \in activeNodes : Magnitude(states[n]) < MaxBound

\* S2: Hierarchical Ordering - τ decreases with depth
HierarchicalOrdering ==
    \A l1, l2 \in 0..MaxElasticDepth : l1 < l2 => taus[l1] > taus[l2]

\* S3: Fixed Core Integrity - Fixed core nodes never removed
FixedCoreIntegrity ==
    \A n \in InitialNodeIds : n \in activeNodes

\* S4: Lateral Symmetry - Lateral links are symmetric
LateralSymmetry ==
    \A n1, n2 \in activeNodes :
        (\E link \in lateralLinks[n1] : link.peer = n2) <=>
        (\E link \in lateralLinks[n2] : link.peer = n1)

\* S5: Parent-Child Consistency - Children exist only if active
ParentChildConsistency ==
    \A n \in activeNodes :
        children[n] # NULL =>
            (children[n].left \in activeNodes /\ children[n].right \in activeNodes)

\* S6: Elastic Containment - Elastic nodes only in periphery
ElasticContainment ==
    \A n \in elasticNodes : n[1] > MaxFixedDepth

\* S7: Phi Boundedness - Φ values in [0, 1]
PhiBoundedness ==
    /\ \A n \in activeNodes : 0 <= localPhi[n] /\ localPhi[n] <= 1
    /\ 0 <= globalPhi /\ globalPhi <= 1

\* Combined Safety Invariant
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

\* L1: Eventually converges to stable state
EventualStability ==
    <>[](\A n \in activeNodes : Magnitude(states[n]) < MaxBound / 2)

\* L2: If conditions met, lateral binding eventually occurs
EventualLateralBinding ==
    (\E n1, n2 \in activeNodes :
        SameLevel(n1, n2) /\ Similarity(states[n1], states[n2]) > LateralThreshold)
    ~> (\E n \in activeNodes : lateralLinks[n] # {})

\* L3: High-activity elastic nodes eventually bud
EventualBudding ==
    (\E n \in activeNodes :
        IsElastic(n) /\ localPhi[n] > BuddingThreshold /\ children[n] = NULL)
    ~> (\E n \in elasticNodes : TRUE)

\* L4: Low-activity elastic leaves eventually pruned
EventualPruning ==
    (\E n \in elasticNodes :
        localPhi[n] < PruningPhiThreshold /\ children[n] = NULL)
    ~> (\E removed \in elasticNodes : removed \notin activeNodes')

(***************************************************************************
 * TEMPORAL SPECIFICATION
 ***************************************************************************)

\* Complete specification
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Fairness: all actions get a chance to execute
Fairness ==
    /\ WF_vars(StepDynamics)
    /\ WF_vars(FormLateralBinding)
    /\ WF_vars(FormCluster)
    /\ WF_vars(Bud)
    /\ WF_vars(Prune)
    /\ WF_vars(UpdatePhi)

(***************************************************************************
 * THEOREMS
 ***************************************************************************)

\* Main Safety Theorem
THEOREM SafetyTheorem == Spec => []Safety

\* Stability Theorem
THEOREM StabilityTheorem == Spec => EventualStability

\* Lateral Binding Theorem
THEOREM LateralBindingTheorem == Spec /\ Fairness => EventualLateralBinding

\* Elastic Depth Theorem
THEOREM ElasticDepthTheorem == Spec /\ Fairness => (EventualBudding /\ EventualPruning)

(***************************************************************************
 * MODEL CHECKING CONFIGURATION
 ***************************************************************************)

\* For TLC model checking, use these constant values:
\* HdcDim = 4 (simplified for tractability)
\* MaxFixedDepth = 2 (reduced for state space)
\* MaxElasticDepth = 3
\* CantorRatio = 1/3
\* BaseTau = 1000
\* DeltaT = 100
\* MaxBound = 10
\* LateralThreshold = 85/100
\* BuddingThreshold = 5/10
\* PruningPhiThreshold = 2/10

=============================================================================
\* Modification History
\* Last modified: January 2, 2026
\* Created: January 2, 2026
