# 📊 Mycelix Protocol Visual Diagrams

**Interactive diagrams explaining key concepts and architecture**

---

## 1. Byzantine Fault Tolerance: Breaking the 33% Barrier

### Classical BFT Limitation

```mermaid
graph LR
    subgraph "Classical BFT (33% Limit)"
        N1[Node 1<br/>Honest] --> A1[Vote]
        N2[Node 2<br/>Honest] --> A1
        N3[Node 3<br/>Byzantine ❌] --> A1
        A1 --> R1[Consensus ✅<br/>67% > 50%]
    end

    subgraph "System Fails"
        N4[Node 1<br/>Honest] --> A2[Vote]
        N5[Node 2<br/>Byzantine ❌] --> A2
        N6[Node 3<br/>Byzantine ❌] --> A2
        A2 --> R2[No Consensus ❌<br/>34% < 50%]
    end
```

**Problem:** With >33% Byzantine nodes, honest nodes can't reach majority consensus.

### MATL Solution: Reputation-Weighted Validation

```mermaid
graph TD
    subgraph "MATL (45% Tolerance)"
        N1[New Node<br/>Rep: 0.1] --> W1[Weight: 0.01]
        N2[Trusted Node<br/>Rep: 0.9] --> W2[Weight: 0.81]
        N3[Byzantine<br/>Rep: 0.1] --> W3[Weight: 0.01]

        W1 --> A[Weighted Sum]
        W2 --> A
        W3 --> A

        A --> R[Byzantine Power < Threshold ✅<br/>System Safe!]
    end
```

**Key Insight:** `Byzantine_Power = Σ(malicious_reputation²)`

Even with 45% malicious **nodes**, if they have low reputation, their Byzantine **power** stays below the safety threshold.

---

## 2. The Epistemic Cube: 3D Truth Framework

### Three Independent Axes

```mermaid
graph TB
    subgraph "Epistemic Cube (E/N/M)"
        E[E-Axis: Empirical<br/>How do we verify?]
        N[N-Axis: Normative<br/>Who agrees?]
        M[M-Axis: Materiality<br/>How long matters?]

        E -->|E0-E4| EV[E0: Unverifiable<br/>E1: Testimonial<br/>E2: Audit<br/>E3: ZKP<br/>E4: Public]
        N -->|N0-N3| NV[N0: Personal<br/>N1: Communal<br/>N2: Network<br/>N3: Axiomatic]
        M -->|M0-M3| MV[M0: Ephemeral<br/>M1: Temporal<br/>M2: Persistent<br/>M3: Foundational]
    end
```

### Examples of Claims

| Claim | E-Axis | N-Axis | M-Axis | Classification |
|-------|--------|--------|--------|----------------|
| "I like pizza" | E0 | N0 | M0 | (0,0,0) - Personal preference |
| Community vote result | E0 | N2 | M3 | (0,2,3) - Governance record |
| Mathematical proof | E4 | N3 | M3 | (4,3,3) - Universal truth |
| Encrypted health record | E2 | N1 | M2 | (2,1,2) - Private persistent data |

---

## 3. MATL System Architecture

### Three-Mode Operation

```mermaid
flowchart TD
    Start[Client submits gradient] --> Mode{Select MATL Mode}

    Mode -->|Mode 0<br/>33% BFT| Peer[Peer Comparison]
    Mode -->|Mode 1<br/>45% BFT| Oracle[PoGQ Oracle]
    Mode -->|Mode 2<br/>50% BFT| TEE[PoGQ + TEE]

    Peer --> P1[Compare with peers]
    P1 --> P2[Statistical validation]
    P2 --> Trust1[Trust Score: 0-1]

    Oracle --> O1[Submit to oracle]
    O1 --> O2[Cryptographic validation]
    O2 --> Trust2[Trust Score: 0-1]

    TEE --> T1[Trusted execution]
    T1 --> T2[Hardware-backed proof]
    T2 --> Trust3[Trust Score: 0-1]

    Trust1 --> Agg[Reputation-Weighted Aggregation]
    Trust2 --> Agg
    Trust3 --> Agg

    Agg --> Update[Update global model]
```

**Mode Selection:**
- **Mode 0**: Fastest, peer-to-peer validation
- **Mode 1**: ⭐ Recommended for most use cases
- **Mode 2**: Maximum security for critical applications

---

## 4. Federated Learning Data Flow

### Without MATL (Vulnerable)

```mermaid
sequenceDiagram
    participant C1 as Client 1
    participant C2 as Client 2
    participant C3 as Client 3 ⚠️<br/>(Malicious)
    participant S as Server

    S->>C1: Global model
    S->>C2: Global model
    S->>C3: Global model

    Note over C1,C2: Train locally<br/>on honest data
    Note over C3: ❌ Generate<br/>malicious gradient

    C1->>S: Gradient
    C2->>S: Gradient
    C3->>S: Poisoned gradient ☠️

    S->>S: Simple average<br/>(FedAvg)

    Note over S: ❌ Model poisoned!
```

### With MATL (Protected)

```mermaid
sequenceDiagram
    participant C1 as Client 1
    participant C2 as Client 2
    participant C3 as Client 3 ⚠️<br/>(Malicious)
    participant M as MATL Layer
    participant S as Server

    S->>C1: Global model
    S->>C2: Global model
    S->>C3: Global model

    Note over C1,C2: Train locally<br/>on honest data
    Note over C3: ❌ Generate<br/>malicious gradient

    C1->>M: Gradient
    C2->>M: Gradient
    C3->>M: Poisoned gradient

    M->>M: 🛡️ Validate gradients<br/>Trust scores:<br/>C1: 0.95<br/>C2: 0.93<br/>C3: 0.12

    M->>S: Weighted aggregate<br/>(low trust filtered)

    Note over S: ✅ Model protected!
```

---

## 5. System Component Interactions

### Complete Mycelix Stack

```mermaid
graph TB
    subgraph "Application Layer"
        FL[Federated Learning]
        DAO[DAO Governance]
        DKG[Knowledge Graph]
    end

    subgraph "MATL Layer (Byzantine Resistance)"
        Trust[Trust Scoring]
        Agg[Aggregation]
        Cartel[Cartel Detection]
    end

    subgraph "Agent-Centric Layer (Holochain)"
        DHT[Distributed Hash Table]
        Source[Source Chains]
        Validate[Validation Rules]
    end

    subgraph "Storage Layer (Multi-Backend)"
        PG[(PostgreSQL<br/>Fast)]
        ETH[(Ethereum<br/>Immutable)]
        COS[(Cosmos<br/>Inter-chain)]
    end

    FL --> Trust
    DAO --> DHT
    DKG --> DHT

    Trust --> Agg
    Agg --> Cartel
    Cartel --> DHT

    DHT --> Source
    Source --> Validate
    Validate --> PG
    Validate --> ETH
    Validate --> COS
```

---

## 6. Trust Score Evolution Over Time

### Reputation Learning

```mermaid
graph LR
    subgraph "Round 1"
        N1_1[New Node<br/>Rep: 0.5] --> V1{Validate}
        V1 -->|Good gradient| R1_1[Rep: 0.6 ↑]
        V1 -->|Bad gradient| R1_2[Rep: 0.3 ↓]
    end

    subgraph "Round 5"
        R1_1 --> N5_1[Honest Node<br/>Rep: 0.85] --> V5{Validate}
        R1_2 --> N5_2[Byzantine Node<br/>Rep: 0.05] --> V5
        V5 --> Agg[Weighted Aggregation]
        Agg --> Result[Honest node dominates ✅]
    end

    subgraph "Round 10"
        N5_1 --> N10_1[Trusted Node<br/>Rep: 0.95]
        N5_2 --> N10_2[Expelled<br/>Rep → 0]
        N10_1 --> System[System secure]
        N10_2 --> Ignored[Contribution ignored]
    end
```

**Key Properties:**
- ✅ Honest nodes gain reputation over time
- ❌ Byzantine nodes lose reputation quickly
- 🛡️ System becomes more secure with each round

---

## 7. Attack Detection Flow

### How MATL Detects Byzantine Behavior

```mermaid
flowchart TD
    Gradient[Client submits gradient] --> Check1{PoGQ Check}

    Check1 -->|Statistical anomaly| Flag1[🚩 Flag: Outlier]
    Check1 -->|Passes| Check2{TCDM Check}

    Check2 -->|Deviation from consensus| Flag2[🚩 Flag: Deviation]
    Check2 -->|Passes| Check3{Entropy Check}

    Check3 -->|Information theory violation| Flag3[🚩 Flag: Entropy]
    Check3 -->|Passes| Check4{Cartel Check}

    Check4 -->|Coordinated with others| Flag4[🚩 Flag: Cartel]
    Check4 -->|Passes| Accept[✅ Accept gradient<br/>High trust score]

    Flag1 --> Score1[Low trust score: 0.2]
    Flag2 --> Score2[Low trust score: 0.3]
    Flag3 --> Score3[Low trust score: 0.1]
    Flag4 --> Score4[Very low: 0.05]

    Score1 --> Aggregate[Weighted aggregation]
    Score2 --> Aggregate
    Score3 --> Aggregate
    Score4 --> Aggregate
    Accept --> Aggregate

    Aggregate --> Update[Update global model safely]
```

**Defense Mechanisms:**
1. **PoGQ**: Proof of Gradient Quality (statistical validation)
2. **TCDM**: Trust-Corrected Debiased Mean
3. **Entropy**: Information-theoretic anomaly detection
4. **Cartel**: Graph-based clustering analysis

---

## 8. Deployment Options

### From Development to Production

```mermaid
graph LR
    subgraph "Development"
        Dev[Local testing<br/>PostgreSQL backend<br/>Simulated attacks]
    end

    subgraph "Staging"
        Stage[Testnet deployment<br/>Holochain DHT<br/>Real network latency]
    end

    subgraph "Production"
        Prod1[Small scale<br/>Mode 0<br/>< 100 nodes]
        Prod2[Medium scale<br/>Mode 1<br/>100-1000 nodes]
        Prod3[Enterprise<br/>Mode 2<br/>1000+ nodes]
    end

    Dev --> Stage
    Stage --> Prod1
    Prod1 --> Prod2
    Prod2 --> Prod3
```

**Scaling Path:**
- Start small with Mode 0 (faster, simpler)
- Scale to Mode 1 for higher BFT tolerance
- Add Mode 2 with TEE for mission-critical applications

---

## 📖 Related Documentation

- **[MATL Architecture](../0TML/docs/06-architecture/matl_architecture.md)** - Technical deep dive
- **[MATL Integration Tutorial](../tutorials/matl_integration.md)** - Hands-on guide
- **[System Architecture v5.2](../architecture/Mycelix Protocol_ Integrated System Architecture v5.2.md)** - Complete design
- **[Epistemic Charter v2.0](../architecture/THE EPISTEMIC CHARTER (v2.0).md)** - 3D truth framework details

---

**Diagrams created with [Mermaid.js](https://mermaid.js.org/)**
*Interactive and version-controlled documentation*
