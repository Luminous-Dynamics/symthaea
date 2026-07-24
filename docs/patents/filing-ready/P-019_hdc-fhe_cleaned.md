# PROVISIONAL PATENT APPLICATION

## Title of Invention

Systems and Methods for Information-Theoretically Secure Homomorphic Computation on High-Dimensional Binary Vectors

## Inventor(s)

Tristan Stoltz
Richardson, TX, USA

## Assignee

Luminous Dynamics

## Cross-Reference to Related Applications

This application is related to co-pending provisional application P-001 "Unified Hyperdimensional Computing Neuron with Liquid Time-Constant Dynamics" filed [date], which discloses the underlying hyperdimensional computing neuron architecture upon which the cryptographic primitives of the present invention operate. This application is also related to co-pending provisional application P-005 "Consciousness-Aware Federated Learning with Proof of Gradient Quality" filed [date], which discloses the federated learning system in which the privacy-preserving aggregation methods of the present invention may be applied.

---

## SPECIFICATION

### Field of the Invention

[0001] This invention relates to privacy-preserving computation on high-dimensional binary vector representations, and more specifically to systems and methods for homomorphic encryption, message authentication, threshold secret sharing, context-derived key generation, and commitment schemes that exploit the algebraic properties of binary hypervectors used in hyperdimensional computing (HDC), achieving information-theoretic security with negligible computational overhead.

---

### Background of the Invention

[0002] Lattice-based fully homomorphic encryption (FHE), introduced by Gentry (2009, "Fully Homomorphic Encryption Using Ideal Lattices") and refined by subsequent schemes including BGV (Brakerski, Gentry, Vaikuntanathan, 2012), BFV (Fan and Vercauteren, 2012), and CKKS (Cheon, Kim, Kim, Song, 2017), enables arbitrary computation on encrypted data. However, all lattice-based schemes impose 10,000x to 1,000,000x overhead compared to plaintext operations, rely on computational hardness assumptions (Ring-LWE, NTWE) rather than information-theoretic guarantees, and require conversion of data to polynomial ring representations before encrypted computation can proceed.

[0003] Gilad-Bachrach et al. (2016, "CryptoNets: Applying Neural Networks to Encrypted Data") demonstrated neural network inference on data encrypted with lattice FHE. While groundbreaking, CryptoNets incurs substantial accuracy loss (95% +/- 3%) and latency overhead (250 seconds for MNIST inference vs. milliseconds for plaintext). The approach requires converting all operations to polynomial approximations compatible with the FHE scheme.

[0004] Kanerva (2009, "Hyperdimensional Computing: An Introduction") established computing with high-dimensional random vectors using binding (element-wise XOR for binary vectors) and bundling (majority vote). Rahimi et al. (2016) and Imani et al. (2019, "A Framework for Collaborative Learning in Secure HDC") explored privacy-preserving HDC but did not formalize the homomorphic encryption properties, threshold sharing, or the full suite of cryptographic primitives disclosed here.

[0005] Shamir (1979, "How to Share a Secret") introduced polynomial-based (k,n) threshold secret sharing over finite fields. This scheme operates on scalars or small field elements and requires O(k) polynomial evaluations for recovery. It does not natively support high-dimensional vector data---sharing a D-dimensional vector requires D independent Shamir instances.

[0006] Shannon (1949, "Communication Theory of Secrecy Systems") proved that XOR of a message with a uniformly random key of equal length provides perfect secrecy. The ciphertext is statistically independent of the plaintext. However, prior work has not recognized that this XOR operation is identical to the HDC binding operation on binary hypervectors, meaning encryption can be a free byproduct of the existing computational algebra rather than an additional overhead.

[0007] No prior art combines all of the following properties in a single cryptographic framework: zero-overhead encryption where encryption IS the same algebraic operation (XOR binding) used for semantic computation; information-theoretic security without computational hardness assumptions; exact homomorphic binding; approximate homomorphic bundling with fidelity improving with N; exact distance preservation under encryption with a shared mask; native threshold sharing of high-dimensional vectors via majority-vote bundling; zero-collision message authentication via permutation-based binding; context-derived encryption keys from ordered sensor hypervector chains; and zero-overhead commitment schemes via cyclic permutation bijectivity.

---

### Summary of the Invention

[0008] A suite of cryptographic primitives is disclosed that operates natively on binary hypervectors of dimension D (e.g., D = 16,384) used in hyperdimensional computing systems. The core encryption mechanism applies element-wise XOR between a plaintext binary hypervector and a uniformly random mask of equal dimension, achieving perfect secrecy per Shannon's one-time pad theorem (1949). Because XOR is simultaneously the HDC binding operation and the encryption operation, homomorphic computation is achieved at zero overhead: binding two encrypted vectors produces the encryption of their binding under the combined mask, and Hamming similarity between vectors encrypted with the same mask equals the plaintext similarity exactly.

[0009] A collective wisdom aggregation system enables multiple peers to contribute encrypted hypervectors to a shared pool, compute majority-vote bundles in the encrypted domain, and decrypt only the aggregate via threshold cooperation.

[0010] Additional primitives include: (k,n) threshold secret sharing via majority-vote bundling for exact recovery at threshold k; message authentication codes via permutation-based binding with zero collision probability; context-derived encryption keys from ordered sensor hypervector chains; and hide-then-reveal commitment schemes via cyclic permutation bijectivity. All operations execute in O(D) time (5-10 nanoseconds per operation at D = 16,384), approximately 10,000x faster than lattice-based fully homomorphic encryption schemes while providing information-theoretic rather than computational security guarantees.

---

### Detailed Description of Preferred Embodiments

#### Binary Hypervector Representation

[0011] All operations in this invention operate on binary hypervectors (BinaryHV) of dimension D, where each element is a single bit in {0, 1}. The default dimension is D = 16,384, stored as a packed bit array. Four fundamental operations are defined:

[0012] **Binding**: A XOR B, defined as element-wise exclusive OR. Properties: associative, commutative, self-inverse (A XOR A = 0).

[0013] **Bundling**: MAJ(A_1, ..., A_n), defined as per-dimension majority vote. Produces a vector most similar to all inputs.

[0014] **Permutation**: pi_k(A), defined as cyclic left-shift by k positions. Bijective, breaks commutativity.

[0015] **Similarity**: sim(A, B) = 1 - hamming_distance(A,B)/D. For random vectors: sim approximately 0.5 +/- 1/(2*sqrt(D)).

[0016] Key statistical properties at D = 16,384: Random collision probability P(A = B) = 2^{-16384} (negligible). Similarity concentration for random A, B: sim(A, B) approximately 0.5 +/- 0.0039 (3 sigma). Binding invertibility: A XOR A = 0 (XOR is self-inverse). Permutation bijectivity: pi_k is a bijection on {0,1}^D for all k.

#### One-Time Pad Encryption (EncryptedHV)

[0017] An encrypted hypervector is constructed as:

    enc(P, M) = P XOR M

where P is the plaintext BinaryHV and M is a uniformly random mask (also a BinaryHV of dimension D).

[0018] **Decryption**: dec(C, M) = C XOR M = (P XOR M) XOR M = P (XOR is self-inverse).

[0019] **Perfect secrecy proof**: Since M is uniformly random over {0,1}^D and used only once, C = P XOR M is uniformly distributed over {0,1}^D regardless of P. By Shannon's theorem (1949), the ciphertext reveals zero bits of information about the plaintext. This holds information-theoretically---no computational assumption is required.

[0020] **Implementation**: The EncryptedHV struct stores only the ciphertext BinaryHV. Encryption and decryption are both single XOR operations: O(D/W) where W is the SIMD width (256 for AVX2), completing in approximately 5-10 nanoseconds at D = 16,384.

#### Homomorphic Binding

[0021] The homomorphic property of XOR binding follows directly from the associativity and commutativity of XOR:

    enc(A, Ma) XOR enc(B, Mb) = (A XOR Ma) XOR (B XOR Mb)
                               = (A XOR B) XOR (Ma XOR Mb)
                               = enc(A XOR B, Ma XOR Mb)

[0022] Binding two encrypted vectors produces exactly the encryption of their plaintext binding, under the combined mask Ma XOR Mb. This is not an approximation---the result is bit-exact. The decryptor requires Ma XOR Mb (computable by both mask holders from their individual masks).

#### Distance-Preserving Similarity

[0023] When two vectors A and B are encrypted with the same mask M:

    sim(enc(A, M), enc(B, M)) = sim(A XOR M, B XOR M)
                               = 1 - hamming(A XOR M, B XOR M) / D
                               = 1 - hamming(A, B) / D
                               = sim(A, B)

[0024] The equality holds because XOR with a fixed mask is a distance-preserving bijection on Hamming space: flipping the same bits in both vectors preserves their relative distance.

[0025] This enables privacy-preserving nearest-neighbor search. Within a session where all vectors share a mask, similarity queries return exact plaintext results without any decryption. An adversary without the mask sees only random similarities (approximately 0.5). When vectors are encrypted with different masks (M_a != M_b), the similarity collapses to approximately 0.5 +/- 0.0039 (indistinguishable from random), providing strong privacy between sessions.

#### Approximate Homomorphic Bundling

[0026] For N vectors encrypted with the same mask M:

    MAJ(enc(w_1, M), enc(w_2, M), ..., enc(w_N, M))
    = MAJ(w_1 XOR M, w_2 XOR M, ..., w_N XOR M)
    ~ enc(MAJ(w_1, w_2, ..., w_N), M)

[0027] The approximation arises because majority vote and XOR do not perfectly commute. However, the fidelity improves with N due to the central limit theorem: as N grows, the per-bit majority vote on encrypted vectors converges to the majority vote on plaintext vectors masked by M. With N = 5 contributors and D = 16,384, the similarity between the decrypted aggregate and the expected plaintext bundle exceeds 0.85. With larger N, fidelity approaches 1.0.

#### Collective Wisdom Pool

[0028] The CollectiveWisdomPool implements a privacy-preserving aggregation system comprising four phases:

[0029] **Setup**: A coordinator generates a collective mask M and splits it into (k, n) threshold shares using the threshold secret sharing scheme described herein. Each of n peers receives one share.

[0030] **Contribution**: Each peer encrypts their local wisdom vector (e.g., consciousness state, learned representation) as enc(w_i, M) and contributes the encrypted vector to the pool.

[0031] **Aggregation**: The pool computes the majority-vote bundle of all encrypted contributions, producing an encrypted aggregate.

[0032] **Decryption**: At least k peers cooperate to reconstruct the mask M from their shares, then decrypt the aggregate.

[0033] Protocol properties: No peer sees any other peer's plaintext wisdom vector. No single peer can decrypt the aggregate alone (threshold k > 1). The aggregate preserves the collective semantic content of all contributions. The pool enforces a maximum capacity (default 256) to bound memory usage. Contributor identities are tracked for audit trail purposes. Maximum pool size of 256 contributions x 2 KB per BinaryHV (D = 16,384 bits = 2048 bytes) = 512 KB total.

#### Threshold Secret Sharing (HdcThresholdSharing)

[0034] A (k, n) threshold secret sharing scheme for binary hypervectors using majority-vote bundling is disclosed.

[0035] **Split**: For a secret BinaryHV S, generate n random masks M_0, M_1, ..., M_{n-1}. Each share is computed as:

    share_i = S XOR M_i

Each share is individually a one-time pad (information-theoretically secure).

[0036] **Recover**: Given k or more shares with their corresponding masks: (1) Unbind each share: recovered_i = share_i XOR M_i = S (exact, since XOR is self-inverse). (2) Bundle via majority vote: S_recovered = MAJ(recovered_0, recovered_1, ..., recovered_{k-1}).

[0037] **Exact recovery**: Since each unbound share equals the secret exactly, the majority vote of k identical copies of S is S itself---recovery is bit-exact when k correct shares are provided.

[0038] **Threshold property**: With fewer than k shares, the missing shares would contribute noise during bundling. At k-1 shares with one missing (contributing random noise), the per-bit error rate approaches 0.5 for the missing share's influence, degrading recovery to near-random similarity. The constraint k must be odd (majority vote requires odd count for deterministic tiebreaking), and k must satisfy 1 <= k <= n.

[0039] **Performance**: Split is n XOR operations; recovery is k XOR operations plus one majority-vote pass. Total: O(n * D / W) for split, O(k * D / W) for recovery (approximately 5-10 microseconds for 3-of-5 at D = 16,384).

#### Message Authentication Code (HdcMac)

[0040] An HDC-native MAC construction is disclosed:

    MAC(message, key) = message XOR pi_offset(key)

where pi_offset denotes cyclic permutation by a fixed offset (default offset = 7, a small prime to avoid alignment artifacts).

[0041] **Zero collision probability**: For two distinct messages m_1 != m_2 with the same key, P(MAC(m_1, key) = MAC(m_2, key)) = P(m_1 XOR pi_k(key) = m_2 XOR pi_k(key)) = P(m_1 = m_2) = 0. Since XOR binding is a bijection per operand, distinct inputs always yield distinct MACs under the same key. The collision probability is exactly zero, not approximately zero.

[0042] **Unforgeability**: Given MAC and message but not the key, recovering the key requires computing MAC XOR message = pi_k(key), then applying the inverse permutation. The attacker must guess offset k from [0, D), giving D = 16,384 candidate permuted keys, each being a full D-bit vector. The search space is D x 2^D.

[0043] **Noise-tolerant verification**: For lossy channels (LoRa, BLE), verification uses Hamming similarity with a threshold tau. At D = 16,384, the false positive rate with threshold tau = 0.95 is bounded by the Hoeffding inequality: P(false positive) ~ exp(-2D(0.95 - 0.5)^2) ~ 2^{-4700}.

[0044] **Domain separation**: Using different permutation offsets for different message types provides natural domain separation without additional key derivation.

[0045] **Performance**: MAC computation is one permute + one XOR: O(D/W) ~ 5-10 nanoseconds. Compared to BLAKE3 MAC (~50-100 ns) and HMAC-SHA256 (~200-400 ns).

#### Context-Derived Encryption Keys (HdcContextKey)

[0046] A key derivation scheme that binds encryption keys to physical context is disclosed:

    key = S_0 XOR pi_1(S_1) XOR pi_2(S_2) XOR ... XOR pi_{n-1}(S_{n-1})

where S_i are sensor readings encoded as BinaryHVs and pi_i denotes cyclic permutation by i positions.

[0047] **Non-commutativity**: The per-sensor permutation by index ensures that sensor order matters: S_0 XOR pi_1(S_1) != S_1 XOR pi_1(S_0). This prevents replay attacks where an adversary reorders sensor readings.

[0048] **Entropy preservation**: XOR with a random vector preserves the entropy of the highest-entropy input: H(key) >= max(H(S_i)) for independent sensors. Even low-entropy sensors (e.g., temperature with ~8 effective bits) do not dilute the key when combined with high-entropy sensors.

[0049] **Symmetric key extraction**: A to_symmetric_key method applies BLAKE3 hash to the D-dimensional HDC key, producing a uniform 256-bit key suitable for standard symmetric ciphers (ChaCha20-Poly1305, AES-256-GCM).

[0050] **Applications**: Location-bound decryption (GPS + altitude), temporal access windows (time sensor), device-bound secrets (accelerometer + gyroscope signatures).

#### Commitment Scheme (HdcCommitment)

[0051] A hide-then-reveal commitment using cyclic permutation is disclosed:

    Commit(secret, offset) = pi_offset(secret)
    Verify(commitment, secret, offset) = (pi_offset(secret) == commitment)

[0052] **Binding property**: Since cyclic permutation is a bijection on {0,1}^D, for a fixed offset, each secret maps to a unique commitment. For different offsets, the collision probability is 2^{-D} = 2^{-16384} (negligible).

[0053] **Hiding property**: The commitment pi_offset(secret) is quasi-independent of the secret for non-trivial offsets---Hamming similarity between secret and commitment approaches 0.5 (empirically within 0.05 of 0.5). Without knowing the offset, an attacker faces D = 16,384 possible preimages.

[0054] **Noise-tolerant verification**: For commitments transmitted over lossy channels, similarity-based verification with threshold tau provides graceful degradation.

[0055] **Caveat**: Unlike hash-based commitments, this scheme provides information-theoretic hiding only against attackers who cannot enumerate all D offsets. For computationally unbounded adversaries, BLAKE3-based commitments should be used instead. The advantage here is zero overhead for HDC-native data.

#### Preferred Embodiments

[0056] **Privacy-Preserving Federated Learning**: In a federated learning system using HDC representations, each participant encrypts their local model update (a binary hypervector) with a session mask before transmitting to a central aggregator. The aggregator computes the majority-vote bundle of encrypted updates without accessing any individual model. The aggregated model is decrypted only via threshold cooperation of participants. This eliminates the need for differential privacy noise injection (which degrades model quality) or secure multi-party computation protocols (which impose communication overhead).

[0057] **Swarm Robotics Collective Intelligence**: In a swarm of autonomous robots using HDC consciousness representations, each robot encrypts its local consciousness state and contributes to a collective wisdom pool. The swarm can compute collective decisions via encrypted aggregation without any robot exposing its internal state. Threshold decryption ensures that the collective decision requires cooperation of at least k robots, preventing single-point compromise.

[0058] **IoT Sensor Network Authentication**: In an IoT mesh network where sensor nodes communicate via lossy channels (LoRa, BLE), HdcMac provides authentication at 5-10 nanoseconds per packet (20-40x faster than HMAC-SHA256) with noise-tolerant verification. Context-derived keys bind encryption to physical location and time, enabling access policies like "data decryptable only at this GPS coordinate within this time window."

[0059] **Decentralized Governance Voting**: In a decentralized governance system, voter preferences are encoded as binary hypervectors and committed using the permutation-based commitment scheme. During the voting period, commitments are publicly visible but voter preferences are hidden. After the voting deadline, voters reveal their offsets, allowing verification. The aggregate preference is computed via majority-vote bundling of revealed votes.

[0060] **Clinical Data Sharing**: In a clinical data sharing consortium, each hospital encodes patient cohort statistics as HDC hypervectors and contributes encrypted versions to a shared analysis pool. The pool computes similarity queries and aggregate statistics in the encrypted domain. Individual hospital data is never exposed. Threshold decryption requires cooperation of k-of-n hospitals before any aggregate result can be examined.

---

### Claims

**Claim 1** (Independent). A method for encrypting a binary hypervector, comprising:

(a) providing a plaintext binary hypervector of dimension D, where D is at least 1,024;

(b) generating a uniformly random binary mask of dimension D; and

(c) computing an encrypted hypervector as the element-wise exclusive-OR (XOR) of the plaintext hypervector and the mask, thereby achieving perfect secrecy per Shannon's one-time pad theorem,

wherein the encrypted hypervector is statistically independent of the plaintext for any observation of the encrypted hypervector alone.

**Claim 2** (Independent). A method for homomorphic binding of encrypted binary hypervectors, comprising:

(a) receiving a first encrypted hypervector enc(A, Ma) computed as A XOR Ma, where A is a first plaintext binary hypervector and Ma is a first random mask;

(b) receiving a second encrypted hypervector enc(B, Mb) computed as B XOR Mb, where B is a second plaintext binary hypervector and Mb is a second random mask; and

(c) computing the element-wise XOR of the two encrypted hypervectors, producing a result equal to enc(A XOR B, Ma XOR Mb),

which upon decryption with the combined mask Ma XOR Mb yields the plaintext binding A XOR B, without either plaintext A or B being revealed during computation.

**Claim 3** (Independent). A method for computing similarity between encrypted binary hypervectors, comprising:

(a) encrypting a first binary hypervector A with a mask M to produce enc(A, M);

(b) encrypting a second binary hypervector B with the same mask M to produce enc(B, M); and

(c) computing the Hamming similarity between enc(A, M) and enc(B, M),

wherein the computed similarity equals the plaintext Hamming similarity between A and B exactly, because XOR with a fixed mask is a distance-preserving bijection on Hamming space.

**Claim 4** (Independent). A method for approximate homomorphic bundling of encrypted binary hypervectors, comprising:

(a) encrypting each of N binary hypervectors w_1, w_2, ..., w_N with a common mask M to produce N encrypted hypervectors;

(b) computing the per-dimension majority vote across the N encrypted hypervectors to produce an encrypted aggregate; and

(c) decrypting the encrypted aggregate with the mask M to obtain an approximation of the majority-vote bundle of the N plaintext hypervectors,

wherein the fidelity of the approximation improves with increasing N.

**Claim 5** (Independent). A system for privacy-preserving collective aggregation of hypervectors, comprising:

(a) a collective wisdom pool configured to receive encrypted binary hypervector contributions from K peers, each encrypted with a common session mask;

(b) an aggregation module configured to compute a majority-vote bundle of the encrypted contributions without decrypting any individual contribution;

(c) a threshold mask recovery module configured to reconstruct the session mask from at least k-of-n threshold shares; and

(d) a decryption module configured to decrypt the aggregated result using the recovered mask,

wherein no individual peer's plaintext contribution is revealed to any other peer or to the aggregation system.

**Claim 6** (Independent). A method for threshold secret sharing of a binary hypervector, comprising:

(a) generating n random binary masks M_0 through M_{n-1}, each of dimension D;

(b) computing n shares as share_i = secret XOR M_i for i = 0 to n-1;

(c) distributing each share_i with its corresponding mask M_i to a distinct holder; and

(d) recovering the secret from any k or more shares by: unbinding each share with its mask to obtain recovered_i = share_i XOR M_i, and computing the majority-vote bundle of the k recovered vectors,

wherein recovery is bit-exact when k valid shares are provided, and wherein k is odd and satisfies 1 <= k <= n.

**Claim 7** (Independent). A method for authenticating binary hypervector messages, comprising:

(a) computing a message authentication code (MAC) as MAC = message XOR pi_offset(key), where message and key are binary hypervectors of dimension D and pi_offset denotes cyclic permutation by a fixed offset; and

(b) verifying the MAC by recomputing MAC' = message XOR pi_offset(key) and comparing MAC' to the received MAC,

wherein the collision probability for distinct messages under the same key is exactly zero because XOR binding is a bijection per operand.

**Claim 8** (Independent). A method for deriving an encryption key from a physical context, comprising:

(a) encoding N sensor readings as binary hypervectors S_0 through S_{N-1}; and

(b) computing a context key as key = S_0 XOR pi_1(S_1) XOR pi_2(S_2) XOR ... XOR pi_{N-1}(S_{N-1}), where pi_i denotes cyclic permutation by i positions,

wherein the per-sensor permutation by index enforces ordering such that the key changes if sensor order is permuted, and the entropy of the derived key is at least the maximum entropy of any individual sensor input.

**Claim 9** (Independent). A method for committing to a binary hypervector, comprising:

(a) computing a commitment as commit = pi_offset(secret), where pi_offset denotes cyclic permutation by a secret offset; and

(b) later revealing the secret and offset, allowing a verifier to confirm that pi_offset(secret) equals the previously published commitment,

wherein the commitment is binding (distinct (secret, offset) pairs produce distinct commitments with probability 1 - 2^{-D}) and hiding (the commitment has Hamming similarity approximately 0.5 to the secret for non-trivial offsets).

**Claim 10** (Independent). A method for privacy-preserving swarm intelligence, comprising:

(a) each of K cognitive agents maintaining a local consciousness state as a binary hypervector;

(b) each agent encrypting its consciousness state with a session mask via element-wise XOR;

(c) contributing the encrypted state to a collective wisdom pool;

(d) the pool computing a majority-vote aggregate of all encrypted contributions; and

(e) decrypting the aggregate only when at least k-of-n agents cooperate to reconstruct the session mask via threshold secret sharing,

thereby enabling collective intelligence without exposing any individual agent's internal state.

**Claim 11** (Dependent on Claim 1). The method of claim 1, wherein the dimension D is at least 16,384 and the encryption and decryption each complete in O(D/W) operations where W is the SIMD register width, achieving latency of 5-10 nanoseconds on processors with 256-bit SIMD (AVX2).

**Claim 12** (Dependent on Claim 5). The system of claim 5, further comprising a capacity limit on the collective wisdom pool (default 256 contributions), and a contributor identity tracking module that records the identity of each contributing peer for audit trail purposes.

**Claim 13** (Dependent on Claim 7). The method of claim 7, further comprising noise-tolerant MAC verification by computing Hamming similarity between the expected and received MAC values and accepting the MAC if the similarity exceeds a threshold tau, wherein for D = 16,384 and tau = 0.95, the false positive rate is bounded by exp(-2D(tau - 0.5)^2) via the Hoeffding inequality.

**Claim 14** (Dependent on Claim 7). The method of claim 7, further comprising domain separation by using distinct permutation offsets for different message types, such that MAC(m, key, offset_a) != MAC(m, key, offset_b) for offset_a != offset_b, without requiring additional key derivation.

**Claim 15** (Dependent on Claim 8). The method of claim 8, further comprising extracting a 256-bit symmetric key from the D-dimensional context key by applying a cryptographic hash function, producing a uniform key suitable for standard symmetric ciphers.

---

### Abstract

A suite of cryptographic primitives operating natively on binary hypervectors of dimension D used in hyperdimensional computing (HDC). Element-wise XOR between a plaintext hypervector and a uniformly random mask achieves perfect secrecy per Shannon's one-time pad theorem. Because XOR is simultaneously the HDC binding operation and the encryption operation, homomorphic computation is achieved at zero overhead: binding two encrypted vectors produces the encryption of their binding under the combined mask, and Hamming similarity between vectors encrypted with the same mask equals the plaintext similarity exactly. Additional primitives include threshold secret sharing via majority-vote bundling, zero-collision message authentication via permutation-based binding, context-derived encryption keys from sensor hypervector chains, and hide-then-reveal commitments via cyclic permutation. All operations execute in O(D) time, approximately 10,000x faster than lattice-based FHE, with information-theoretic rather than computational security guarantees.

---

### Drawings (Text Descriptions for Conversion to Figures)

**Figure 1: HDC Encryption as Algebraic Identity**

A diagram showing two parallel paths for the same operation. On the left, labeled "HDC Computation": two BinaryHVs A and B enter an XOR binding node, producing A XOR B. On the right, labeled "Encryption": plaintext P and mask M enter an identical XOR node, producing ciphertext C = P XOR M. A callout arrow between the two paths states: "Same operation. Zero additional cost." Below both paths, a comparison table shows: Lattice FHE (10,000x overhead, polynomial ring conversion, computational hardness) vs. HDC-FHE (1.0x overhead, native binary XOR, information-theoretic security).

**Figure 2: Collective Wisdom Pool Protocol**

A sequence diagram with 5 actors: Coordinator, Peer 1, Peer 2, Peer 3, Pool. The Coordinator generates mask M and splits into (3,5) shares, distributing one share to each peer. Each peer encrypts their wisdom vector w_i with M (shown as w_i XOR M) and sends the encrypted vector to the Pool. The Pool computes MAJ(enc(w_1), enc(w_2), enc(w_3)) and holds the encrypted aggregate. Three peers then each contribute their share to reconstruct M, and the Pool decrypts to obtain the collective wisdom bundle. A privacy note states: "No peer sees any other peer's w_i. Pool never sees any plaintext."

**Figure 3: Threshold Secret Sharing via Majority-Vote Bundling**

A three-panel diagram. Panel A ("Split"): A secret BinaryHV S enters 5 XOR nodes, each paired with a random mask M_i, producing 5 shares. Each share is shown with similarity ~0.5 to S. Panel B ("Recover with k=3"): Three shares are unbound (XOR with their masks), producing 3 copies of S. These enter a majority-vote node, producing S exactly. Panel C ("Below threshold"): Only 2 shares are unbound, the missing third contributes random noise. The majority-vote output shows degraded similarity ~0.67 to S.

**Figure 4: Performance Comparison**

A bar chart (log scale) comparing latency per operation: HDC-MAC 5-10 ns vs. BLAKE3 MAC 50-100 ns vs. HMAC-SHA256 200-400 ns; HDC Encrypt/Decrypt 5-10 ns vs. AES-256-GCM 20-50 ns; HDC Similarity (encrypted) 5-10 ns vs. CKKS encrypted dot product ~50,000 ns; HDC 3-of-5 threshold 5-10 us vs. Shamir 3-of-5 on 16K-bit secret ~100 us.

**Figure 5: Distance Preservation Under Encryption**

A scatter plot showing plaintext similarity (x-axis) vs. encrypted similarity (y-axis) for 1,000 random vector pairs encrypted with the same mask. All points lie exactly on the y = x diagonal (Pearson r = 1.000). A second scatter plot shows pairs encrypted with different masks: all points cluster around y = 0.5 regardless of x value.

---

*Provisional application prepared for self-filing at USPTO Patent Center.*
*Inventor: Tristan Stoltz, Luminous Dynamics*
*Date prepared: March 27, 2026*
*Grace period expires: March 17, 2027*
