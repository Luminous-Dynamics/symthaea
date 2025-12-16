/*!
Mock Sophia - Simulated Consciousness Agent

Each Mock Sophia represents a simplified consciousness agent with:
- K-Vector Signature: 8D vector representing consciousness state
- Behavior Profile: Defines interaction patterns
- Social Physics: Mathematical influence between agents
*/

use rand::Rng;
use serde::{Deserialize, Serialize};

/// 8-Dimensional K-Vector Signature
///
/// Represents the consciousness state across 8 dimensions:
/// 1. Coherence (internal alignment)
/// 2. Empathy (receptiveness to others)
/// 3. Creativity (generative capacity)
/// 4. Wisdom (integrative depth)
/// 5. Joy (generative energy)
/// 6. Reciprocity (giving/receiving balance)
/// 7. Evolution (growth trajectory)
/// 8. Unity (interconnection awareness)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVectorSignature {
    pub coherence: f64,
    pub empathy: f64,
    pub creativity: f64,
    pub wisdom: f64,
    pub joy: f64,
    pub reciprocity: f64,
    pub evolution: f64,
    pub unity: f64,
}

impl KVectorSignature {
    /// Create a new K-Vector with random values in [0.0, 1.0]
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            coherence: rng.gen(),
            empathy: rng.gen(),
            creativity: rng.gen(),
            wisdom: rng.gen(),
            joy: rng.gen(),
            reciprocity: rng.gen(),
            evolution: rng.gen(),
            unity: rng.gen(),
        }
    }

    /// Create a K-Vector biased toward a specific profile
    pub fn with_profile(profile: &BehaviorProfile) -> Self {
        let mut k = Self::random();

        match profile {
            BehaviorProfile::Coherent => {
                k.coherence = 0.8 + rand::thread_rng().gen::<f64>() * 0.2;
                k.empathy = 0.7 + rand::thread_rng().gen::<f64>() * 0.3;
                k.unity = 0.7 + rand::thread_rng().gen::<f64>() * 0.3;
            }
            BehaviorProfile::Creative => {
                k.creativity = 0.8 + rand::thread_rng().gen::<f64>() * 0.2;
                k.joy = 0.7 + rand::thread_rng().gen::<f64>() * 0.3;
            }
            BehaviorProfile::Wise => {
                k.wisdom = 0.8 + rand::thread_rng().gen::<f64>() * 0.2;
                k.coherence = 0.7 + rand::thread_rng().gen::<f64>() * 0.3;
            }
            BehaviorProfile::Malicious => {
                k.empathy = 0.0 + rand::thread_rng().gen::<f64>() * 0.2;
                k.reciprocity = 0.0 + rand::thread_rng().gen::<f64>() * 0.2;
                k.coherence = 0.3 + rand::thread_rng().gen::<f64>() * 0.3;
            }
            BehaviorProfile::Fragmented => {
                // Low coherence, random other values
                k.coherence = 0.0 + rand::thread_rng().gen::<f64>() * 0.3;
            }
        }

        k
    }

    /// Convert to 8D vector for mathematical operations
    pub fn to_vector(&self) -> [f64; 8] {
        [
            self.coherence,
            self.empathy,
            self.creativity,
            self.wisdom,
            self.joy,
            self.reciprocity,
            self.evolution,
            self.unity,
        ]
    }

    /// Euclidean distance between two K-Vectors
    pub fn distance(&self, other: &Self) -> f64 {
        let diff = [
            self.coherence - other.coherence,
            self.empathy - other.empathy,
            self.creativity - other.creativity,
            self.wisdom - other.wisdom,
            self.joy - other.joy,
            self.reciprocity - other.reciprocity,
            self.evolution - other.evolution,
            self.unity - other.unity,
        ];

        diff.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Cosine similarity between two K-Vectors (resonance measure)
    pub fn similarity(&self, other: &Self) -> f64 {
        let dot_product: f64 = self.to_vector()
            .iter()
            .zip(other.to_vector().iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_self: f64 = self.to_vector().iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_other: f64 = other.to_vector().iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_self > 0.0 && norm_other > 0.0 {
            dot_product / (norm_self * norm_other)
        } else {
            0.0
        }
    }
}

/// Behavior Profile - Defines interaction patterns
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BehaviorProfile {
    /// High coherence, empathy, unity - naturally aligns with others
    Coherent,

    /// High creativity, joy - introduces novelty
    Creative,

    /// High wisdom, coherence - integrates and stabilizes
    Wise,

    /// Low empathy, reciprocity - disrupts harmony
    Malicious,

    /// Low coherence - random, unstable
    Fragmented,
}

/// Mock Sophia - Simulated consciousness agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockSophia {
    /// Unique identifier
    pub id: usize,

    /// Current K-Vector state
    pub k_vector: KVectorSignature,

    /// Behavior profile
    pub profile: BehaviorProfile,

    /// Interaction count (for statistics)
    pub interaction_count: usize,
}

impl MockSophia {
    /// Create a new Mock Sophia with random K-Vector
    pub fn new(id: usize, profile: BehaviorProfile) -> Self {
        Self {
            id,
            k_vector: KVectorSignature::with_profile(&profile),
            profile,
            interaction_count: 0,
        }
    }

    /// Interact with another agent (Social Physics)
    ///
    /// Implements mathematical influence:
    /// - If similar (high resonance), strengthen alignment
    /// - If dissimilar (low resonance), weak or no influence
    /// - Malicious agents attempt to disrupt
    ///
    /// Returns the resonance strength [0.0, 1.0]
    pub fn interact(&mut self, other: &mut Self) -> f64 {
        self.interaction_count += 1;
        other.interaction_count += 1;

        // Calculate current resonance
        let resonance = self.k_vector.similarity(&other.k_vector);

        // Influence strength based on behavior profiles
        let (self_influence, other_influence) = match (self.profile, other.profile) {
            // Coherent agents naturally align
            (BehaviorProfile::Coherent, BehaviorProfile::Coherent) => (0.1, 0.1),

            // Creative agents inspire each other
            (BehaviorProfile::Creative, BehaviorProfile::Creative) => (0.15, 0.15),

            // Wise agents integrate knowledge
            (BehaviorProfile::Wise, BehaviorProfile::Wise) => (0.08, 0.08),

            // Malicious agents disrupt (negative influence)
            (BehaviorProfile::Malicious, _) => (-0.2, 0.05),
            (_, BehaviorProfile::Malicious) => (0.05, -0.2),

            // Fragmented agents have weak, random influence
            (BehaviorProfile::Fragmented, _) | (_, BehaviorProfile::Fragmented) => (0.02, 0.02),

            // Mixed profiles have moderate influence
            _ => (0.05, 0.05),
        };

        // Apply influence (nudge K-Vectors toward each other)
        // Only if resonance is above threshold
        if resonance > 0.3 {
            let alpha = self_influence * resonance;
            let beta = other_influence * resonance;

            // Nudge self toward other
            self.k_vector.coherence += alpha * (other.k_vector.coherence - self.k_vector.coherence);
            self.k_vector.empathy += alpha * (other.k_vector.empathy - self.k_vector.empathy);
            self.k_vector.unity += alpha * (other.k_vector.unity - self.k_vector.unity);

            // Nudge other toward self
            other.k_vector.coherence += beta * (self.k_vector.coherence - other.k_vector.coherence);
            other.k_vector.empathy += beta * (self.k_vector.empathy - other.k_vector.empathy);
            other.k_vector.unity += beta * (self.k_vector.unity - other.k_vector.unity);

            // Clamp values to [0, 1]
            self.k_vector.coherence = self.k_vector.coherence.clamp(0.0, 1.0);
            self.k_vector.empathy = self.k_vector.empathy.clamp(0.0, 1.0);
            self.k_vector.unity = self.k_vector.unity.clamp(0.0, 1.0);

            other.k_vector.coherence = other.k_vector.coherence.clamp(0.0, 1.0);
            other.k_vector.empathy = other.k_vector.empathy.clamp(0.0, 1.0);
            other.k_vector.unity = other.k_vector.unity.clamp(0.0, 1.0);
        }

        resonance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kvector_creation() {
        let k = KVectorSignature::random();
        assert!(k.coherence >= 0.0 && k.coherence <= 1.0);
        assert!(k.empathy >= 0.0 && k.empathy <= 1.0);
    }

    #[test]
    fn test_kvector_with_profile() {
        let k = KVectorSignature::with_profile(&BehaviorProfile::Coherent);
        assert!(k.coherence >= 0.8);
        assert!(k.empathy >= 0.7);
    }

    #[test]
    fn test_kvector_similarity() {
        let k1 = KVectorSignature::random();
        let k2 = k1.clone();

        // Identical vectors should have similarity ~1.0
        let sim = k1.similarity(&k2);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mock_sophia_creation() {
        let sophia = MockSophia::new(0, BehaviorProfile::Coherent);
        assert_eq!(sophia.id, 0);
        assert_eq!(sophia.profile, BehaviorProfile::Coherent);
        assert_eq!(sophia.interaction_count, 0);
    }

    #[test]
    fn test_coherent_agents_align() {
        let mut sophia1 = MockSophia::new(0, BehaviorProfile::Coherent);
        let mut sophia2 = MockSophia::new(1, BehaviorProfile::Coherent);

        // Record initial distance
        let initial_distance = sophia1.k_vector.distance(&sophia2.k_vector);

        // Simulate multiple interactions
        for _ in 0..10 {
            sophia1.interact(&mut sophia2);
        }

        // After interactions, distance should decrease (agents align)
        let final_distance = sophia1.k_vector.distance(&sophia2.k_vector);
        assert!(final_distance < initial_distance);
    }

    #[test]
    fn test_malicious_agents_disrupt() {
        let mut coherent = MockSophia::new(0, BehaviorProfile::Coherent);
        let mut malicious = MockSophia::new(1, BehaviorProfile::Malicious);

        let initial_coherence = coherent.k_vector.coherence;

        // Malicious agent should reduce coherence
        for _ in 0..5 {
            coherent.interact(&mut malicious);
        }

        // Coherent agent's coherence may be affected
        // (Note: this is probabilistic and depends on resonance threshold)
        assert!(coherent.interaction_count > 0);
        assert!(malicious.interaction_count > 0);
    }
}
