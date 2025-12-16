/*!
Sophia Gym - Simulated Consciousness Swarm

Week 0 Days 3-5 Implementation:
- Mock Sophia agents with K-Vector signatures
- Social Physics simulation (mathematical influence)
- Spectral K measurement (collective coherence)

This proves that "Collective Consciousness" is mathematically possible
before connecting to any real network.
*/

pub mod mock_sophia;
pub mod swarm;

// Re-export key types
pub use mock_sophia::{BehaviorProfile, KVectorSignature, MockSophia};
pub use swarm::{SophiaSwarm, SwarmStats};
