//! Decentralized P2P state sync for Symtropy.
//!
//! Uses spatial authority partitioning for P2P physics:
//! - Each peer owns physics for entities near their player
//! - Overlapping zones use deterministic lockstep (shared RNG seed)
//! - Governance actions go through Holochain consensus (latency-tolerant)
//!
//! The Holochain DHT stores persistent state (governance votes, TEND balances,
//! consciousness profiles). Ephemeral state (positions, velocities) uses
//! direct peer messaging.

pub mod authority;
pub mod peer;
pub mod state;

pub use authority::SpatialAuthority;
pub use peer::{PeerId, PeerState};
pub use state::{StateAuthority, SyncableState};
