# symtropy-net-core

Permissively-licensed (Apache-2.0 OR MIT) core types and traits for symtropy-net.

## Features
- `PeerId`: Unique identifier for networking peers.
- `PeerStateCore`: Core state tracking for peers.
- `StateAuthority`: Classification of state authority (Local, Replicated, Consensus).
- `SyncableState`: Container for state that can be synced across peers.
- `SpatialAuthority`: System for determining which peer computes physics for which bodies based on proximity.

This crate contains NO AGPL-licensed dependencies and is safe for use in proprietary applications.
