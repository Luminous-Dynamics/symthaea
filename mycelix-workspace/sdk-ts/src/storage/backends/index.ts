/**
 * UESS Storage Backends
 *
 * Backend adapters for different M-levels:
 * - M0: Memory (ephemeral) - In-memory with TTL
 * - M1: Local (temporal) - Browser/Node local storage with automatic persistence
 * - M2: DHT (persistent) - Holochain DHT integration
 * - M3: IPFS (immutable) - Content-addressed IPFS storage
 */

export { MemoryBackend, createMemoryBackend, type StorageBackendAdapter } from './memory.js';
export { LocalBackend, createLocalBackend, createEphemeralLocalBackend, type LocalBackendOptions } from './local.js';
export { DHTBackend, createDHTBackend, type DHTBackendConfig, type HolochainClient } from './dht.js';
export { IPFSBackend, createIPFSBackend, type IPFSBackendConfig } from './ipfs.js';

// Persistence utilities (for advanced use cases)
export {
  createPersistenceAdapter,
  isBrowser,
  isNode,
  type PersistenceAdapter,
  IndexedDBPersistence,
  FileSystemPersistence,
  LocalStoragePersistence,
  InMemoryPersistence,
} from './persistence.js';
