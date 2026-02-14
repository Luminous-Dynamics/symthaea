/**
 * DHT Storage Integration Example
 *
 * Demonstrates how to configure the Mycelix SDK with Holochain DHT backend
 * for persistent, distributed storage of epistemic data.
 *
 * @see https://mycelix.net/docs/sdk/storage
 */
import {
  createEpistemicStorage,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type HolochainClient,
  type EpistemicStorageConfig,
} from '@mycelix/sdk/storage';

// =============================================================================
// Basic Usage: Connect to Holochain Conductor
// =============================================================================

/**
 * Create a storage instance connected to Holochain DHT.
 *
 * Prerequisites:
 * - Holochain conductor running with epistemic_storage hApp installed
 * - @holochain/client package installed
 */
async function basicUsage() {
  // Import Holochain client (installed separately)
  // import { AppWebsocket } from '@holochain/client';

  // Connect to Holochain conductor
  // const client = await AppWebsocket.connect('ws://localhost:8888');

  // For this example, we'll use a mock client
  const mockClient: HolochainClient = {
    callZome: async (request) => {
      console.log(`Zome call: ${request.fn_name}`, request.payload);
      return null;
    },
  };

  // Configure storage with DHT backend
  const config: EpistemicStorageConfig = {
    agentId: 'agent:alice',
    dhtBackend: {
      client: mockClient,
      zomeName: 'epistemic_storage', // Default zome name
      dnaRole: 'epistemic_storage', // Default DNA role
      enableLocalCache: true, // Cache DHT reads locally
      localCacheTtlMs: 60000, // Cache for 1 minute
    },
  };

  const storage = createEpistemicStorage(config);

  // Verify DHT is enabled
  console.log('DHT enabled:', storage.isDHTEnabled());

  return storage;
}

// =============================================================================
// M-Level Routing: Data Goes to the Right Backend
// =============================================================================

async function mLevelRouting() {
  const storage = await basicUsage();

  // M0 (Ephemeral) → Memory backend (fast, lost on restart)
  const m0Receipt = await storage.store(
    'session:temp-token',
    { token: 'xyz123', expires: Date.now() + 3600000 },
    {
      empirical: EmpiricalLevel.E0_Unverified,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M0_Ephemeral,
    },
    { schema: { id: 'session', version: '1.0.0' } }
  );
  console.log('M0 stored in:', m0Receipt.tier.backend); // 'memory'

  // M1 (Temporal) → Local backend (persists to disk/IndexedDB)
  const m1Receipt = await storage.store(
    'cache:user-preferences',
    { theme: 'dark', fontSize: 14 },
    {
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M1_Temporal,
    },
    { schema: { id: 'preferences', version: '1.0.0' } }
  );
  console.log('M1 stored in:', m1Receipt.tier.backend); // 'local'

  // M2 (Persistent) → DHT backend (distributed, replicated)
  const m2Receipt = await storage.store(
    'profile:alice',
    { name: 'Alice', bio: 'Holochain developer', joined: Date.now() },
    {
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    },
    { schema: { id: 'profile', version: '1.0.0' } }
  );
  console.log('M2 stored in:', m2Receipt.tier.backend); // 'dht'

  // M3 (Immutable) → DHT with higher replication
  const m3Receipt = await storage.store(
    'constitution:v1',
    {
      title: 'Community Constitution',
      ratified: Date.now(),
      signatories: ['alice', 'bob', 'carol'],
    },
    {
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N3_Axiomatic,
      materiality: MaterialityLevel.M3_Immutable,
    },
    { schema: { id: 'constitution', version: '1.0.0' } }
  );
  console.log('M3 stored in:', m3Receipt.tier.backend); // 'dht'

  storage.dispose();
}

// =============================================================================
// Retrieve Data by Key or CID
// =============================================================================

async function retrieveData() {
  const storage = await basicUsage();

  // Store some data
  const receipt = await storage.store(
    'document:readme',
    { content: 'Hello, Mycelix!' },
    {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    },
    { schema: { id: 'document', version: '1.0.0' } }
  );

  // Retrieve by key
  const byKey = await storage.retrieve<{ content: string }>('document:readme');
  console.log('Retrieved:', byKey?.data.content);

  // Retrieve by CID (content-addressed)
  const byCid = await storage.retrieveByCid<{ content: string }>(receipt.cid);
  console.log('Retrieved by CID:', byCid?.data.content);

  storage.dispose();
}

// =============================================================================
// N-Level Access Control
// =============================================================================

async function accessControl() {
  const storage = await basicUsage();

  // N0 (Personal) - Only owner can access
  await storage.store(
    'private:diary',
    { entry: 'Today I learned about Holochain...' },
    {
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N0_Personal, // Private to owner
      materiality: MaterialityLevel.M2_Persistent,
    },
    { schema: { id: 'diary', version: '1.0.0' } }
  );

  // N1 (Communal) - Requires CapBAC (Capability-Based Access Control)
  await storage.store(
    'shared:project-notes',
    { notes: 'Meeting at 3pm' },
    {
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N1_Communal, // Shared with capability holders
      materiality: MaterialityLevel.M2_Persistent,
    },
    { schema: { id: 'notes', version: '1.0.0' } }
  );

  // N2 (Network) - Public to all network participants
  await storage.store(
    'public:announcement',
    { message: 'Welcome to Mycelix!' },
    {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network, // Public
      materiality: MaterialityLevel.M2_Persistent,
    },
    { schema: { id: 'announcement', version: '1.0.0' } }
  );

  storage.dispose();
}

// =============================================================================
// Direct DHT Backend Access
// =============================================================================

async function directDHTAccess() {
  const storage = await basicUsage();

  // Get direct access to DHT backend for advanced operations
  const dhtBackend = storage.getDHTBackend();

  if (dhtBackend) {
    // Check replication status
    const replStatus = await dhtBackend.getReplicationStatus('profile:alice');
    console.log('Replication status:', replStatus);

    // Get DHT statistics
    const stats = await dhtBackend.stats();
    console.log('DHT stats:', stats);

    // List all keys
    const keys = await dhtBackend.keys();
    console.log('DHT keys:', keys);
  }

  storage.dispose();
}

// =============================================================================
// Storage Statistics
// =============================================================================

async function getStatistics() {
  const storage = await basicUsage();

  // Get comprehensive storage statistics
  const stats = await storage.getStats();

  console.log('Storage Statistics:');
  console.log('  Total items:', stats.totalItems);
  console.log('  Total size:', stats.totalSizeBytes, 'bytes');
  console.log('  Items by backend:');
  console.log('    Memory:', stats.itemsByBackend.memory);
  console.log('    Local:', stats.itemsByBackend.local);
  console.log('    DHT:', stats.itemsByBackend.dht);

  storage.dispose();
}

// =============================================================================
// Run Examples
// =============================================================================

async function main() {
  console.log('=== Mycelix DHT Storage Examples ===\n');

  console.log('1. Basic Usage');
  await basicUsage();

  console.log('\n2. M-Level Routing');
  await mLevelRouting();

  console.log('\n3. Retrieve Data');
  await retrieveData();

  console.log('\n4. Access Control');
  await accessControl();

  console.log('\n5. Direct DHT Access');
  await directDHTAccess();

  console.log('\n6. Statistics');
  await getStatistics();

  console.log('\n=== Done ===');
}

main().catch(console.error);
