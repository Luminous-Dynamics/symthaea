<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';

  // ============================================================================
  // Types
  // ============================================================================

  interface Node {
    id: string;
    name: string;
    location: string;
    status: 'online' | 'offline' | 'syncing' | 'degraded';
    happs: string[];
    peers: number;
    uptime: number;
    lastSeen: number;
    trustScore: number;
  }

  interface HappHealth {
    name: string;
    nodes: number;
    status: 'healthy' | 'degraded' | 'offline';
    latency: number;
    throughput: number;
  }

  interface NetworkStats {
    totalNodes: number;
    onlineNodes: number;
    totalPeers: number;
    avgLatency: number;
    messagesPerSecond: number;
    byzantineDetections: number;
  }

  interface PeerConnection {
    source: string;
    target: string;
    latency: number;
    status: 'stable' | 'unstable' | 'new';
  }

  // ============================================================================
  // Stores
  // ============================================================================

  const stats = writable<NetworkStats>({
    totalNodes: 156,
    onlineNodes: 142,
    totalPeers: 1234,
    avgLatency: 45,
    messagesPerSecond: 2456,
    byzantineDetections: 3,
  });

  const nodes = writable<Node[]>([
    { id: 'node-001', name: 'portland-hub', location: 'Portland, OR', status: 'online', happs: ['core', 'marketplace', 'mail'], peers: 24, uptime: 0.999, lastSeen: Date.now(), trustScore: 0.96 },
    { id: 'node-002', name: 'berlin-gateway', location: 'Berlin, DE', status: 'online', happs: ['core', 'governance', 'identity'], peers: 31, uptime: 0.998, lastSeen: Date.now(), trustScore: 0.94 },
    { id: 'node-003', name: 'tokyo-relay', location: 'Tokyo, JP', status: 'syncing', happs: ['core', 'edunet'], peers: 18, uptime: 0.987, lastSeen: Date.now() - 30000, trustScore: 0.91 },
    { id: 'node-004', name: 'austin-maker', location: 'Austin, TX', status: 'online', happs: ['core', 'fabrication', 'supplychain'], peers: 22, uptime: 0.995, lastSeen: Date.now(), trustScore: 0.93 },
    { id: 'node-005', name: 'london-finance', location: 'London, UK', status: 'degraded', happs: ['core', 'finance'], peers: 15, uptime: 0.956, lastSeen: Date.now() - 120000, trustScore: 0.88 },
    { id: 'node-006', name: 'sydney-bridge', location: 'Sydney, AU', status: 'online', happs: ['core', 'marketplace', 'supplychain'], peers: 12, uptime: 0.992, lastSeen: Date.now(), trustScore: 0.90 },
  ]);

  const happHealth = writable<HappHealth[]>([
    { name: 'Core', nodes: 142, status: 'healthy', latency: 23, throughput: 1456 },
    { name: 'Marketplace', nodes: 89, status: 'healthy', latency: 34, throughput: 678 },
    { name: 'Identity', nodes: 76, status: 'healthy', latency: 28, throughput: 345 },
    { name: 'Mail', nodes: 65, status: 'healthy', latency: 41, throughput: 234 },
    { name: 'Governance', nodes: 54, status: 'degraded', latency: 67, throughput: 123 },
    { name: 'EduNet', nodes: 45, status: 'healthy', latency: 52, throughput: 189 },
    { name: 'Fabrication', nodes: 34, status: 'healthy', latency: 38, throughput: 145 },
    { name: 'Supply Chain', nodes: 28, status: 'healthy', latency: 45, throughput: 98 },
  ]);

  const recentConnections = writable<PeerConnection[]>([
    { source: 'portland-hub', target: 'berlin-gateway', latency: 142, status: 'stable' },
    { source: 'berlin-gateway', target: 'tokyo-relay', latency: 198, status: 'stable' },
    { source: 'austin-maker', target: 'portland-hub', latency: 45, status: 'stable' },
    { source: 'london-finance', target: 'berlin-gateway', latency: 89, status: 'unstable' },
    { source: 'sydney-bridge', target: 'tokyo-relay', latency: 156, status: 'new' },
  ]);

  // ============================================================================
  // Helpers
  // ============================================================================

  // ============================================================================
  // Mesh Bridge Health
  // ============================================================================

  interface MeshBridgeHealth {
    status: 'running' | 'offline' | 'unknown';
    service: string;
    version: string;
    metrics: {
      messages_sent: number;
      messages_received: number;
      fragments_dropped: number;
      peers_seen: number;
      last_send_at: number;
      last_recv_at: number;
      poll_cycles: number;
      connection_failures: number;
    };
  }

  const meshHealth = writable<MeshBridgeHealth>({
    status: 'unknown',
    service: 'mycelix-mesh-bridge',
    version: '0.1.0',
    metrics: {
      messages_sent: 0,
      messages_received: 0,
      fragments_dropped: 0,
      peers_seen: 0,
      last_send_at: 0,
      last_recv_at: 0,
      poll_cycles: 0,
      connection_failures: 0,
    },
  });

  let meshTransport = 'unknown';

  async function fetchMeshHealth() {
    const meshUrl = (typeof window !== 'undefined' && window.location)
      ? `${window.location.protocol}//${window.location.hostname}:9100/health`
      : 'http://localhost:9100/health';
    try {
      const resp = await fetch(meshUrl, { signal: AbortSignal.timeout(3000) });
      if (resp.ok) {
        const data = await resp.json();
        meshHealth.set({
          status: data.status === 'running' ? 'running' : 'offline',
          service: data.service || 'mycelix-mesh-bridge',
          version: data.version || '0.1.0',
          metrics: data.metrics || {},
        });
        meshTransport = data.transport || 'unknown';
      } else {
        meshHealth.update(h => ({ ...h, status: 'offline' }));
      }
    } catch {
      meshHealth.update(h => ({ ...h, status: 'offline' }));
    }
  }

  function formatMeshTime(epochMs: number): string {
    if (!epochMs) return 'never';
    const seconds = Math.floor((Date.now() - epochMs) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  }

  let currentTime = new Date().toLocaleTimeString();
  let interval: ReturnType<typeof setInterval>;

  onMount(() => {
    fetchMeshHealth();
    interval = setInterval(() => {
      currentTime = new Date().toLocaleTimeString();
      simulateNetwork();
      fetchMeshHealth();
    }, 5000);
  });

  onDestroy(() => {
    if (interval) clearInterval(interval);
  });

  function simulateNetwork() {
    stats.update(s => ({
      ...s,
      messagesPerSecond: s.messagesPerSecond + Math.floor(Math.random() * 100) - 50,
      avgLatency: Math.max(20, Math.min(80, s.avgLatency + Math.floor(Math.random() * 10) - 5)),
    }));

    nodes.update(n => n.map(node => ({
      ...node,
      lastSeen: node.status === 'online' ? Date.now() : node.lastSeen,
      peers: node.peers + Math.floor(Math.random() * 3) - 1,
    })));
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'online': case 'healthy': case 'stable': return 'bg-green-500';
      case 'syncing': case 'new': return 'bg-blue-500';
      case 'degraded': case 'unstable': return 'bg-yellow-500';
      case 'offline': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  }

  function getStatusBadge(status: string): string {
    switch (status) {
      case 'online': case 'healthy': case 'stable': return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'syncing': case 'new': return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'degraded': case 'unstable': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'offline': return 'bg-red-500/20 text-red-400 border-red-500/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  }

  function formatUptime(uptime: number): string {
    return (uptime * 100).toFixed(2) + '%';
  }

  function formatLatency(ms: number): string {
    return ms + 'ms';
  }

  function formatThroughput(ops: number): string {
    if (ops >= 1000) return (ops / 1000).toFixed(1) + 'K/s';
    return ops + '/s';
  }

  function timeSince(ts: number): string {
    const seconds = Math.floor((Date.now() - ts) / 1000);
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
    return Math.floor(seconds / 3600) + 'h ago';
  }
</script>

<svelte:head>
  <title>Network | Mycelix Observatory</title>
</svelte:head>

<div class="text-white">
  <!-- Page Header -->
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl">🌐</span>
        <div>
          <h1 class="text-lg font-bold">Network Health</h1>
          <p class="text-xs text-gray-400">Node Connectivity & DHT Status</p>
        </div>
      </div>
      <div class="flex items-center gap-4">
        <div class="text-right">
          <p class="text-xs text-gray-400">Avg Latency</p>
          <p class="text-lg font-bold text-green-400">{$stats.avgLatency}ms</p>
        </div>
        <span class="text-gray-400 font-mono text-sm">{currentTime}</span>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <!-- Stats Grid -->
    <div class="grid grid-cols-2 md:grid-cols-6 gap-4 mb-8">
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Total Nodes</h3>
        <p class="text-2xl font-bold mt-1">{$stats.totalNodes}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Online</h3>
        <p class="text-2xl font-bold mt-1 text-green-400">{$stats.onlineNodes}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Peer Connections</h3>
        <p class="text-2xl font-bold mt-1">{$stats.totalPeers.toLocaleString()}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Avg Latency</h3>
        <p class="text-2xl font-bold mt-1 text-blue-400">{$stats.avgLatency}ms</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Messages/sec</h3>
        <p class="text-2xl font-bold mt-1 text-purple-400">{formatThroughput($stats.messagesPerSecond)}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Byzantine Alerts</h3>
        <p class="text-2xl font-bold mt-1 {$stats.byzantineDetections > 0 ? 'text-yellow-400' : 'text-green-400'}">
          {$stats.byzantineDetections}
        </p>
      </div>
    </div>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Node List -->
      <div class="bg-gray-800 rounded-lg border border-gray-700 lg:col-span-2">
        <div class="p-4 border-b border-gray-700 flex justify-between items-center">
          <h2 class="text-lg font-semibold flex items-center gap-2">
            <span>💻</span> Active Nodes
          </h2>
          <span class="text-sm text-gray-400">{$nodes.filter(n => n.status === 'online').length}/{$nodes.length} online</span>
        </div>
        <div class="p-4 space-y-3">
          {#each $nodes as node}
            <div class="p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700/70 transition-colors">
              <div class="flex justify-between items-start">
                <div>
                  <div class="flex items-center gap-2">
                    <span class={`w-2 h-2 rounded-full ${getStatusColor(node.status)} ${node.status === 'online' ? 'animate-pulse' : ''}`}></span>
                    <p class="font-medium">{node.name}</p>
                    <span class={`text-xs px-2 py-0.5 rounded border ${getStatusBadge(node.status)}`}>
                      {node.status}
                    </span>
                  </div>
                  <p class="text-xs text-gray-400 mt-1">{node.location}</p>
                </div>
                <div class="text-right">
                  <p class="text-sm text-green-400">{(node.trustScore * 100).toFixed(0)}% trust</p>
                  <p class="text-xs text-gray-400">{timeSince(node.lastSeen)}</p>
                </div>
              </div>
              <div class="mt-3 grid grid-cols-4 gap-4 text-xs">
                <div>
                  <p class="text-gray-400">Peers</p>
                  <p class="font-medium">{node.peers}</p>
                </div>
                <div>
                  <p class="text-gray-400">Uptime</p>
                  <p class="font-medium text-green-400">{formatUptime(node.uptime)}</p>
                </div>
                <div>
                  <p class="text-gray-400">hApps</p>
                  <p class="font-medium">{node.happs.length}</p>
                </div>
                <div>
                  <p class="text-gray-400">Running</p>
                  <p class="font-medium text-blue-400">{node.happs.slice(0, 2).join(', ')}{node.happs.length > 2 ? '...' : ''}</p>
                </div>
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- hApp Health & Connections -->
      <div class="space-y-6">
        <!-- hApp Health -->
        <div class="bg-gray-800 rounded-lg border border-gray-700">
          <div class="p-4 border-b border-gray-700">
            <h2 class="text-lg font-semibold flex items-center gap-2">
              <span>📦</span> hApp Health
            </h2>
          </div>
          <div class="p-4 space-y-2 max-h-64 overflow-y-auto">
            {#each $happHealth as happ}
              <div class="p-2 bg-gray-700/50 rounded-lg">
                <div class="flex justify-between items-center">
                  <div class="flex items-center gap-2">
                    <span class={`w-2 h-2 rounded-full ${getStatusColor(happ.status)}`}></span>
                    <span class="text-sm font-medium">{happ.name}</span>
                  </div>
                  <span class="text-xs text-gray-400">{happ.nodes} nodes</span>
                </div>
                <div class="flex justify-between text-xs mt-1">
                  <span class="text-gray-400">{formatLatency(happ.latency)}</span>
                  <span class="text-blue-400">{formatThroughput(happ.throughput)}</span>
                </div>
              </div>
            {/each}
          </div>
        </div>

        <!-- Recent Connections -->
        <div class="bg-gray-800 rounded-lg border border-gray-700">
          <div class="p-4 border-b border-gray-700">
            <h2 class="text-lg font-semibold flex items-center gap-2">
              <span>🔗</span> Recent Connections
            </h2>
          </div>
          <div class="p-4 space-y-2">
            {#each $recentConnections as conn}
              <div class="p-2 bg-gray-700/50 rounded-lg text-sm">
                <div class="flex items-center gap-2">
                  <span class="text-gray-400">{conn.source}</span>
                  <span class="text-gray-500">→</span>
                  <span class="text-gray-400">{conn.target}</span>
                  <span class={`ml-auto text-xs px-1.5 py-0.5 rounded ${getStatusBadge(conn.status)}`}>
                    {conn.status}
                  </span>
                </div>
                <p class="text-xs text-gray-500 mt-1">{formatLatency(conn.latency)}</p>
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>

    <!-- Network Topology Info -->
    <div class="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          <span>🌍</span> Geographic Distribution
        </h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">North America</span>
            <span>42 nodes</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Europe</span>
            <span>38 nodes</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Asia Pacific</span>
            <span>31 nodes</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Other</span>
            <span>31 nodes</span>
          </div>
        </div>
      </div>

      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          <span>⚡</span> DHT Performance
        </h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Gossip Interval</span>
            <span class="text-green-400">2s</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Validation Rate</span>
            <span class="text-green-400">98.7%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Sync Lag</span>
            <span class="text-blue-400">&lt;500ms</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Partition Events</span>
            <span class="text-green-400">0</span>
          </div>
        </div>
      </div>

      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          <span>🛡️</span> Byzantine Status
        </h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">BFT Threshold</span>
            <span class="text-green-400">34%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Current Tolerance</span>
            <span class="text-green-400">36.2%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Detected Bad Actors</span>
            <span class="text-yellow-400">3</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Cartel Detection</span>
            <span class="text-green-400">Active</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Mesh Bridge Status -->
    <div class="mt-6 bg-gray-800 rounded-lg border border-gray-700">
      <div class="p-4 border-b border-gray-700 flex justify-between items-center">
        <h2 class="text-lg font-semibold">Mesh Bridge (Offline Relay)</h2>
        <span class="text-xs px-2 py-1 rounded-full {$meshHealth.status === 'running' ? 'bg-green-500/20 text-green-400' : $meshHealth.status === 'offline' ? 'bg-red-500/20 text-red-400' : 'bg-gray-500/20 text-gray-400'}">
          {$meshHealth.status}
        </span>
      </div>
      {#if $meshHealth.status === 'running'}
        <div class="p-4">
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
            <div class="bg-gray-700/50 rounded p-3">
              <p class="text-xs text-gray-400 uppercase">Sent</p>
              <p class="text-xl font-bold text-blue-400">{$meshHealth.metrics.messages_sent}</p>
            </div>
            <div class="bg-gray-700/50 rounded p-3">
              <p class="text-xs text-gray-400 uppercase">Received</p>
              <p class="text-xl font-bold text-green-400">{$meshHealth.metrics.messages_received}</p>
            </div>
            <div class="bg-gray-700/50 rounded p-3">
              <p class="text-xs text-gray-400 uppercase">Peers Seen</p>
              <p class="text-xl font-bold text-purple-400">{$meshHealth.metrics.peers_seen}</p>
            </div>
            <div class="bg-gray-700/50 rounded p-3">
              <p class="text-xs text-gray-400 uppercase">Dropped</p>
              <p class="text-xl font-bold {$meshHealth.metrics.fragments_dropped > 0 ? 'text-yellow-400' : 'text-gray-400'}">
                {$meshHealth.metrics.fragments_dropped}
              </p>
            </div>
          </div>
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
            <div>
              <p class="text-xs text-gray-400">Transport</p>
              <p class="text-gray-200">{meshTransport}</p>
            </div>
            <div>
              <p class="text-xs text-gray-400">Poll Cycles</p>
              <p class="text-gray-200">{$meshHealth.metrics.poll_cycles}</p>
            </div>
            <div>
              <p class="text-xs text-gray-400">Last Send</p>
              <p class="text-gray-200">{formatMeshTime($meshHealth.metrics.last_send_at)}</p>
            </div>
            <div>
              <p class="text-xs text-gray-400">Last Receive</p>
              <p class="text-gray-200">{formatMeshTime($meshHealth.metrics.last_recv_at)}</p>
            </div>
          </div>
          {#if $meshHealth.metrics.connection_failures > 0}
            <div class="mt-3 p-2 bg-yellow-900/20 border border-yellow-800/50 rounded text-sm text-yellow-300">
              {$meshHealth.metrics.connection_failures} conductor connection failure{$meshHealth.metrics.connection_failures !== 1 ? 's' : ''}
            </div>
          {/if}
        </div>
      {:else if $meshHealth.status === 'offline'}
        <div class="p-4 text-sm text-gray-400">
          <p>Mesh bridge is not running. This is normal if you haven't deployed mesh hardware.</p>
          <p class="mt-2">To enable: set <code class="bg-gray-700 px-1 rounded">MESH_TRANSPORT=lora</code> or <code class="bg-gray-700 px-1 rounded">MESH_TRANSPORT=batman</code> in your deployment config.</p>
        </div>
      {:else}
        <div class="p-4 text-sm text-gray-400">Checking mesh bridge status...</div>
      {/if}
    </div>

    <!-- Footer -->
    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Mycelix Network Monitor v0.1.0 • Holochain 0.6.0</p>
      <p class="mt-1">Distributed Infrastructure for the Civilizational OS</p>
    </footer>
  </main>
</div>

<style>
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  .animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
</style>
