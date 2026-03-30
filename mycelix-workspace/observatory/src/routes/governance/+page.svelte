<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';

  // ============================================================================
  // Types
  // ============================================================================

  interface Proposal {
    id: string;
    title: string;
    description: string;
    proposer: string;
    status: 'active' | 'passed' | 'rejected' | 'executed' | 'pending';
    category: string;
    votesFor: number;
    votesAgainst: number;
    abstentions: number;
    quorum: number;
    quorumReached: boolean;
    createdAt: number;
    endsAt: number;
    epistemicWeight: number;
  }

  interface Council {
    id: string;
    name: string;
    members: number;
    activeProposals: number;
    domain: string;
    trustThreshold: number;
  }

  interface GovernanceStats {
    totalProposals: number;
    activeProposals: number;
    passedProposals: number;
    totalVoters: number;
    participationRate: number;
    avgTurnout: number;
  }

  interface VoteActivity {
    id: string;
    proposalTitle: string;
    voter: string;
    vote: 'for' | 'against' | 'abstain';
    weight: number;
    timestamp: number;
  }

  // ============================================================================
  // Stores
  // ============================================================================

  const stats = writable<GovernanceStats>({
    totalProposals: 156,
    activeProposals: 8,
    passedProposals: 112,
    totalVoters: 2341,
    participationRate: 0.67,
    avgTurnout: 0.54,
  });

  const proposals = writable<Proposal[]>([
    {
      id: 'prop-001',
      title: 'Increase Byzantine Tolerance Threshold',
      description: 'Proposal to research advanced BFT mechanisms targeting 40% threshold (current validated max: 34%)',
      proposer: 'alice.dao',
      status: 'active',
      category: 'Protocol',
      votesFor: 1234,
      votesAgainst: 456,
      abstentions: 89,
      quorum: 1500,
      quorumReached: true,
      createdAt: Date.now() - 86400000 * 5,
      endsAt: Date.now() + 86400000 * 2,
      epistemicWeight: 0.85,
    },
    {
      id: 'prop-002',
      title: 'Community Treasury Allocation Q1 2026',
      description: 'Allocate 50,000 SAP to development initiatives',
      proposer: 'treasury.council',
      status: 'active',
      category: 'Treasury',
      votesFor: 892,
      votesAgainst: 234,
      abstentions: 156,
      quorum: 1200,
      quorumReached: true,
      createdAt: Date.now() - 86400000 * 3,
      endsAt: Date.now() + 86400000 * 4,
      epistemicWeight: 0.78,
    },
    {
      id: 'prop-003',
      title: 'Add New Supply Chain Verification Zome',
      description: 'Integrate advanced provenance tracking for material passports',
      proposer: 'dev.collective',
      status: 'pending',
      category: 'Technical',
      votesFor: 0,
      votesAgainst: 0,
      abstentions: 0,
      quorum: 1000,
      quorumReached: false,
      createdAt: Date.now() - 86400000 * 1,
      endsAt: Date.now() + 86400000 * 6,
      epistemicWeight: 0.92,
    },
    {
      id: 'prop-004',
      title: 'Marketplace Fee Reduction',
      description: 'Reduce transaction fees from 2% to 1.5%',
      proposer: 'merchant.guild',
      status: 'passed',
      category: 'Economic',
      votesFor: 1567,
      votesAgainst: 423,
      abstentions: 210,
      quorum: 1500,
      quorumReached: true,
      createdAt: Date.now() - 86400000 * 14,
      endsAt: Date.now() - 86400000 * 7,
      epistemicWeight: 0.72,
    },
  ]);

  const councils = writable<Council[]>([
    { id: 'council-001', name: 'Technical Council', members: 12, activeProposals: 3, domain: 'Protocol & Code', trustThreshold: 0.85 },
    { id: 'council-002', name: 'Treasury Council', members: 7, activeProposals: 2, domain: 'Finance & Allocation', trustThreshold: 0.90 },
    { id: 'council-003', name: 'Community Council', members: 15, activeProposals: 1, domain: 'Governance & Policy', trustThreshold: 0.75 },
    { id: 'council-004', name: 'Ethics Council', members: 5, activeProposals: 2, domain: 'Disputes & Appeals', trustThreshold: 0.95 },
  ]);

  const recentVotes = writable<VoteActivity[]>([
    { id: 'vote-001', proposalTitle: 'Byzantine Tolerance', voter: 'alice.dao', vote: 'for', weight: 1.2, timestamp: Date.now() - 60000 },
    { id: 'vote-002', proposalTitle: 'Treasury Allocation', voter: 'bob.coop', vote: 'for', weight: 0.8, timestamp: Date.now() - 180000 },
    { id: 'vote-003', proposalTitle: 'Byzantine Tolerance', voter: 'carol.node', vote: 'against', weight: 1.5, timestamp: Date.now() - 300000 },
    { id: 'vote-004', proposalTitle: 'Treasury Allocation', voter: 'dave.maker', vote: 'abstain', weight: 0.9, timestamp: Date.now() - 420000 },
  ]);

  // ============================================================================
  // Helpers
  // ============================================================================

  let currentTime = new Date().toLocaleTimeString();
  let interval: ReturnType<typeof setInterval>;

  onMount(() => {
    interval = setInterval(() => {
      currentTime = new Date().toLocaleTimeString();
      simulateVotes();
    }, 4000);
  });

  onDestroy(() => {
    if (interval) clearInterval(interval);
  });

  function simulateVotes() {
    proposals.update(props => props.map(p => {
      if (p.status === 'active') {
        const newVotes = Math.floor(Math.random() * 5);
        return { ...p, votesFor: p.votesFor + newVotes };
      }
      return p;
    }));
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'active': return 'bg-blue-500';
      case 'passed': case 'executed': return 'bg-green-500';
      case 'rejected': return 'bg-red-500';
      case 'pending': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  }

  function getStatusBadge(status: string): string {
    switch (status) {
      case 'active': return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'passed': case 'executed': return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'rejected': return 'bg-red-500/20 text-red-400 border-red-500/50';
      case 'pending': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  }

  function getVoteColor(vote: string): string {
    switch (vote) {
      case 'for': return 'text-green-400';
      case 'against': return 'text-red-400';
      case 'abstain': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  }

  function formatTime(ts: number): string {
    return new Date(ts).toLocaleTimeString();
  }

  function daysRemaining(ts: number): string {
    const days = Math.ceil((ts - Date.now()) / 86400000);
    if (days < 0) return 'Ended';
    if (days === 0) return 'Ends today';
    return `${days} days left`;
  }

  function getVotePercentage(proposal: Proposal): number {
    const total = proposal.votesFor + proposal.votesAgainst + proposal.abstentions;
    if (total === 0) return 0;
    return (proposal.votesFor / total) * 100;
  }
</script>

<svelte:head>
  <title>Governance | Mycelix Observatory</title>
</svelte:head>

<div class="text-white">
  <!-- Page Header -->
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl">🏛️</span>
        <div>
          <h1 class="text-lg font-bold">Governance Commons</h1>
          <p class="text-xs text-gray-400">Democratic Decision-Making</p>
        </div>
      </div>
      <div class="flex items-center gap-4">
        <div class="text-right">
          <p class="text-xs text-gray-400">Participation Rate</p>
          <p class="text-lg font-bold text-green-400">{($stats.participationRate * 100).toFixed(0)}%</p>
        </div>
        <span class="text-gray-400 font-mono text-sm">{currentTime}</span>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <!-- Stats Grid -->
    <div class="grid grid-cols-2 md:grid-cols-6 gap-4 mb-8">
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Total Proposals</h3>
        <p class="text-2xl font-bold mt-1">{$stats.totalProposals}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Active Votes</h3>
        <p class="text-2xl font-bold mt-1 text-blue-400">{$stats.activeProposals}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Passed</h3>
        <p class="text-2xl font-bold mt-1 text-green-400">{$stats.passedProposals}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Total Voters</h3>
        <p class="text-2xl font-bold mt-1">{$stats.totalVoters.toLocaleString()}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Participation</h3>
        <p class="text-2xl font-bold mt-1 text-purple-400">{($stats.participationRate * 100).toFixed(0)}%</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Avg Turnout</h3>
        <p class="text-2xl font-bold mt-1 text-cyan-400">{($stats.avgTurnout * 100).toFixed(0)}%</p>
      </div>
    </div>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Active Proposals -->
      <div class="bg-gray-800 rounded-lg border border-gray-700 lg:col-span-2">
        <div class="p-4 border-b border-gray-700 flex justify-between items-center">
          <h2 class="text-lg font-semibold flex items-center gap-2">
            <span>📜</span> Proposals
          </h2>
          <span class="text-sm text-gray-400">{$proposals.filter(p => p.status === 'active').length} active</span>
        </div>
        <div class="p-4 space-y-4">
          {#each $proposals as proposal}
            <div class="p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700/70 transition-colors">
              <div class="flex justify-between items-start">
                <div>
                  <div class="flex items-center gap-2">
                    <p class="font-medium">{proposal.title}</p>
                    <span class={`text-xs px-2 py-0.5 rounded border ${getStatusBadge(proposal.status)}`}>
                      {proposal.status}
                    </span>
                    <span class="text-xs px-2 py-0.5 rounded bg-gray-600 text-gray-300">
                      {proposal.category}
                    </span>
                  </div>
                  <p class="text-xs text-gray-400 mt-1">{proposal.description}</p>
                </div>
                <div class="text-right text-sm">
                  <p class="text-gray-400">{daysRemaining(proposal.endsAt)}</p>
                </div>
              </div>

              <!-- Vote Progress -->
              <div class="mt-4">
                <div class="flex justify-between text-xs mb-1">
                  <span class="text-green-400">For: {proposal.votesFor}</span>
                  <span class="text-red-400">Against: {proposal.votesAgainst}</span>
                  <span class="text-yellow-400">Abstain: {proposal.abstentions}</span>
                </div>
                <div class="w-full bg-gray-600 rounded-full h-2 flex overflow-hidden">
                  <div class="bg-green-500 h-2" style="width: {getVotePercentage(proposal)}%"></div>
                  <div class="bg-red-500 h-2" style="width: {(proposal.votesAgainst / (proposal.votesFor + proposal.votesAgainst + proposal.abstentions || 1)) * 100}%"></div>
                </div>
                <div class="flex justify-between text-xs mt-1">
                  <span class="text-gray-400">Quorum: {proposal.quorumReached ? '✓' : `${proposal.votesFor + proposal.votesAgainst + proposal.abstentions}/${proposal.quorum}`}</span>
                  <span class="text-purple-400">Epistemic Weight: {(proposal.epistemicWeight * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Recent Votes & Councils -->
      <div class="space-y-6">
        <!-- Recent Votes -->
        <div class="bg-gray-800 rounded-lg border border-gray-700">
          <div class="p-4 border-b border-gray-700">
            <h2 class="text-lg font-semibold flex items-center gap-2">
              <span>🗳️</span> Recent Votes
            </h2>
          </div>
          <div class="p-4 space-y-2 max-h-64 overflow-y-auto">
            {#each $recentVotes as vote}
              <div class="p-2 bg-gray-700/50 rounded-lg text-sm">
                <div class="flex justify-between items-center">
                  <span class="font-medium">{vote.voter}</span>
                  <span class={`font-bold ${getVoteColor(vote.vote)}`}>{vote.vote.toUpperCase()}</span>
                </div>
                <div class="flex justify-between text-xs text-gray-400 mt-1">
                  <span>{vote.proposalTitle}</span>
                  <span>Weight: {vote.weight}x</span>
                </div>
              </div>
            {/each}
          </div>
        </div>

        <!-- Councils -->
        <div class="bg-gray-800 rounded-lg border border-gray-700">
          <div class="p-4 border-b border-gray-700">
            <h2 class="text-lg font-semibold flex items-center gap-2">
              <span>👥</span> Councils
            </h2>
          </div>
          <div class="p-4 space-y-2">
            {#each $councils as council}
              <div class="p-3 bg-gray-700/50 rounded-lg">
                <div class="flex justify-between items-start">
                  <div>
                    <p class="font-medium text-sm">{council.name}</p>
                    <p class="text-xs text-gray-400">{council.domain}</p>
                  </div>
                  <span class="text-xs text-blue-400">{council.activeProposals} active</span>
                </div>
                <div class="flex justify-between text-xs mt-2">
                  <span class="text-gray-400">{council.members} members</span>
                  <span class="text-green-400">Min trust: {(council.trustThreshold * 100).toFixed(0)}%</span>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>

    <!-- Governance Mechanisms -->
    <div class="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          <span>⚖️</span> Voting Power
        </h3>
        <div class="bg-gray-900 rounded p-3 font-mono text-xs">
          <p class="text-green-400">VP = SAP × Patience × Trust</p>
        </div>
        <div class="mt-3 space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Base</span>
            <span>SAP Balance</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Patience Mult</span>
            <span>1.0 - 3.0x</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Trust Mult</span>
            <span>MATL Score</span>
          </div>
        </div>
      </div>

      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          <span>🎯</span> Quorum Rules
        </h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Protocol Changes</span>
            <span class="text-yellow-400">67%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Treasury</span>
            <span class="text-yellow-400">60%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Community</span>
            <span class="text-green-400">50%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Emergency</span>
            <span class="text-red-400">75%</span>
          </div>
        </div>
      </div>

      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
          <span>🛡️</span> Constitution
        </h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Version</span>
            <span>v1.2.0</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Amendments</span>
            <span>12</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Last Updated</span>
            <span>2026-01-15</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Signatories</span>
            <span class="text-green-400">847</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Footer -->
    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Mycelix Governance v0.1.0 • 7 Zomes • HDK 0.6.0</p>
      <p class="mt-1">Democratic Coordination for the Civilizational OS</p>
    </footer>
  </main>
</div>
