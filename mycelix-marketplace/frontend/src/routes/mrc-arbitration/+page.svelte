<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  /**
   * MRC Arbitration Interface
   *
   * Multi-Resonance Council interface for dispute arbitration:
   * - View pending disputes
   * - Review evidence with IPFS integration
   * - Cast votes on remedies
   * - View voting consensus progress
   * - Track dispute resolution status
   *
   * This demonstrates:
   * - MRC constitutional governance
   * - Weighted voting by PoGQ trust scores
   * - Evidence review with PhotoGallery
   * - Dispute lifecycle management
   * - Constitutional quorum enforcement
   */

  import { onMount } from 'svelte';
  import PhotoGallery from '$lib/components/PhotoGallery.svelte';
  import { initHolochainClient } from '$lib/holochain';
  import {
    isArbitrator as isArbitratorZome,
    getArbitratorProfile,
    getMyArbitrationCases,
    castArbitratorVote,
  } from '$lib/holochain/disputes';
  import { notifications } from '$lib/stores';
  import type { Dispute, ArbitratorProfile, CastVoteInput } from '$types';

  // Arbitrator state
  let isArbitrator = false;
  let arbitratorProfile: ArbitratorProfile | null = null;
  let loading = true;
  let error = '';

  // Disputes state
  let pendingDisputes: Dispute[] = [];
  let activeDisputes: Dispute[] = [];
  let resolvedDisputes: Dispute[] = [];
  let selectedDispute: Dispute | null = null;

  // Tab state
  let activeTab: 'pending' | 'active' | 'resolved' = 'pending';

  // Voting state
  let votingInProgress = false;
  let voteError = '';
  let voteSuccess = false;

  /**
   * Load arbitrator profile and disputes
   */
  onMount(async () => {
    loading = true;
    error = '';

    try {
      const client = await initHolochainClient();

      // Check arbitrator status once
      const arbitratorStatus = await isArbitratorZome(client);

      if (!arbitratorStatus) {
        isArbitrator = false;
        error = 'You are not registered as an MRC arbitrator';
        notifications.warning('Access Denied', 'You are not an MRC arbitrator');
        return;
      }

      isArbitrator = true;

      // Fetch profile and disputes in parallel without re-checking arbitrator status
      const [profileResult, disputesResult] = await Promise.allSettled([
        getArbitratorProfile(client),
        getMyArbitrationCases(client),
      ]);

      if (profileResult.status === 'fulfilled') {
        arbitratorProfile = profileResult.value;
      } else {
        console.warn('Failed to load arbitrator profile:', profileResult.reason);
        arbitratorProfile = null;
      }

      const disputes =
        disputesResult.status === 'fulfilled' ? disputesResult.value : [];

      if (disputesResult.status === 'rejected') {
        console.warn('Failed to load arbitration cases:', disputesResult.reason);
      }

      // Categorize disputes by status
      pendingDisputes = disputes.filter((d) => d.status === 'pending');
      activeDisputes = disputes.filter((d) => d.status === 'active');
      resolvedDisputes = disputes.filter((d) => d.status === 'resolved' || d.status === 'rejected');

      notifications.success(
        'MRC Interface Loaded',
        `${pendingDisputes.length} pending, ${activeDisputes.length} active cases`
      );
    } catch (e: any) {
      error = e.message || 'Failed to load arbitration interface';
      notifications.error('Loading Failed', error);
    } finally {
      loading = false;
    }
  });

  /**
   * Select dispute for detailed view
   */
  function selectDispute(dispute: any) {
    selectedDispute = dispute;
    voteError = '';
    voteSuccess = false;
  }

  /**
   * Cast vote on dispute
   */
  async function castVote(approve: boolean) {
    if (!selectedDispute) return;

    votingInProgress = true;
    voteError = '';
    voteSuccess = false;

    try {
      const client = await initHolochainClient();

      const voteInput: CastVoteInput = {
        claim_id: selectedDispute.claim_id,
        vote: approve ? 'Approve' : 'Reject',
        reasoning: '', // Can be extended to include reasoning field in UI
      };

      // Cast vote
      const updatedDispute = await castArbitratorVote(client, voteInput);

      // Update local state with returned dispute
      selectedDispute = updatedDispute;

      // Check if dispute was resolved
      if (updatedDispute.consensus_reached) {
        // Move to resolved tab
        resolvedDisputes = [updatedDispute, ...resolvedDisputes];
        activeDisputes = activeDisputes.filter(
          (d) => d.claim_id !== updatedDispute.claim_id
        );
        pendingDisputes = pendingDisputes.filter(
          (d) => d.claim_id !== updatedDispute.claim_id
        );

        activeTab = 'resolved';

        notifications.success(
          'Dispute Resolved',
          `Final decision: ${updatedDispute.final_decision?.toUpperCase()}`
        );
      } else {
        // Update in active list
        activeDisputes = activeDisputes.map((d) =>
          d.claim_id === updatedDispute.claim_id ? updatedDispute : d
        );
        pendingDisputes = pendingDisputes.map((d) =>
          d.claim_id === updatedDispute.claim_id ? updatedDispute : d
        );

        notifications.success(
          'Vote Cast',
          `${updatedDispute.votes_cast}/${updatedDispute.votes_required} votes cast`
        );
      }

      voteSuccess = true;

      // Clear success message after 3 seconds
      setTimeout(() => {
        voteSuccess = false;
      }, 3000);
    } catch (e: any) {
      voteError = e.message || 'Failed to cast vote';
      notifications.error('Vote Failed', voteError);
    } finally {
      votingInProgress = false;
    }
  }

  /**
   * Format date to relative time
   */
  function formatDate(timestamp: number): string {
    const now = Date.now();
    const diff = now - timestamp;
    const days = Math.floor(diff / (24 * 60 * 60 * 1000));
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
    if (days < 365) return `${Math.floor(days / 30)} months ago`;
    return `${Math.floor(days / 365)} years ago`;
  }

  /**
   * Get dispute type display name
   */
  function getDisputeTypeName(reason: string): string {
    const reasonMap: Record<string, string> = {
      defective_product: 'Defective Product',
      wrong_item: 'Wrong Item Received',
      non_delivery: 'Item Not Received',
      not_as_described: 'Product Not As Described',
      damaged_in_shipping: 'Damaged During Shipping',
      seller_unresponsive: 'Seller Unresponsive',
      other: 'Other Issue',
    };
    return reasonMap[reason] || reason.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
  }

  /**
   * Get status badge class
   */
  function getStatusClass(status: string): string {
    const statusMap: Record<string, string> = {
      Filed: 'status-warning',
      UnderReview: 'status-info',
      Resolved: 'status-success',
      Rejected: 'status-error',
    };
    return statusMap[status] || 'status-default';
  }
</script>

<div class="mrc-arbitration">
  <div class="container">
    {#if loading}
      <!-- Loading State -->
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading arbitration interface...</p>
      </div>
    {:else if error || !isArbitrator}
      <!-- Error / Not Authorized -->
      <div class="error-state">
        <span class="error-icon">⚠️</span>
        <h2>Access Denied</h2>
        <p>{error || 'You are not authorized as an MRC arbitrator'}</p>
        <a href="/dashboard" class="btn btn-secondary">Return to Dashboard</a>
      </div>
    {:else}
      <!-- Arbitrator Dashboard -->
      <div class="arbitrator-header">
        <div class="header-content">
          <h1>MRC Arbitration Interface</h1>
          <p class="subtitle">Multi-Resonance Council Dispute Resolution</p>
        </div>

        {#if arbitratorProfile}
          <div class="arbitrator-stats">
            <div class="stat-card">
              <div class="stat-value">{(arbitratorProfile.pogq_score * 100).toFixed(1)}%</div>
              <div class="stat-label">PoGQ Score</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">{arbitratorProfile.cases_arbitrated}</div>
              <div class="stat-label">Cases Arbitrated</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">{arbitratorProfile.approval_rate.toFixed(0)}%</div>
              <div class="stat-label">Approval Rate</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">
                {arbitratorProfile.pending_cases + arbitratorProfile.active_cases}
              </div>
              <div class="stat-label">Active Cases</div>
            </div>
          </div>
        {/if}
      </div>

      <!-- Navigation Tabs -->
      <div class="dispute-tabs">
        <button
          class="tab"
          class:active={activeTab === 'pending'}
          on:click={() => {
            activeTab = 'pending';
            selectedDispute = null;
          }}
        >
          Pending ({pendingDisputes.length})
        </button>
        <button
          class="tab"
          class:active={activeTab === 'active'}
          on:click={() => {
            activeTab = 'active';
            selectedDispute = null;
          }}
        >
          Under Review ({activeDisputes.length})
        </button>
        <button
          class="tab"
          class:active={activeTab === 'resolved'}
          on:click={() => {
            activeTab = 'resolved';
            selectedDispute = null;
          }}
        >
          Resolved ({resolvedDisputes.length})
        </button>
      </div>

      <!-- Disputes Content -->
      <div class="disputes-content">
        {#if activeTab === 'pending'}
          <!-- Pending Disputes -->
          {#if pendingDisputes.length === 0}
            <div class="empty-state">
              <span>⚖️</span>
              <p>No pending disputes</p>
            </div>
          {:else}
            <div class="disputes-grid">
              {#each pendingDisputes as dispute}
                <button
                  class="dispute-card"
                  class:selected={selectedDispute?.claim_id === dispute.claim_id}
                  on:click={() => selectDispute(dispute)}
                  on:keydown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      selectDispute(dispute);
                    }
                  }}
                  aria-label="View dispute for {dispute.title}"
                >
                  <div class="dispute-header">
                    <span class={`status-badge ${getStatusClass(dispute.status)}`}>
                      {dispute.status}
                    </span>
                    <span class="dispute-date">{formatDate(dispute.created_at)}</span>
                  </div>
                  <h3>{getDisputeTypeName(dispute.reason)}</h3>
                  <p class="dispute-listing">Listing: {dispute.title}</p>
                  <p class="dispute-parties">
                    {dispute.buyer_name} vs {dispute.seller_name}
                  </p>
                  <div class="dispute-remedy">
                    Refund: <strong>${dispute.refund_amount?.toFixed(2) || 'TBD'}</strong>
                  </div>
                </button>
              {/each}
            </div>
          {/if}
        {:else if activeTab === 'active'}
          <!-- Active Disputes -->
          {#if activeDisputes.length === 0}
            <div class="empty-state">
              <span>⚖️</span>
              <p>No disputes under review</p>
            </div>
          {:else}
            <div class="disputes-grid">
              {#each activeDisputes as dispute}
                <button
                  class="dispute-card"
                  class:selected={selectedDispute?.claim_id === dispute.claim_id}
                  on:click={() => selectDispute(dispute)}
                  on:keydown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      selectDispute(dispute);
                    }
                  }}
                  aria-label="View active dispute for {dispute.title}"
                >
                  <div class="dispute-header">
                    <span class={`status-badge ${getStatusClass(dispute.status)}`}>
                      {dispute.status}
                    </span>
                    <span class="dispute-date">{formatDate(dispute.created_at)}</span>
                  </div>
                  <h3>{getDisputeTypeName(dispute.reason)}</h3>
                  <p class="dispute-listing">Listing: {dispute.title}</p>

                  <div class="voting-progress">
                    <div class="progress-bar">
                      <div
                        class="progress-fill"
                        style="width: {(dispute.votes_cast / dispute.votes_required) * 100}%"
                      ></div>
                    </div>
                    <p class="progress-text">
                      {dispute.votes_cast} / {dispute.votes_required} votes
                      · {(
                        (dispute.approval_voting_power / dispute.total_voting_power) *
                        100
                      ).toFixed(0)}% approval
                    </p>
                  </div>

                  {#if dispute.my_vote}
                    <div class="my-vote">
                      Your vote: <strong>{dispute.my_vote}</strong>
                    </div>
                  {/if}
                </button>
              {/each}
            </div>
          {/if}
        {:else if activeTab === 'resolved'}
          <!-- Resolved Disputes -->
          {#if resolvedDisputes.length === 0}
            <div class="empty-state">
              <span>✅</span>
              <p>No resolved disputes</p>
            </div>
          {:else}
            <div class="disputes-grid">
              {#each resolvedDisputes as dispute}
                <button
                  class="dispute-card"
                  on:click={() => selectDispute(dispute)}
                  on:keydown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      selectDispute(dispute);
                    }
                  }}
                  aria-label="View resolved dispute for {dispute.title}"
                >
                  <div class="dispute-header">
                    <span class={`status-badge ${getStatusClass(dispute.status)}`}>
                      {dispute.final_decision || dispute.status}
                    </span>
                    <span class="dispute-date">{formatDate(dispute.resolved_at || dispute.created_at)}</span>
                  </div>
                  <h3>{getDisputeTypeName(dispute.reason)}</h3>
                  <p class="dispute-listing">Listing: {dispute.title}</p>
                  <p class="resolution-consensus">
                    Final: {(
                      (dispute.approval_voting_power / dispute.total_voting_power) *
                      100
                    ).toFixed(0)}% approval ({dispute.votes_cast} votes)
                  </p>
                </button>
              {/each}
            </div>
          {/if}
        {/if}
      </div>

      <!-- Dispute Detail Modal -->
      {#if selectedDispute}
        <div
          class="dispute-modal"
          on:click={(event) => {
            if (event.currentTarget === event.target) {
              selectedDispute = null;
            }
          }}
          on:keydown={(e) => {
            if (e.key === 'Escape') {
              selectedDispute = null;
            }
          }}
          tabindex="-1"
          role="presentation"
        >
          <div
            class="modal-content"
            role="dialog"
            aria-modal="true"
            aria-labelledby="dispute-modal-title"
          >
            <button class="modal-close" on:click={() => (selectedDispute = null)}>✕</button>

            <h2>Dispute Details</h2>

            <div class="modal-section">
              <h3>Type</h3>
              <p class="dispute-type-badge">
                {getDisputeTypeName(selectedDispute.reason)}
              </p>
            </div>

            <div class="modal-section">
              <h3>Transaction</h3>
              <p><strong>Listing:</strong> {selectedDispute.title}</p>
              <p><strong>Buyer:</strong> {selectedDispute.buyer_name}</p>
              <p><strong>Seller:</strong> {selectedDispute.seller_name} ({(selectedDispute.seller_trust_score * 100).toFixed(1)}% trust)</p>
              <p><strong>Filed:</strong> {formatDate(selectedDispute.created_at)}</p>
            </div>

            <div class="modal-section">
              <h3>Description</h3>
              <p class="dispute-description">{selectedDispute.description}</p>
            </div>

            <div class="modal-section">
              <h3>Refund Amount</h3>
              <p class="remedy-badge">
                ${selectedDispute.refund_amount?.toFixed(2) || 'To Be Determined'}
              </p>
            </div>

            {#if selectedDispute.evidence_ipfs_cids && selectedDispute.evidence_ipfs_cids.length > 0}
              <div class="modal-section">
                <h3>Evidence</h3>
                <PhotoGallery
                  cids={selectedDispute.evidence_ipfs_cids}
                  alt="Dispute evidence"
                  layout="grid"
                  maxHeight="300px"
                />
              </div>
            {/if}

            <!-- Voting Section (for pending/active disputes) -->
            {#if selectedDispute.status !== 'resolved' && !selectedDispute.my_vote && arbitratorProfile}
              <div class="modal-section voting-section">
                <h3>Cast Your Vote</h3>
                <p class="voting-info">
                  As an MRC arbitrator, your vote is weighted by your PoGQ trust score
                  ({(arbitratorProfile.pogq_score * 100).toFixed(1)}%).
                </p>

                {#if voteError}
                  <div class="alert alert-error">
                    <span class="alert-icon">⚠️</span>
                    {voteError}
                  </div>
                {/if}

                {#if voteSuccess}
                  <div class="alert alert-success">
                    <span class="alert-icon">✓</span>
                    Vote cast successfully!
                  </div>
                {/if}

                <div class="voting-buttons">
                  <button
                    class="btn btn-success"
                    on:click={() => castVote(true)}
                    disabled={votingInProgress}
                  >
                    {#if votingInProgress}
                      Voting...
                    {:else}
                      ✓ Approve Remedy
                    {/if}
                  </button>
                  <button
                    class="btn btn-danger"
                    on:click={() => castVote(false)}
                    disabled={votingInProgress}
                  >
                    {#if votingInProgress}
                      Voting...
                    {:else}
                      ✕ Reject Remedy
                    {/if}
                  </button>
                </div>
              </div>
            {:else if selectedDispute.my_vote}
              <div class="modal-section">
                <div class="alert alert-info">
                  <span class="alert-icon">ℹ️</span>
                  You have already voted: <strong>{selectedDispute.my_vote}</strong>
                </div>
              </div>
            {:else if selectedDispute.status === 'resolved' || selectedDispute.status === 'rejected'}
              <div class="modal-section">
                <div class="alert alert-success">
                  <span class="alert-icon">✓</span>
                  Resolved: <strong>{selectedDispute.final_decision?.toUpperCase()}</strong>
                  <br />
                  Final Approval: {(
                    (selectedDispute.approval_voting_power / selectedDispute.total_voting_power) *
                    100
                  ).toFixed(0)}% ({selectedDispute.votes_cast} votes)
                </div>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>

<style>
  .mrc-arbitration {
    min-height: 100vh;
    padding: 2rem 1rem;
    background: #f7fafc;
  }

  .container {
    max-width: 1400px;
    margin: 0 auto;
  }

  /* Loading State */
  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    gap: 1rem;
  }

  .spinner {
    width: 50px;
    height: 50px;
    border: 4px solid #e2e8f0;
    border-top-color: #4299e1;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* Error State */
  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    background: white;
    border-radius: 0.5rem;
    padding: 3rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .error-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
  }

  .error-state h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.5rem;
  }

  .error-state p {
    color: #718096;
    margin-bottom: 2rem;
  }

  /* Arbitrator Header */
  .arbitrator-header {
    background: white;
    border-radius: 0.5rem;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .header-content h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 0.5rem;
  }

  .subtitle {
    color: #718096;
    font-size: 1.125rem;
    margin-bottom: 2rem;
  }

  .arbitrator-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
  }

  .stat-card {
    text-align: center;
    padding: 1rem;
    background: #f7fafc;
    border-radius: 0.375rem;
  }

  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #805ad5;
    margin-bottom: 0.25rem;
  }

  .stat-label {
    font-size: 0.875rem;
    color: #718096;
  }

  /* Dispute Tabs */
  .dispute-tabs {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 2rem;
    border-bottom: 2px solid #e2e8f0;
    overflow-x: auto;
  }

  .tab {
    padding: 1rem 1.5rem;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: #718096;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
  }

  .tab:hover {
    color: #805ad5;
  }

  .tab.active {
    color: #805ad5;
    border-bottom-color: #805ad5;
  }

  /* Disputes Content */
  .disputes-content {
    background: white;
    border-radius: 0.5rem;
    padding: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    min-height: 400px;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    text-align: center;
  }

  .empty-state span {
    font-size: 4rem;
    margin-bottom: 1rem;
  }

  .empty-state p {
    color: #718096;
    font-size: 1.125rem;
  }

  /* Disputes Grid */
  .disputes-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1.5rem;
  }

  .dispute-card {
    padding: 1.5rem;
    border: 2px solid #e2e8f0;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .dispute-card:hover {
    border-color: #805ad5;
    box-shadow: 0 4px 12px rgba(128, 90, 213, 0.1);
  }

  .dispute-card.selected {
    border-color: #805ad5;
    background: #faf5ff;
  }

  .dispute-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .dispute-date {
    font-size: 0.875rem;
    color: #a0aec0;
  }

  .dispute-card h3 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.5rem;
  }

  .dispute-listing {
    font-size: 0.875rem;
    color: #4a5568;
    margin-bottom: 0.25rem;
  }

  .dispute-parties {
    font-size: 0.875rem;
    color: #718096;
    margin-bottom: 1rem;
  }

  .dispute-remedy {
    font-size: 0.875rem;
    color: #4a5568;
  }

  /* Voting Progress */
  .voting-progress {
    margin-top: 1rem;
  }

  .progress-bar {
    height: 8px;
    background: #e2e8f0;
    border-radius: 0.25rem;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }

  .progress-fill {
    height: 100%;
    background: #805ad5;
    transition: width 0.3s;
  }

  .progress-text {
    font-size: 0.75rem;
    color: #718096;
  }

  .my-vote {
    margin-top: 0.75rem;
    padding: 0.5rem;
    background: #e6fffa;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    color: #234e52;
  }

  .resolution-consensus {
    font-size: 0.875rem;
    color: #38a169;
    font-weight: 600;
  }

  /* Status Badges */
  .status-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .status-warning {
    background: #feebc8;
    color: #7c2d12;
  }

  .status-info {
    background: #bee3f8;
    color: #2c5282;
  }

  .status-success {
    background: #c6f6d5;
    color: #22543d;
  }

  .status-error {
    background: #fed7d7;
    color: #742a2a;
  }

  /* Dispute Modal */
  .dispute-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 1rem;
  }

  .modal-content {
    background: white;
    border-radius: 0.5rem;
    padding: 2rem;
    max-width: 800px;
    width: 100%;
    max-height: 90vh;
    overflow-y: auto;
    position: relative;
  }

  .modal-close {
    position: absolute;
    top: 1rem;
    right: 1rem;
    width: 40px;
    height: 40px;
    border: none;
    background: #e2e8f0;
    border-radius: 50%;
    font-size: 1.5rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .modal-close:hover {
    background: #cbd5e0;
  }

  .modal-content h2 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 1.5rem;
  }

  .modal-section {
    margin-bottom: 2rem;
  }

  .modal-section h3 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.75rem;
  }

  .modal-section p {
    color: #4a5568;
    line-height: 1.6;
  }

  .dispute-type-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: #faf5ff;
    color: #805ad5;
    border-radius: 0.375rem;
    font-weight: 600;
  }

  .dispute-description {
    padding: 1rem;
    background: #f7fafc;
    border-radius: 0.375rem;
  }

  .remedy-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: #fff5f5;
    color: #c53030;
    border-radius: 0.375rem;
    font-weight: 600;
  }

  /* Voting Section */
  .voting-section {
    padding: 1.5rem;
    background: #faf5ff;
    border-radius: 0.375rem;
  }

  .voting-info {
    margin-bottom: 1.5rem;
    color: #553c9a;
  }

  .voting-buttons {
    display: flex;
    gap: 1rem;
  }

  /* Buttons */
  .btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 0.375rem;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    display: inline-block;
  }

  .btn-secondary {
    background: #e2e8f0;
    color: #2d3748;
  }

  .btn-secondary:hover {
    background: #cbd5e0;
  }

  .btn-success {
    flex: 1;
    background: #38a169;
    color: white;
  }

  .btn-success:hover:not(:disabled) {
    background: #2f855a;
  }

  .btn-danger {
    flex: 1;
    background: #e53e3e;
    color: white;
  }

  .btn-danger:hover:not(:disabled) {
    background: #c53030;
  }

  .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  /* Alerts */
  .alert {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
    border-radius: 0.375rem;
    margin-bottom: 1rem;
  }

  .alert-icon {
    font-size: 1.25rem;
  }

  .alert-error {
    background: #fed7d7;
    border: 1px solid #fc8181;
    color: #742a2a;
  }

  .alert-success {
    background: #c6f6d5;
    border: 1px solid #68d391;
    color: #22543d;
  }

  .alert-info {
    background: #bee3f8;
    border: 1px solid #63b3ed;
    color: #2c5282;
  }
</style>
