<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { BeliefShare, ValidationVote, ConsensusRecord } from '$services/collective-sensemaking';
  import { ConsensusType, ValidationVoteType } from '$services/collective-sensemaking';

  /**
   * ConsensusHistoryTimeline - Visualizes consensus evolution over time
   *
   * Features:
   * - Timeline of votes and consensus changes
   * - Visual indication of consensus strength
   * - Expandable vote details
   * - Auto-updates with new votes
   */

  // ============================================================================
  // PROPS
  // ============================================================================

  /** The belief to show history for */
  export let belief: BeliefShare | null = null;

  /** History of consensus records */
  export let history: ConsensusRecord[] = [];

  /** History of votes */
  export let votes: ValidationVote[] = [];

  /** Whether to show detailed vote information */
  export let showVoteDetails: boolean = true;

  // ============================================================================
  // STATE
  // ============================================================================

  let expandedItems: Set<string> = new Set();

  // ============================================================================
  // COMPUTED
  // ============================================================================

  interface TimelineEvent {
    id: string;
    type: 'vote' | 'consensus_change' | 'belief_created';
    timestamp: number;
    data: ValidationVote | ConsensusRecord | BeliefShare;
  }

  $: timelineEvents = buildTimeline(belief, history, votes);

  function buildTimeline(
    b: BeliefShare | null,
    h: ConsensusRecord[],
    v: ValidationVote[]
  ): TimelineEvent[] {
    const events: TimelineEvent[] = [];

    // Add belief creation event
    if (b) {
      events.push({
        id: `belief-${b.content_hash}`,
        type: 'belief_created',
        timestamp: b.shared_at,
        data: b,
      });
    }

    // Add vote events
    for (const vote of v) {
      events.push({
        id: `vote-${vote.belief_share_hash?.toString() || Date.now()}-${Math.random()}`,
        type: 'vote',
        timestamp: vote.voted_at,
        data: vote,
      });
    }

    // Add consensus change events
    for (const record of h) {
      events.push({
        id: `consensus-${record.belief_share_hash?.toString() || Date.now()}-${Math.random()}`,
        type: 'consensus_change',
        timestamp: record.reached_at,
        data: record,
      });
    }

    // Sort by timestamp descending (newest first)
    return events.sort((a, b) => b.timestamp - a.timestamp);
  }

  // ============================================================================
  // HELPERS
  // ============================================================================

  function formatTimestamp(ts: number): string {
    const date = new Date(ts / 1000); // Convert microseconds to milliseconds
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;

    return date.toLocaleDateString();
  }

  function getVoteTypeLabel(type: ValidationVoteType): string {
    switch (type) {
      case ValidationVoteType.Corroborate: return 'Corroborate';
      case ValidationVoteType.Plausible: return 'Plausible';
      case ValidationVoteType.Abstain: return 'Abstain';
      case ValidationVoteType.Implausible: return 'Implausible';
      case ValidationVoteType.Contradict: return 'Contradict';
      default: return String(type);
    }
  }

  function getVoteTypeColor(type: ValidationVoteType): string {
    switch (type) {
      case ValidationVoteType.Corroborate: return '#22c55e';
      case ValidationVoteType.Plausible: return '#3b82f6';
      case ValidationVoteType.Abstain: return '#6b7280';
      case ValidationVoteType.Implausible: return '#f59e0b';
      case ValidationVoteType.Contradict: return '#ef4444';
      default: return '#6b7280';
    }
  }

  function getConsensusTypeLabel(type: ConsensusType): string {
    switch (type) {
      case ConsensusType.StrongConsensus: return 'Strong Consensus';
      case ConsensusType.ModerateConsensus: return 'Moderate Consensus';
      case ConsensusType.WeakConsensus: return 'Weak Consensus';
      case ConsensusType.Contested: return 'Contested';
      case ConsensusType.Insufficient: return 'Insufficient Votes';
      default: return String(type);
    }
  }

  function getConsensusTypeColor(type: ConsensusType): string {
    switch (type) {
      case ConsensusType.StrongConsensus: return '#22c55e';
      case ConsensusType.ModerateConsensus: return '#84cc16';
      case ConsensusType.WeakConsensus: return '#eab308';
      case ConsensusType.Contested: return '#ef4444';
      case ConsensusType.Insufficient: return '#6b7280';
      default: return '#6b7280';
    }
  }

  function toggleExpand(id: string) {
    if (expandedItems.has(id)) {
      expandedItems.delete(id);
    } else {
      expandedItems.add(id);
    }
    expandedItems = new Set(expandedItems);
  }
</script>

<div class="timeline-container">
  <h3 class="timeline-title">Consensus History</h3>

  {#if timelineEvents.length === 0}
    <div class="empty-state">
      <span class="empty-icon">&#8987;</span>
      <p>No history yet</p>
      <p class="empty-hint">Votes and consensus changes will appear here</p>
    </div>
  {:else}
    <div class="timeline">
      {#each timelineEvents as event (event.id)}
        <div
          class="timeline-item"
          class:expanded={expandedItems.has(event.id)}
          class:vote={event.type === 'vote'}
          class:consensus={event.type === 'consensus_change'}
          class:belief={event.type === 'belief_created'}
        >
          <div class="timeline-marker">
            <div class="marker-dot" style:--marker-color={
              event.type === 'vote'
                ? getVoteTypeColor((event.data as ValidationVote).vote_type)
                : event.type === 'consensus_change'
                  ? getConsensusTypeColor((event.data as ConsensusRecord).consensus_type)
                  : '#7c3aed'
            }></div>
            <div class="marker-line"></div>
          </div>

          <div
            class="timeline-content"
            role="button"
            tabindex="0"
            on:click={() => toggleExpand(event.id)}
            on:keydown={(e) => e.key === 'Enter' && toggleExpand(event.id)}
          >
            <div class="event-header">
              <span class="event-type">
                {#if event.type === 'vote'}
                  <span class="vote-badge" style:background={getVoteTypeColor((event.data as ValidationVote).vote_type)}>
                    {getVoteTypeLabel((event.data as ValidationVote).vote_type)}
                  </span>
                {:else if event.type === 'consensus_change'}
                  <span class="consensus-badge" style:background={getConsensusTypeColor((event.data as ConsensusRecord).consensus_type)}>
                    {getConsensusTypeLabel((event.data as ConsensusRecord).consensus_type)}
                  </span>
                {:else}
                  <span class="belief-badge">Belief Shared</span>
                {/if}
              </span>
              <span class="event-time">{formatTimestamp(event.timestamp)}</span>
            </div>

            {#if showVoteDetails && expandedItems.has(event.id)}
              <div class="event-details">
                {#if event.type === 'vote'}
                  {@const vote = event.data as ValidationVote}
                  <div class="detail-row">
                    <span class="detail-label">Weight</span>
                    <span class="detail-value">{vote.voter_weight.toFixed(2)}</span>
                  </div>
                  {#if vote.evidence}
                    <div class="detail-row full">
                      <span class="detail-label">Evidence</span>
                      <p class="detail-value evidence">{vote.evidence}</p>
                    </div>
                  {/if}
                {:else if event.type === 'consensus_change'}
                  {@const consensus = event.data as ConsensusRecord}
                  <div class="detail-row">
                    <span class="detail-label">Agreement</span>
                    <span class="detail-value">{Math.round(consensus.agreement_score * 100)}%</span>
                  </div>
                  <div class="detail-row">
                    <span class="detail-label">Validators</span>
                    <span class="detail-value">{consensus.validator_count}</span>
                  </div>
                {:else if event.type === 'belief_created'}
                  {@const b = event.data as BeliefShare}
                  <div class="detail-row full">
                    <span class="detail-label">Content</span>
                    <p class="detail-value content">{b.content}</p>
                  </div>
                  <div class="detail-row">
                    <span class="detail-label">Type</span>
                    <span class="detail-value">{b.belief_type}</span>
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .timeline-container {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .timeline-title {
    margin: 0 0 16px;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary, #f8fafc);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: var(--text-muted, #94a3b8);
    text-align: center;
  }

  .empty-icon {
    font-size: 32px;
    opacity: 0.3;
    margin-bottom: 8px;
  }

  .empty-state p {
    margin: 4px 0;
  }

  .empty-hint {
    font-size: 12px;
    opacity: 0.7;
  }

  .timeline {
    display: flex;
    flex-direction: column;
    gap: 0;
    overflow-y: auto;
    flex: 1;
  }

  .timeline-item {
    display: flex;
    gap: 12px;
    padding: 0;
    position: relative;
  }

  .timeline-item:last-child .marker-line {
    display: none;
  }

  .timeline-marker {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 20px;
    flex-shrink: 0;
  }

  .marker-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--marker-color, #6b7280);
    border: 2px solid var(--bg-primary, #0f172a);
    z-index: 1;
  }

  .marker-line {
    width: 2px;
    flex: 1;
    background: var(--border, #334155);
    margin-top: 4px;
  }

  .timeline-content {
    flex: 1;
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid var(--border, #334155);
  }

  .timeline-content:hover {
    border-color: var(--border-hover, #475569);
  }

  .timeline-item.expanded .timeline-content {
    border-color: var(--color-primary, #3b82f6);
  }

  .event-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .event-type {
    display: flex;
    align-items: center;
  }

  .vote-badge,
  .consensus-badge,
  .belief-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    color: white;
  }

  .belief-badge {
    background: #7c3aed;
  }

  .event-time {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
  }

  .event-details {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border, #334155);
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }

  .detail-row {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .detail-row.full {
    width: 100%;
  }

  .detail-label {
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-muted, #94a3b8);
    letter-spacing: 0.5px;
  }

  .detail-value {
    font-size: 13px;
    color: var(--text-primary, #f8fafc);
  }

  .detail-value.evidence,
  .detail-value.content {
    background: var(--bg-secondary, #1e293b);
    padding: 8px;
    border-radius: 4px;
    font-size: 12px;
    line-height: 1.5;
    margin: 4px 0 0;
  }
</style>
