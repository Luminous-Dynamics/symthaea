<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Thought, EpistemicClassification } from '@mycelix/lucid-client';
  import { ThoughtType, EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel } from '@mycelix/lucid-client';
  import type { BeliefShare, ValidationVote } from '$services/collective-sensemaking';
  import { ValidationVoteType } from '$services/collective-sensemaking';
  import EpistemicBadge from './EpistemicBadge.svelte';

  // Parse epistemic code string (e.g., "E2N1M2H1") to EpistemicClassification
  function parseEpistemicCode(code: string): EpistemicClassification {
    const match = code.match(/E(\d)N(\d)M(\d)H(\d)/);
    if (!match) {
      return {
        empirical: EmpiricalLevel.E0,
        normative: NormativeLevel.N0,
        materiality: MaterialityLevel.M0,
        harmonic: HarmonicLevel.H0,
      };
    }
    const empLevels = [EmpiricalLevel.E0, EmpiricalLevel.E1, EmpiricalLevel.E2, EmpiricalLevel.E3, EmpiricalLevel.E4];
    const normLevels = [NormativeLevel.N0, NormativeLevel.N1, NormativeLevel.N2, NormativeLevel.N3];
    const matLevels = [MaterialityLevel.M0, MaterialityLevel.M1, MaterialityLevel.M2, MaterialityLevel.M3];
    const harmLevels = [HarmonicLevel.H0, HarmonicLevel.H1, HarmonicLevel.H2, HarmonicLevel.H3, HarmonicLevel.H4];
    return {
      empirical: empLevels[parseInt(match[1])] ?? EmpiricalLevel.E0,
      normative: normLevels[parseInt(match[2])] ?? NormativeLevel.N0,
      materiality: matLevels[parseInt(match[3])] ?? MaterialityLevel.M0,
      harmonic: harmLevels[parseInt(match[4])] ?? HarmonicLevel.H0,
    };
  }

  /**
   * BeliefThread - Threaded discussion component for beliefs
   *
   * Features:
   * - Nested reply structure
   * - Reply composer
   * - Thread collapse/expand
   * - Reply voting
   * - Reference linking
   */

  // ============================================================================
  // TYPES
  // ============================================================================

  interface BeliefReply {
    id: string;
    parentId: string;
    author: string;
    authorCommitment?: string;
    content: string;
    replyType: 'support' | 'counter' | 'question' | 'clarification';
    createdAt: number;
    votes: { up: number; down: number };
    replies: BeliefReply[];
  }

  // ============================================================================
  // PROPS
  // ============================================================================

  /** The root belief being discussed */
  export let belief: BeliefShare;

  /** Replies to the belief */
  export let replies: BeliefReply[] = [];

  /** Whether user can reply */
  export let canReply: boolean = true;

  /** Maximum nesting depth */
  export let maxDepth: number = 4;

  /** Current nesting depth */
  export let depth: number = 0;

  // ============================================================================
  // STATE
  // ============================================================================

  const dispatch = createEventDispatcher<{
    reply: { parentId: string; content: string; replyType: BeliefReply['replyType'] };
    vote: { replyId: string; direction: 'up' | 'down' };
  }>();

  let isComposing = false;
  let replyContent = '';
  let replyType: BeliefReply['replyType'] = 'support';
  let expandedReplies = new Set<string>();
  let collapsedThreads = new Set<string>();

  // ============================================================================
  // HELPERS
  // ============================================================================

  function formatTimestamp(ts: number): string {
    const date = new Date(ts / 1000);
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
  }

  function getReplyTypeColor(type: BeliefReply['replyType']): string {
    switch (type) {
      case 'support': return '#22c55e';
      case 'counter': return '#ef4444';
      case 'question': return '#3b82f6';
      case 'clarification': return '#f59e0b';
      default: return '#6b7280';
    }
  }

  function getReplyTypeLabel(type: BeliefReply['replyType']): string {
    switch (type) {
      case 'support': return 'Supports';
      case 'counter': return 'Counters';
      case 'question': return 'Questions';
      case 'clarification': return 'Clarifies';
      default: return '';
    }
  }

  function toggleComposer() {
    isComposing = !isComposing;
    if (!isComposing) {
      replyContent = '';
      replyType = 'support';
    }
  }

  function submitReply() {
    if (!replyContent.trim()) return;

    dispatch('reply', {
      parentId: belief.content_hash,
      content: replyContent.trim(),
      replyType,
    });

    replyContent = '';
    isComposing = false;
  }

  function toggleThread(replyId: string) {
    if (collapsedThreads.has(replyId)) {
      collapsedThreads.delete(replyId);
    } else {
      collapsedThreads.add(replyId);
    }
    collapsedThreads = new Set(collapsedThreads);
  }

  function voteReply(replyId: string, direction: 'up' | 'down') {
    dispatch('vote', { replyId, direction });
  }

  $: totalReplies = countReplies(replies);

  function countReplies(replies: BeliefReply[]): number {
    return replies.reduce((sum, r) => sum + 1 + countReplies(r.replies), 0);
  }
</script>

<div class="belief-thread" style:--depth={depth}>
  <!-- Root belief (only show at depth 0) -->
  {#if depth === 0}
    <div class="belief-root">
      <div class="belief-header">
        <span class="belief-type">{belief.belief_type}</span>
        {#if belief.epistemic_code}
          <EpistemicBadge epistemic={parseEpistemicCode(belief.epistemic_code)} compact />
        {/if}
        <span class="belief-time">{formatTimestamp(belief.shared_at)}</span>
      </div>
      <p class="belief-content">{belief.content}</p>
      {#if belief.tags.length > 0}
        <div class="belief-tags">
          {#each belief.tags as tag}
            <span class="tag">{tag}</span>
          {/each}
        </div>
      {/if}

      <!-- Reply count and composer toggle -->
      <div class="belief-actions">
        <span class="reply-count">
          {totalReplies} {totalReplies === 1 ? 'reply' : 'replies'}
        </span>
        {#if canReply}
          <button class="reply-btn" on:click={toggleComposer}>
            {isComposing ? 'Cancel' : 'Reply'}
          </button>
        {/if}
      </div>

      <!-- Reply composer -->
      {#if isComposing}
        <div class="reply-composer">
          <div class="composer-header">
            <span>Your reply:</span>
            <select bind:value={replyType} class="reply-type-select">
              <option value="support">Supports</option>
              <option value="counter">Counters</option>
              <option value="question">Questions</option>
              <option value="clarification">Clarifies</option>
            </select>
          </div>
          <textarea
            bind:value={replyContent}
            placeholder="Share your perspective..."
            rows="3"
          ></textarea>
          <div class="composer-actions">
            <button class="submit-btn" on:click={submitReply} disabled={!replyContent.trim()}>
              Submit Reply
            </button>
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Replies -->
  {#if replies.length > 0}
    <div class="replies-section">
      {#each replies as reply (reply.id)}
        {@const isCollapsed = collapsedThreads.has(reply.id)}
        <div class="reply-container" class:collapsed={isCollapsed}>
          <div class="reply-line"></div>
          <div class="reply-content">
            <div class="reply-header">
              <span class="reply-type-badge" style:background={getReplyTypeColor(reply.replyType)}>
                {getReplyTypeLabel(reply.replyType)}
              </span>
              <span class="reply-author">
                {reply.authorCommitment ? `Anonymous (${reply.authorCommitment.slice(0, 8)}...)` : reply.author}
              </span>
              <span class="reply-time">{formatTimestamp(reply.createdAt)}</span>
              {#if reply.replies.length > 0}
                <button class="collapse-btn" on:click={() => toggleThread(reply.id)}>
                  {isCollapsed ? `+ ${countReplies(reply.replies)} hidden` : '−'}
                </button>
              {/if}
            </div>

            <p class="reply-text">{reply.content}</p>

            <div class="reply-actions">
              <button class="vote-btn up" on:click={() => voteReply(reply.id, 'up')}>
                ▲ {reply.votes.up}
              </button>
              <button class="vote-btn down" on:click={() => voteReply(reply.id, 'down')}>
                ▼ {reply.votes.down}
              </button>
              {#if canReply && depth < maxDepth}
                <button class="reply-btn small" on:click={() => expandedReplies.add(reply.id)}>
                  Reply
                </button>
              {/if}
            </div>

            <!-- Inline reply composer for nested replies -->
            {#if expandedReplies.has(reply.id)}
              <div class="inline-composer">
                <input
                  type="text"
                  placeholder="Quick reply..."
                  on:keydown={(e) => {
                    if (e.key === 'Enter' && (e.target as HTMLInputElement).value.trim()) {
                      dispatch('reply', {
                        parentId: reply.id,
                        content: (e.target as HTMLInputElement).value.trim(),
                        replyType: 'support',
                      });
                      (e.target as HTMLInputElement).value = '';
                      expandedReplies.delete(reply.id);
                      expandedReplies = new Set(expandedReplies);
                    }
                  }}
                />
                <button class="cancel-btn" on:click={() => {
                  expandedReplies.delete(reply.id);
                  expandedReplies = new Set(expandedReplies);
                }}>×</button>
              </div>
            {/if}

            <!-- Nested replies -->
            {#if !isCollapsed && reply.replies.length > 0}
              <svelte:self
                belief={belief}
                replies={reply.replies}
                {canReply}
                {maxDepth}
                depth={depth + 1}
                on:reply
                on:vote
              />
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .belief-thread {
    --indent: calc(var(--depth, 0) * 20px);
    margin-left: var(--indent);
  }

  .belief-root {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 16px;
    border: 1px solid var(--border, #334155);
    margin-bottom: 16px;
  }

  .belief-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }

  .belief-type {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--color-primary, #3b82f6);
    background: rgba(59, 130, 246, 0.15);
    padding: 2px 8px;
    border-radius: 4px;
  }

  .belief-time {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
    margin-left: auto;
  }

  .belief-content {
    margin: 0 0 12px;
    font-size: 15px;
    line-height: 1.6;
    color: var(--text-primary, #f8fafc);
  }

  .belief-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 12px;
  }

  .tag {
    font-size: 11px;
    padding: 2px 8px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 4px;
    color: var(--text-muted, #94a3b8);
  }

  .belief-actions {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border, #334155);
  }

  .reply-count {
    font-size: 12px;
    color: var(--text-muted, #94a3b8);
  }

  .reply-btn {
    padding: 6px 12px;
    font-size: 12px;
    background: var(--bg-secondary, #1e293b);
    border: 1px solid var(--border, #334155);
    border-radius: 4px;
    color: var(--text-primary, #f8fafc);
    cursor: pointer;
    transition: all 0.2s;
  }

  .reply-btn:hover {
    background: var(--border, #334155);
  }

  .reply-btn.small {
    padding: 4px 8px;
    font-size: 11px;
  }

  /* Reply composer */
  .reply-composer {
    margin-top: 12px;
    padding: 12px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 6px;
  }

  .composer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 12px;
    color: var(--text-muted, #94a3b8);
  }

  .reply-type-select {
    padding: 4px 8px;
    font-size: 11px;
    background: var(--bg-tertiary, #0f172a);
    border: 1px solid var(--border, #334155);
    border-radius: 4px;
    color: var(--text-primary, #f8fafc);
  }

  .reply-composer textarea {
    width: 100%;
    padding: 10px;
    background: var(--bg-tertiary, #0f172a);
    border: 1px solid var(--border, #334155);
    border-radius: 6px;
    color: var(--text-primary, #f8fafc);
    font-size: 13px;
    resize: vertical;
  }

  .reply-composer textarea:focus {
    outline: none;
    border-color: var(--color-primary, #3b82f6);
  }

  .composer-actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 8px;
  }

  .submit-btn {
    padding: 8px 16px;
    font-size: 12px;
    background: var(--color-primary, #3b82f6);
    border: none;
    border-radius: 4px;
    color: white;
    cursor: pointer;
    transition: background 0.2s;
  }

  .submit-btn:hover:not(:disabled) {
    background: #2563eb;
  }

  .submit-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Replies section */
  .replies-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .reply-container {
    display: flex;
    gap: 12px;
  }

  .reply-container.collapsed .reply-text,
  .reply-container.collapsed .reply-actions {
    display: none;
  }

  .reply-line {
    width: 2px;
    background: var(--border, #334155);
    border-radius: 1px;
    flex-shrink: 0;
  }

  .reply-content {
    flex: 1;
    min-width: 0;
  }

  .reply-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    flex-wrap: wrap;
  }

  .reply-type-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 3px;
    color: white;
  }

  .reply-author {
    font-size: 12px;
    color: var(--text-muted, #94a3b8);
  }

  .reply-time {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
    opacity: 0.7;
  }

  .collapse-btn {
    padding: 2px 6px;
    font-size: 11px;
    background: none;
    border: none;
    color: var(--color-primary, #3b82f6);
    cursor: pointer;
  }

  .reply-text {
    margin: 0 0 8px;
    font-size: 13px;
    line-height: 1.5;
    color: var(--text-primary, #f8fafc);
  }

  .reply-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .vote-btn {
    padding: 3px 8px;
    font-size: 11px;
    background: var(--bg-secondary, #1e293b);
    border: 1px solid var(--border, #334155);
    border-radius: 3px;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
    transition: all 0.2s;
  }

  .vote-btn:hover {
    border-color: var(--text-muted, #94a3b8);
  }

  .vote-btn.up:hover {
    color: #22c55e;
    border-color: #22c55e;
  }

  .vote-btn.down:hover {
    color: #ef4444;
    border-color: #ef4444;
  }

  /* Inline composer */
  .inline-composer {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }

  .inline-composer input {
    flex: 1;
    padding: 6px 10px;
    font-size: 12px;
    background: var(--bg-tertiary, #0f172a);
    border: 1px solid var(--border, #334155);
    border-radius: 4px;
    color: var(--text-primary, #f8fafc);
  }

  .inline-composer input:focus {
    outline: none;
    border-color: var(--color-primary, #3b82f6);
  }

  .cancel-btn {
    padding: 4px 8px;
    font-size: 14px;
    background: none;
    border: none;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
  }

  .cancel-btn:hover {
    color: var(--text-primary, #f8fafc);
  }
</style>
