<script lang="ts">
  import type { ProofTrailItem } from '$types';

  export let items: ProofTrailItem[] = [];

  const formatDate = (timestamp: number) =>
    new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
</script>

<section class="proof-trail">
  <header class="trail-header">
    <div>
      <p class="eyebrow">Proof trail</p>
      <h4>Evidence & hashes</h4>
    </div>
    <span class="count">{items.length} entries</span>
  </header>

  {#if !items.length}
    <p class="muted">No proofs published yet.</p>
  {:else}
    <ul class="trail-list">
      {#each items as item (item.id)}
        <li class="trail-item">
          <div class="row">
            <div>
              <p class="label">{item.label}</p>
              <p class="meta">
                Issuer {item.issuer} · {formatDate(item.issued_at)}
                {#if item.cid}
                  · CID {item.cid}
                {:else if item.hash}
                  · Hash {item.hash}
                {/if}
              </p>
            </div>
            <span class:verified={item.verified}>
              {item.verified ? 'verified' : 'pending'}
            </span>
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .proof-trail {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0.9rem;
    padding: 1rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
  }

  .trail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    font-weight: 700;
    color: #6366f1;
    margin: 0;
  }

  h4 {
    margin: 0.15rem 0 0;
    color: #0f172a;
  }

  .count {
    background: #eef2ff;
    color: #4338ca;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-weight: 700;
  }

  .trail-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .trail-item {
    border: 1px solid #e2e8f0;
    border-radius: 0.8rem;
    padding: 0.75rem 0.85rem;
    background: #f8fafc;
  }

  .row {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: flex-start;
  }

  .label {
    margin: 0;
    font-weight: 700;
    color: #0f172a;
  }

  .meta {
    margin: 0.35rem 0 0;
    color: #475569;
    font-size: 0.9rem;
  }

  span.verified {
    background: #dcfce7;
    color: #15803d;
  }

  span:not(.verified) {
    background: #fff7ed;
    color: #c2410c;
  }

  span {
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-weight: 700;
    border: 1px solid rgba(0, 0, 0, 0.05);
    white-space: nowrap;
  }

  .muted {
    color: #94a3b8;
    margin: 0;
  }

  @media (max-width: 640px) {
    .row {
      flex-direction: column;
      align-items: flex-start;
    }

    span {
      margin-top: 0.5rem;
    }
  }
</style>
