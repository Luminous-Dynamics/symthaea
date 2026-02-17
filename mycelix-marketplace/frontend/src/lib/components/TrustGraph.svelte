<script lang="ts">
  import type { ReputationClaim, TrustGraphEdge, TrustGraphNode } from '$types';

  export let subject: string;
  export let nodes: TrustGraphNode[] = [];
  export let edges: TrustGraphEdge[] = [];
  export let claims: ReputationClaim[] = [];
  export let summary: {
    score: number;
    confidence: number;
    attestations: number;
    zk_capable: boolean;
    last_update: number;
  } | null = null;
  export let updatedAt: number | null = null;
  export let usingMock: boolean = false;

  const formatScore = (value: number) => (value > 1 ? value : value * 100);

  $: subjectNode = nodes.find((node) => node.id === subject);
  $: peerNodes = nodes.filter((node) => node.id !== subject);
  $: latestClaims = claims.slice(0, 3);

  function roleLabel(node?: TrustGraphNode) {
    if (!node?.role) return 'peer';
    if (node.role === 'arbitrator') return 'arbitrator';
    return node.role;
  }

  function formatRelative(timestamp?: number) {
    if (!timestamp) return '';
    const diff = Date.now() - timestamp;
    const minutes = Math.floor(diff / (1000 * 60));
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }
</script>

<section class="trust-graph">
  <header class="graph-header">
    <div>
      <p class="eyebrow">Verifiable trust graph</p>
      <h3>Proof-backed confidence</h3>
      <p class="lede">
        Aggregated attestations from peers, arbitrators, and past trades — ready for zk range proofs.
      </p>
      <div class="live-meta">
        <span class={`pill ${usingMock ? 'pill-mock' : 'pill-live'}`}>
          {usingMock ? 'Preview (mock)' : 'Live data'}
        </span>
        {#if updatedAt}
          <span class="timestamp">Updated {formatRelative(updatedAt)}</span>
        {/if}
      </div>
    </div>
    {#if summary}
      <div class="score-card">
        <div class="score-value">{formatScore(summary.score).toFixed(1)}%</div>
        <p class="score-label">PoGQ score</p>
        <div class="score-meta">
          <span>Confidence {Math.round(summary.confidence * 100)}%</span>
          <span>{summary.attestations} attestations</span>
          <span class:active={summary.zk_capable}>zk-ready</span>
        </div>
        <p class="timestamp">Updated {formatRelative(summary.last_update)}</p>
      </div>
    {/if}
  </header>

  <div class="graph-body">
    <div class="graph-canvas" aria-label="Trust graph">
      <div class="subject-node">
        <p class="label">{subjectNode?.label || 'Seller'}</p>
        <p class="score">{formatScore(subjectNode?.score ?? summary?.score ?? 0).toFixed(0)}%</p>
        <p class="role">{roleLabel(subjectNode)}</p>
      </div>

      <div class="peer-ring">
        {#each peerNodes as node (node.id)}
          <div class="peer-node">
            <p class="label">{node.label}</p>
            <p class="score">{formatScore(node.score).toFixed(0)}%</p>
            <p class="role">{roleLabel(node)}</p>
          </div>
        {/each}
      </div>
    </div>

    <div class="edges-panel">
      <div class="edge-list">
        {#if edges.length === 0}
          <p class="muted">No trust edges yet.</p>
        {:else}
          {#each edges as edge, index (edge.from + edge.to + index)}
            <div class="edge-card">
              <div class="edge-row">
                <span class="edge-kind">{edge.kind}</span>
                <span class="edge-weight">{Math.round(edge.weight * 100)}%</span>
              </div>
              <p class="edge-desc">
                {edge.from} → {edge.to}
              </p>
              {#if edge.evidence_cid}
                <p class="edge-proof">Proof CID: {edge.evidence_cid}</p>
              {/if}
            </div>
          {/each}
        {/if}
      </div>

      <div class="claims-panel">
        <div class="claims-header">
          <span class="eyebrow">Latest claims</span>
          <span class="count">{claims.length} total</span>
        </div>
        {#if latestClaims.length === 0}
          <p class="muted">No claims yet.</p>
        {:else}
          <ul class="claims-list">
            {#each latestClaims as claim (claim.id)}
              <li class="claim">
                <div class="claim-top">
                  <span class="pill">{claim.claim_type}</span>
                  <span class="score">{Math.round(claim.score)} pts</span>
                </div>
                <p class="claim-text">{claim.description}</p>
                <p class="claim-meta">
                  Issuer {claim.issuer} · {formatRelative(claim.issued_at)}
                  {#if claim.zk_range_proof}
                    · zk range {claim.zk_range_proof.lower_bound}-{claim.zk_range_proof.upper_bound}%
                  {/if}
                </p>
              </li>
            {/each}
          </ul>
        {/if}
      </div>
    </div>
  </div>
</section>

<style>
  .trust-graph {
    background: radial-gradient(circle at 20% 20%, #f1f5f9 0, #f8fafc 40%, #ffffff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 14px 44px rgba(67, 56, 202, 0.08);
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  .graph-header {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
  }

  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    font-weight: 700;
    color: #6366f1;
    margin: 0 0 0.25rem 0;
  }

  h3 {
    margin: 0;
    color: #0f172a;
    font-size: 1.35rem;
  }

  .lede {
    margin: 0.25rem 0 0;
    color: #475569;
    max-width: 520px;
    line-height: 1.6;
  }

  .live-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.35rem;
  }

  .pill {
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
  }

  .pill-live {
    background: #dcfce7;
    color: #15803d;
    border: 1px solid #bbf7d0;
  }

  .pill-mock {
    background: #fff7ed;
    color: #c2410c;
    border: 1px solid #fed7aa;
  }

  .score-card {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 0.9rem;
    min-width: 240px;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
  }

  .score-value {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
  }

  .score-label {
    margin: 0;
    opacity: 0.85;
    font-weight: 600;
  }

  .score-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 0.75rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    opacity: 0.92;
  }

  .score-meta span.active {
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.2);
    font-weight: 700;
  }

  .timestamp {
    margin: 0;
    font-size: 0.8rem;
    opacity: 0.82;
  }

  .graph-body {
    display: grid;
    grid-template-columns: minmax(320px, 1fr) 1.1fr;
    gap: 1.25rem;
  }

  .graph-canvas {
    position: relative;
    background: radial-gradient(circle at 50% 50%, #eef2ff, #ffffff 60%);
    border-radius: 1rem;
    padding: 1rem;
    min-height: 340px;
    overflow: hidden;
    border: 1px dashed #e2e8f0;
  }

  .subject-node {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: white;
    border: 2px solid #4f46e5;
    display: grid;
    place-items: center;
    gap: 0.15rem;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    text-align: center;
    padding: 0.5rem;
  }

  .subject-node .label {
    margin: 0;
    font-weight: 700;
    color: #0f172a;
  }

  .subject-node .score {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 800;
    color: #4f46e5;
  }

  .subject-node .role {
    margin: 0;
    color: #475569;
    font-size: 0.9rem;
  }

  .peer-ring {
    position: absolute;
    inset: 1.25rem;
    display: grid;
    place-items: center;
    gap: 0.75rem;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  }

  .peer-node {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0.8rem;
    padding: 0.75rem;
    text-align: center;
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.05);
  }

  .peer-node .label {
    margin: 0;
    font-weight: 700;
    color: #111827;
  }

  .peer-node .score {
    margin: 0.15rem 0;
    color: #2563eb;
    font-weight: 800;
  }

  .peer-node .role {
    margin: 0;
    color: #64748b;
    font-size: 0.85rem;
  }

  .edges-panel {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
  }

  .edge-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
  }

  .edge-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 0.75rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
  }

  .edge-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
  }

  .edge-kind {
    text-transform: capitalize;
    font-weight: 700;
    color: #0f172a;
  }

  .edge-weight {
    font-weight: 700;
    color: #0ea5e9;
  }

  .edge-desc {
    margin: 0;
    color: #475569;
    font-size: 0.95rem;
  }

  .edge-proof {
    margin: 0.2rem 0 0;
    color: #6366f1;
    font-size: 0.85rem;
    word-break: break-all;
  }

  .claims-panel {
    background: #0f172a;
    color: white;
    border-radius: 0.9rem;
    padding: 0.9rem;
    border: 1px solid #1e293b;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.35);
  }

  .claims-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .count {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-size: 0.85rem;
  }

  .claims-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .claim {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0.75rem;
    padding: 0.6rem 0.7rem;
  }

  .claim-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
  }

  .pill {
    background: rgba(99, 102, 241, 0.2);
    color: #c7d2fe;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    text-transform: capitalize;
    font-weight: 700;
    font-size: 0.85rem;
  }

  .claim .score {
    color: #38bdf8;
    font-weight: 800;
  }

  .claim-text {
    margin: 0;
    color: #e2e8f0;
    line-height: 1.45;
  }

  .claim-meta {
    margin: 0.3rem 0 0;
    font-size: 0.85rem;
    color: #cbd5e1;
  }

  .muted {
    color: #94a3b8;
    margin: 0;
  }

  @media (max-width: 960px) {
    .graph-body {
      grid-template-columns: 1fr;
    }

    .graph-canvas {
      min-height: 280px;
    }

    .peer-ring {
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    }
  }
</style>
