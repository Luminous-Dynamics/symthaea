<script lang="ts">
  import type {
    IntentBundleSuggestion,
    IntentRequest,
    ListingWithContext,
    IntentBundleItem,
  } from '$types';
  import { initHolochainClient } from '$lib/holochain';
  import { getAllListings } from '$lib/holochain/listings';
  import { getUserProfile } from '$lib/holochain/users';
  import { getMockListings } from '$lib/mock/listings';
  import { analyzeRisk } from '$lib/risk';
  import { cartItems, getProofStatus } from '$lib/stores';
  import { goto } from '$app/navigation';

  const defaultIntent: IntentRequest = {
    title: 'Need a durable travel kit',
    categories: ['Travel', 'Accessories'],
    budgetMin: 50,
    budgetMax: 200,
    deliveryDays: 7,
    region: 'North America',
    mustHaveProof: true,
    allowBundles: true,
    notes: 'Water-resistant preferred, TSA compliant.',
  };

  let intent: IntentRequest = { ...defaultIntent };
  let categoryInput = defaultIntent.categories.join(', ');
  let suggestions: IntentBundleSuggestion[] = [];
  let loading = false;
  let usingMock = false;
  let availableListings: ListingWithContext['listing'][] = [];
  let riskScores: Record<string, number> = {};
  let proofLookup: Record<string, string> = {};
  let addingBundle = false;
  let sellerNames: Record<string, string> = {};
  let lastFitFactors: Record<string, { proof: number; risk: number; budget: number }> = {};
  const CACHE_KEY = 'intent_bundle_cache';

  function scoreBundles(): IntentBundleSuggestion[] {
    const fitsProof = intent.mustHaveProof;
    const listings = availableListings.map((l) => ({
      listing_id: l.listing_hash || l.id,
      title: l.title,
      seller: l.seller_agent_id,
      price: l.price,
      proof_status:
        proofLookup[l.listing_hash || l.id] === 'fulfilled'
          ? 'fulfilled'
          : (proofLookup[l.listing_hash || l.id] as 'pending' | 'none' | undefined) || 'none',
      risk_score: riskScores[l.listing_hash || l.id] || 0,
      delivery: '3-7 days',
      categories: [l.category as string],
    })) as (IntentBundleItem & { delivery: string; categories: string[] })[];

    const filtered = listings.filter((l) => {
      if (!intent.allowBundles && l.price > intent.budgetMax) return false;
      if (fitsProof && l.proof_status !== 'fulfilled') return false;
      if (intent.categories.length > 0 && !intent.categories.some((c) => l.categories.includes(c))) {
        return false;
      }
      return true;
    });

    const soloBundles: IntentBundleSuggestion[] = filtered
      .filter((l) => l.price >= intent.budgetMin && l.price <= intent.budgetMax)
      .map((l, idx) => {
        const proofFactor = l.proof_status === 'fulfilled' ? 0.3 : l.proof_status === 'pending' ? 0.1 : 0;
        const riskFactor = Math.max(0, 0.3 - (l.risk_score || 0));
        const budgetFactor = 0.4;
        const fit = proofFactor + riskFactor + budgetFactor;
        lastFitFactors[`solo-${idx}`] = {
          proof: proofFactor,
          risk: riskFactor,
          budget: budgetFactor,
        };
        return {
          id: `solo-${idx}`,
          total: l.price,
          deliveryEstimate: l.delivery,
          fitScore: fit,
          items: [l],
        };
      });

    const combo = filtered.slice(0, 3);
    const comboTotal = combo.reduce((sum, l) => sum + l.price, 0);
    const proofFactor =
      combo.filter((c) => c.proof_status === 'fulfilled').length / Math.max(combo.length, 1);
    const riskFactor = Math.max(
      0,
      0.3 - combo.reduce((sum, c) => sum + (c.risk_score || 0), 0) / Math.max(combo.length, 1)
    );
    const budgetFactor = comboTotal >= intent.budgetMin && comboTotal <= intent.budgetMax ? 0.4 : 0.1;

    const comboBundle: IntentBundleSuggestion = {
      id: 'bundle-1',
      total: comboTotal,
      deliveryEstimate: '4-6 days',
      fitScore: proofFactor + riskFactor + budgetFactor,
      items: combo,
    };
    lastFitFactors['bundle-1'] = {
      proof: proofFactor,
      risk: riskFactor,
      budget: budgetFactor,
    };

    return [...soloBundles, comboBundle].sort((a, b) => b.fitScore - a.fitScore);
  }

  function resetIntent() {
    intent = { ...defaultIntent };
    suggestions = [];
  }

  async function generateBundles() {
    loading = true;
    suggestions = [];
    await new Promise((resolve) => setTimeout(resolve, 100));
    suggestions = scoreBundles();
    loading = false;
  }

  async function loadListings() {
    loading = true;
    hydrateFromCache();
    try {
      const client = await initHolochainClient();
      const listings = await getAllListings(client);
      availableListings = listings;
      usingMock = false;
    } catch (e) {
      availableListings = getMockListings();
      usingMock = true;
    }
    const risk = await analyzeRisk(availableListings);
    riskScores = Object.keys(risk).reduce((acc, key) => {
      acc[key] = risk[key].score;
      return acc;
    }, {} as Record<string, number>);
    proofLookup = availableListings.reduce((acc, listing) => {
      const status = getProofStatus(listing.seller_agent_id, listing.listing_hash || listing.id);
      if (status) acc[listing.listing_hash || listing.id] = status;
      return acc;
    }, {} as Record<string, string>);
    await loadSellerNames();
    persistCache();
    loading = false;
  }

  loadListings();

  async function loadSellerNames() {
    try {
      const client = await initHolochainClient();
      const ids = Array.from(new Set(availableListings.map((l) => l.seller_agent_id)));
      const profiles = await Promise.all(
        ids.map((id) =>
          getUserProfile(client, id).catch(() => null)
        )
      );
      sellerNames = profiles.reduce((acc, profile) => {
        if (profile?.agent_id) {
          acc[profile.agent_id] = profile.username || profile.agent_id;
        }
        return acc;
      }, {} as Record<string, string>);
    } catch {
      // ignore lookup failures; fall back to agent id
    }
  }

  function persistCache() {
    if (typeof localStorage === 'undefined') return;
    const payload = {
      riskScores,
      proofLookup,
      sellerNames,
    };
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify(payload));
    } catch {
      // ignore
    }
  }

  function hydrateFromCache() {
    if (typeof localStorage === 'undefined') return;
    try {
      const raw = localStorage.getItem(CACHE_KEY);
      if (!raw) return;
      const cache = JSON.parse(raw) as {
        riskScores?: Record<string, number>;
        proofLookup?: Record<string, string>;
        sellerNames?: Record<string, string>;
      };
      riskScores = cache.riskScores || {};
      proofLookup = cache.proofLookup || {};
      sellerNames = cache.sellerNames || {};
    } catch {
      // ignore
    }
  }

  async function addBundleToCart(bundle: IntentBundleSuggestion) {
    addingBundle = true;
    const lookup = new Map(availableListings.map((l) => [l.listing_hash || l.id, l]));
    bundle.items.forEach((item) => {
      const listing = lookup.get(item.listing_id);
      cartItems.addItem({
        listing_hash: item.listing_id,
        title: item.title,
        price: item.price,
        quantity: 1,
        photo_cid: listing?.photos_ipfs_cids?.[0],
        seller_agent_id: listing?.seller_agent_id || item.seller,
        seller_name: item.seller,
      });
    });
    addingBundle = false;
    goto('/cart');
  }
</script>

<svelte:head>
  <title>Intent-Based Bundles | Mycelix</title>
</svelte:head>

<div class="intent-page">
  <div class="container">
    <header class="page-header">
      <div>
        <p class="eyebrow">Intent-based buying</p>
        <h1>Compose bundles from your intent</h1>
        <p class="lede">
          Describe constraints, we suggest multi-seller bundles that satisfy budget, region, and proof requirements.
        </p>
      </div>
      <div class="actions">
        <button
          class={`btn btn-ghost ${intent.mustHaveProof ? 'active' : ''}`}
          type="button"
          on:click={() => (intent.mustHaveProof = !intent.mustHaveProof)}
        >
          {intent.mustHaveProof ? 'Proof-fulfilled only ✓' : 'Require proof-fulfilled'}
        </button>
        <button class="btn btn-secondary" on:click={resetIntent}>Reset</button>
        <button class="btn btn-primary" on:click={generateBundles} disabled={loading}>
          {loading ? 'Scoring…' : 'Generate bundles'}
        </button>
      </div>
    </header>

    <div class="layout">
      <form class="intent-form" on:submit|preventDefault={generateBundles}>
        <label>
          Title
          <input type="text" bind:value={intent.title} placeholder="e.g., Need a durable travel kit" />
        </label>

        <label>
          Categories (comma separated)
          <input
            type="text"
            bind:value={categoryInput}
            on:change={() => {
              intent.categories = categoryInput
                .split(',')
                .map((s) => s.trim())
                .filter(Boolean);
            }}
            placeholder="Travel, Accessories"
          />
        </label>

        <div class="row">
          <label>
            Budget min ($)
            <input type="number" min="0" bind:value={intent.budgetMin} />
          </label>
          <label>
            Budget max ($)
            <input type="number" min="0" bind:value={intent.budgetMax} />
          </label>
        </div>

        <div class="row">
          <label>
            Delivery (days)
            <input type="number" min="1" bind:value={intent.deliveryDays} />
          </label>
          <label>
            Region
            <input type="text" bind:value={intent.region} placeholder="North America" />
          </label>
        </div>

        <label>
          Notes
          <textarea rows="3" bind:value={intent.notes} placeholder="Constraints, materials, must-haves"></textarea>
        </label>

        <div class="checkboxes">
          <label>
            <input type="checkbox" bind:checked={intent.mustHaveProof} />
            Must be proof-fulfilled
          </label>
          <label>
            <input type="checkbox" bind:checked={intent.allowBundles} />
            Allow multi-seller bundles
          </label>
        </div>

        <button class="btn btn-primary" type="submit" disabled={loading}>
          {loading ? 'Scoring…' : 'Find matches'}
        </button>
      </form>

      <section class="suggestions">
        <h2>Suggested bundles</h2>
        {#if loading}
          <div class="loading">
            <div class="spinner"></div>
            <p>Scoring bundles…</p>
          </div>
        {:else if suggestions.length === 0}
          <p class="muted">No suggestions yet. Describe your intent and generate bundles.</p>
        {:else}
          <div class="bundles">
            {#each suggestions as suggestion}
              <div class="bundle-card">
                <div class="bundle-header">
                  <h3>Bundle {suggestion.id}</h3>
                  <div class="fit">Fit {(suggestion.fitScore * 100).toFixed(0)}%</div>
                </div>
                {#if lastFitFactors[suggestion.id]}
                  {@const factors = lastFitFactors[suggestion.id]}
                  <p
                    class="bundle-meta"
                    title={`Proof ${(factors.proof * 100).toFixed(0)}% · Risk ${(factors.risk * 100).toFixed(0)}% · Budget ${(factors.budget * 100).toFixed(0)}%`}
                  >
                    ${suggestion.total.toFixed(2)} · {suggestion.deliveryEstimate}
                  </p>
                  <div class="fit-breakdown">
                    <span>Proof {(factors.proof * 100).toFixed(0)}%</span>
                    <span>Risk {(factors.risk * 100).toFixed(0)}%</span>
                    <span>Budget {(factors.budget * 100).toFixed(0)}%</span>
                  </div>
                {:else}
                  <p class="bundle-meta">
                    ${suggestion.total.toFixed(2)} · {suggestion.deliveryEstimate}
                  </p>
                {/if}
                <ul class="bundle-items">
                  {#each suggestion.items as item}
                    <li>
                      <div>
                        <p class="item-title">{item.title}</p>
                        <p class="item-meta">
                          {sellerNames[item.seller] || item.seller} · ${item.price.toFixed(2)} · {item.listing_id.slice(0, 6)}...
                        </p>
                      </div>
                      <span class={`proof-pill ${item.proof_status}`}>
                        {item.proof_status}
                      </span>
                      <button
                        class="link-button"
                        type="button"
                        on:click={() => goto(`/listing/${item.listing_id}#trust`)}
                      >
                        View trust →
                      </button>
                    </li>
                  {/each}
                </ul>
                <button class="btn btn-secondary" on:click={() => addBundleToCart(suggestion)} disabled={addingBundle}>
                  {addingBundle ? 'Adding…' : 'Add bundle to cart'}
                </button>
            </div>
          {/each}
        </div>
      {/if}
    </section>
    </div>
  </div>
</div>

<style>
  .intent-page {
    min-height: 100vh;
    background: #f7fafc;
    padding: 2rem 1rem;
  }

  .container {
    max-width: 1200px;
    margin: 0 auto;
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
  }

  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: #6366f1;
    margin: 0 0 0.25rem;
  }

  h1 {
    margin: 0;
    color: #0f172a;
  }

  .lede {
    margin: 0.25rem 0 0;
    color: #475569;
    max-width: 640px;
  }

  .actions {
    display: flex;
    gap: 0.75rem;
  }

  .btn-ghost {
    background: #f8fafc;
    color: #1f2937;
    border: 1px solid #e2e8f0;
  }

  .btn-ghost.active {
    border-color: #4f46e5;
    color: #4338ca;
    background: #eef2ff;
  }

  .layout {
    display: grid;
    grid-template-columns: 1fr 1.2fr;
    gap: 1.5rem;
  }

  .intent-form {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    color: #334155;
    font-weight: 600;
  }

  input,
  textarea {
    padding: 0.65rem 0.85rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    font-size: 0.95rem;
  }

  .row {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
  }

  .checkboxes {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .suggestions {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 1.25rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
  }

  .bundles {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
  }

  .bundle-card {
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 0.9rem;
    background: #f8fafc;
  }

  .bundle-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
  }

  .fit {
    background: #eef2ff;
    color: #4338ca;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-weight: 700;
  }

  .bundle-meta {
    margin: 0 0 0.5rem;
    color: #475569;
  }

  .fit-breakdown {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin: 0 0 0.5rem;
    color: #475569;
    font-size: 0.9rem;
  }

  .bundle-items {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .bundle-items li {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.35rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 0.5rem 0.6rem;
    background: white;
  }

  .item-title {
    margin: 0;
    font-weight: 700;
    color: #0f172a;
  }

  .item-meta {
    margin: 0;
    color: #475569;
    font-size: 0.9rem;
  }

  .bundle-items .link-button {
    border: none;
    background: none;
    color: #4338ca;
    font-weight: 700;
    cursor: pointer;
  }

  .proof-pill {
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-weight: 700;
    text-transform: capitalize;
    border: 1px solid transparent;
  }

  .proof-pill.fulfilled {
    background: #dcfce7;
    color: #166534;
    border-color: #bbf7d0;
  }

  .proof-pill.pending {
    background: #eef2ff;
    color: #4338ca;
    border-color: #c7d2fe;
  }

  .proof-pill.none {
    background: #fff7ed;
    color: #c2410c;
    border-color: #fed7aa;
  }

  .loading {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .spinner {
    width: 28px;
    height: 28px;
    border: 3px solid #e2e8f0;
    border-top-color: #6366f1;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .muted {
    color: #94a3b8;
  }

  .btn {
    border: none;
    border-radius: 0.5rem;
    padding: 0.65rem 1rem;
    font-weight: 700;
    cursor: pointer;
  }

  .btn-primary {
    background: #4f46e5;
    color: white;
  }

  .btn-secondary {
    background: #edf2f7;
    color: #1f2937;
    border: 1px solid #e2e8f0;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  @media (max-width: 960px) {
    .layout {
      grid-template-columns: 1fr;
    }

    .actions {
      width: 100%;
      justify-content: flex-end;
    }
  }
</style>
