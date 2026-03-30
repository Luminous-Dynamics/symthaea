<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import {
    MarketCard,
    PredictionForm,
    CalibrationProfile,
    WisdomSeedDisplay,
    type Market,
    type CalibrationProfile as CalibrationProfileType,
  } from '$lib/components/epistemic-markets';

  // State
  let markets: Market[] = [];
  let myProfile: CalibrationProfileType | null = null;
  let selectedMarket: Market | null = null;
  let activeTab: 'markets' | 'questions' | 'wisdom' | 'profile' = 'markets';
  let loading = true;

  // Demo data for illustration
  onMount(async () => {
    // In production, this would fetch from Holochain
    await loadDemoData();
    loading = false;
  });

  async function loadDemoData() {
    // Simulated markets for demonstration
    markets = [
      {
        id: 'market-001',
        title: 'Will the community garden project have 50+ active participants by year end?',
        description: 'Success is defined as having at least 50 individuals who have contributed at least 10 hours in the past 3 months.',
        outcomes: [
          { name: 'Yes', probability: 0.72, volume: 15420 },
          { name: 'No', probability: 0.28, volume: 6180 },
        ],
        epistemic_position: {
          empirical: 'testimonial',
          normative: 'communal',
          materiality: 'persistent',
        },
        mechanism: { type: 'LMSR' },
        status: 'open',
        created_at: Date.now() - 86400000 * 30,
        closes_at: Date.now() + 86400000 * 60,
        total_volume: 21600,
        prediction_count: 47,
        creator: 'uhCAk...abc123',
        resolution_config: {
          min_oracles: 7,
          consensus_threshold: 0.75,
          byzantine_tolerance: 0.34,
        },
      },
      {
        id: 'market-002',
        title: 'What will be the average solar panel efficiency improvement by 2027?',
        description: 'Measured as percentage improvement over current commercial panel efficiency (~22%)',
        outcomes: [
          { name: '<5%', probability: 0.15, volume: 3200 },
          { name: '5-10%', probability: 0.45, volume: 9800 },
          { name: '10-15%', probability: 0.30, volume: 6500 },
          { name: '>15%', probability: 0.10, volume: 2100 },
        ],
        epistemic_position: {
          empirical: 'measurable',
          normative: 'universal',
          materiality: 'foundational',
        },
        mechanism: { type: 'Parimutuel' },
        status: 'open',
        created_at: Date.now() - 86400000 * 45,
        closes_at: Date.now() + 86400000 * 365,
        total_volume: 21600,
        prediction_count: 128,
        creator: 'uhCAk...def456',
        resolution_config: {
          min_oracles: 3,
          consensus_threshold: 0.95,
          byzantine_tolerance: 0.34,
        },
      },
      {
        id: 'market-003',
        title: 'Will the governance proposal for treasury diversification pass?',
        description: 'MIP-47: Proposal to allocate 20% of treasury to regenerative agriculture investments',
        outcomes: [
          { name: 'Pass', probability: 0.58, volume: 45200 },
          { name: 'Fail', probability: 0.42, volume: 32800 },
        ],
        epistemic_position: {
          empirical: 'cryptographic',
          normative: 'network',
          materiality: 'persistent',
        },
        mechanism: { type: 'Binary' },
        status: 'open',
        created_at: Date.now() - 86400000 * 7,
        closes_at: Date.now() + 86400000 * 7,
        total_volume: 78000,
        prediction_count: 234,
        creator: 'uhCAk...ghi789',
        resolution_config: {
          min_oracles: 5,
          consensus_threshold: 0.9,
          byzantine_tolerance: 0.34,
        },
      },
    ];

    myProfile = {
      agent: 'uhCAktest12345678901234567890',
      brier_score: 0.18,
      calibration_buckets: [
        { range_start: 0.0, range_end: 0.2, prediction_count: 12, actual_frequency: 0.08 },
        { range_start: 0.2, range_end: 0.4, prediction_count: 18, actual_frequency: 0.33 },
        { range_start: 0.4, range_end: 0.6, prediction_count: 25, actual_frequency: 0.52 },
        { range_start: 0.6, range_end: 0.8, prediction_count: 30, actual_frequency: 0.70 },
        { range_start: 0.8, range_end: 1.0, prediction_count: 15, actual_frequency: 0.87 },
      ],
      domain_scores: {
        governance: { prediction_count: 23, accuracy: 0.78, reputation: 85, calibration: 0.82 },
        technology: { prediction_count: 31, accuracy: 0.71, reputation: 72, calibration: 0.75 },
        community: { prediction_count: 18, accuracy: 0.83, reputation: 91, calibration: 0.88 },
        climate: { prediction_count: 12, accuracy: 0.67, reputation: 58, calibration: 0.70 },
      },
      resolution_score: 0.76,
      reliability_score: 0.82,
      epistemic_virtues: {
        updates_beliefs: 0.85,
        acknowledges_uncertainty: 0.72,
        productive_disagreement: 0.68,
        reasoning_transparency: 0.91,
      },
      total_predictions: 84,
      created_at: Date.now() - 86400000 * 180,
      updated_at: Date.now(),
    };
  }

  function handleMarketSelect(market: Market) {
    selectedMarket = selectedMarket?.id === market.id ? null : market;
  }

  function handlePredictionSubmit(event: CustomEvent) {
    console.log('Prediction submitted:', event.detail);
    selectedMarket = null;
    // In production: call Holochain zome function
  }
</script>

<svelte:head>
  <title>Epistemic Markets | Mycelix Observatory</title>
</svelte:head>

<div class="epistemic-markets-page">
  <!-- Header -->
  <header class="page-header">
    <div class="header-content">
      <h1>Epistemic Markets</h1>
      <p class="subtitle">Collective truth-seeking through prediction and wisdom</p>
    </div>
    <div class="header-stats">
      <div class="stat">
        <span class="stat-value">{markets.length}</span>
        <span class="stat-label">Active Markets</span>
      </div>
      <div class="stat">
        <span class="stat-value">{markets.reduce((sum, m) => sum + m.prediction_count, 0)}</span>
        <span class="stat-label">Predictions</span>
      </div>
      <div class="stat">
        <span class="stat-value">{(markets.reduce((sum, m) => sum + m.total_volume, 0) / 1000).toFixed(1)}k</span>
        <span class="stat-label">Total Volume</span>
      </div>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <nav class="tabs">
    <button
      class="tab"
      class:active={activeTab === 'markets'}
      on:click={() => activeTab = 'markets'}
    >
      Markets
    </button>
    <button
      class="tab"
      class:active={activeTab === 'questions'}
      on:click={() => activeTab = 'questions'}
    >
      Question Markets
    </button>
    <button
      class="tab"
      class:active={activeTab === 'wisdom'}
      on:click={() => activeTab = 'wisdom'}
    >
      Wisdom Seeds
    </button>
    <button
      class="tab"
      class:active={activeTab === 'profile'}
      on:click={() => activeTab = 'profile'}
    >
      My Calibration
    </button>
  </nav>

  <!-- Content -->
  <main class="content">
    {#if loading}
      <div class="loading">
        <div class="loading-spinner"></div>
        <p>Connecting to the epistemic network...</p>
      </div>
    {:else if activeTab === 'markets'}
      <div class="markets-layout">
        <!-- Market List -->
        <section class="market-list">
          <div class="section-header">
            <h2>Active Prediction Markets</h2>
            <button class="create-btn">+ Create Market</button>
          </div>

          <div class="market-grid">
            {#each markets as market (market.id)}
              <MarketCard
                {market}
                showDetails={selectedMarket?.id === market.id}
                on:click={() => handleMarketSelect(market)}
              />
            {/each}
          </div>
        </section>

        <!-- Prediction Form (shown when market selected) -->
        {#if selectedMarket}
          <aside class="prediction-panel">
            <PredictionForm
              market={selectedMarket}
              on:submit={handlePredictionSubmit}
            />
          </aside>
        {/if}
      </div>

    {:else if activeTab === 'questions'}
      <div class="questions-section">
        <div class="section-header">
          <h2>Question Markets</h2>
          <p class="section-description">
            What questions are worth answering? Trade on the value of knowledge itself.
          </p>
        </div>

        <div class="coming-soon">
          <div class="coming-soon-icon">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/>
            </svg>
          </div>
          <h3>Coming Soon</h3>
          <p>Question Markets will let you trade on which questions are most valuable to answer.</p>
        </div>
      </div>

    {:else if activeTab === 'wisdom'}
      <div class="wisdom-section">
        <div class="section-header">
          <h2>Wisdom Seeds</h2>
          <p class="section-description">
            Lessons from past predictions, maturing over time into collective wisdom.
          </p>
        </div>

        <div class="wisdom-preview">
          <WisdomSeedDisplay
            wisdomSeed={{
              if_correct: {
                lesson: 'Community projects succeed when there is sustained early engagement from diverse stakeholders',
                implication: 'Focus on breadth of participation, not just depth',
                domain_relevance: ['community_organizing', 'governance', 'social_movements'],
              },
              if_incorrect: {
                lesson: 'Early enthusiasm does not guarantee long-term commitment without structural support',
                implication: 'Build infrastructure for sustained participation alongside initial excitement',
                domain_relevance: ['community_organizing', 'psychology', 'sustainability'],
              },
              meta_lesson: 'The relationship between initial energy and final success is mediated by the structures we build to channel that energy.',
              letter_to_future: 'Dear future predictor: Watch not just the spark, but the vessel that holds the flame.',
            }}
            resolved={true}
            outcome="correct"
            dormancyComplete={true}
          />
        </div>
      </div>

    {:else if activeTab === 'profile'}
      <div class="profile-section">
        {#if myProfile}
          <CalibrationProfile profile={myProfile} showDetails={true} />
        {:else}
          <div class="no-profile">
            <p>Make your first prediction to start building your calibration profile.</p>
          </div>
        {/if}
      </div>
    {/if}
  </main>
</div>

<style>
  .epistemic-markets-page {
    min-height: 100vh;
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: white;
  }

  .page-header {
    padding: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .header-content h1 {
    margin: 0;
    font-size: 2rem;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .subtitle {
    margin: 0.5rem 0 0 0;
    color: rgba(255, 255, 255, 0.6);
  }

  .header-stats {
    display: flex;
    gap: 2rem;
  }

  .stat {
    text-align: center;
  }

  .stat-value {
    display: block;
    font-size: 1.5rem;
    font-weight: 700;
    color: #10b981;
  }

  .stat-label {
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
  }

  .tabs {
    display: flex;
    padding: 0 2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .tab {
    padding: 1rem 1.5rem;
    background: transparent;
    border: none;
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.95rem;
    cursor: pointer;
    transition: all 0.2s ease;
    border-bottom: 2px solid transparent;
  }

  .tab:hover {
    color: white;
  }

  .tab.active {
    color: #a78bfa;
    border-bottom-color: #a78bfa;
  }

  .content {
    padding: 2rem;
  }

  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem;
    color: rgba(255, 255, 255, 0.6);
  }

  .loading-spinner {
    width: 40px;
    height: 40px;
    border: 3px solid rgba(255, 255, 255, 0.1);
    border-top-color: #a78bfa;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .markets-layout {
    display: grid;
    grid-template-columns: 1fr;
    gap: 2rem;
  }

  @media (min-width: 1024px) {
    .markets-layout {
      grid-template-columns: 2fr 1fr;
    }
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .section-header h2 {
    margin: 0;
    font-size: 1.25rem;
  }

  .section-description {
    color: rgba(255, 255, 255, 0.6);
    margin: 0.5rem 0 0 0;
  }

  .create-btn {
    padding: 0.5rem 1rem;
    background: #10b981;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.2s ease;
  }

  .create-btn:hover {
    background: #059669;
  }

  .market-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1.5rem;
  }

  .prediction-panel {
    position: sticky;
    top: 2rem;
    max-height: calc(100vh - 4rem);
    overflow-y: auto;
  }

  .questions-section,
  .wisdom-section,
  .profile-section {
    max-width: 800px;
  }

  .coming-soon {
    text-align: center;
    padding: 4rem 2rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    border: 1px dashed rgba(255, 255, 255, 0.2);
  }

  .coming-soon-icon {
    width: 64px;
    height: 64px;
    margin: 0 auto 1rem;
    color: #a78bfa;
  }

  .coming-soon h3 {
    margin: 0 0 0.5rem 0;
    color: #a78bfa;
  }

  .coming-soon p {
    color: rgba(255, 255, 255, 0.6);
    margin: 0;
  }

  .wisdom-preview {
    margin-top: 1rem;
  }

  .no-profile {
    text-align: center;
    padding: 4rem 2rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    color: rgba(255, 255, 255, 0.6);
  }
</style>
