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
  import { browser } from '$app/environment';
  import {
    getBalance,
    getOracleState,
    getAllPlots,
    getMyHearths,
    getEmergencyPlan,
    getAllWaterSystems,
    getLowStockItems,
    type BalanceInfo,
    type OracleState,
    type FoodPlot,
    type EmergencyPlan,
    type WaterSystem,
    type LowStockItem,
  } from '$lib/resilience-client';

  // ============================================================================
  // State
  // ============================================================================

  let loading = true;
  let error = '';
  let generatedAt = '';

  let balance: BalanceInfo | null = null;
  let oracle: OracleState | null = null;
  let plots: FoodPlot[] = [];
  let emergencyPlan: EmergencyPlan | null = null;
  let waterSystems: WaterSystem[] = [];
  let lowStock: LowStockItem[] = [];

  // Read community config from localStorage (set via /admin Community Config panel)
  function loadCommunityName(): string {
    if (!browser) return 'Mycelix Resilience DAO';
    try {
      const raw = localStorage.getItem('mycelix-community-config');
      if (raw) {
        const cfg = JSON.parse(raw);
        if (cfg.name) return cfg.name;
      }
    } catch { /* ignore parse errors */ }
    return 'Mycelix Resilience DAO';
  }

  function loadCommunityContact(): { name: string; phone: string } | null {
    if (!browser) return null;
    try {
      const raw = localStorage.getItem('mycelix-community-config');
      if (raw) {
        const cfg = JSON.parse(raw);
        if (cfg.contactName || cfg.contactPhone) {
          return { name: cfg.contactName || '', phone: cfg.contactPhone || '' };
        }
      }
    } catch { /* ignore parse errors */ }
    return null;
  }

  const communityName = loadCommunityName();
  const communityContact = loadCommunityContact();

  // ============================================================================
  // Derived values
  // ============================================================================

  $: totalPlotArea = plots.reduce((sum, p) => sum + p.area_sqm, 0);
  $: totalWaterCapacity = waterSystems.reduce((sum, s) => sum + s.capacity_liters, 0);

  // Demurrage: TEND uses a 5% annual demurrage (Silvio Gesell model)
  // Effective balance accounts for time-decay of stored hours
  $: effectiveBalance = balance ? Math.round(balance.balance * 0.95 * 100) / 100 : 0;
  $: demurrageRate = '5% / year';

  // ============================================================================
  // Lifecycle
  // ============================================================================

  onMount(async () => {
    const now = new Date();
    generatedAt = now.toISOString().replace('T', ' ').slice(0, 16);

    try {
      const [bal, ora, pls, hearths, water, stock] = await Promise.all([
        getBalance('self.did'),
        getOracleState(),
        getAllPlots(),
        getMyHearths(),
        getAllWaterSystems(),
        getLowStockItems(),
      ]);

      balance = bal;
      oracle = ora;
      plots = pls;
      waterSystems = water;
      lowStock = stock;

      // Fetch emergency plan from first hearth
      if (hearths.length > 0) {
        emergencyPlan = await getEmergencyPlan(hearths[0].id);
      }
    } catch (e) {
      console.warn('[Print] Failed to load data:', e);
      error = e instanceof Error ? e.message : 'Failed to load community data';
    } finally {
      loading = false;
    }
  });

  function handlePrint() {
    window.print();
  }

  function formatDate(ts: number): string {
    return new Date(ts).toLocaleDateString('en-ZA', {
      year: 'numeric', month: 'short', day: 'numeric',
    });
  }
</script>

<svelte:head>
  <title>Print Summary — Mycelix Observatory</title>
</svelte:head>

<div class="print-page">
  <!-- Print button — hidden when printing -->
  <div class="no-print" style="padding: 12px 0; text-align: right;">
    <button on:click={handlePrint} class="print-btn">
      Print this page
    </button>
    <a href="/" class="back-link">Back to Dashboard</a>
  </div>

  {#if loading}
    <p class="loading-msg">Loading community data...</p>
  {:else if error}
    <p class="error-msg">Error: {error}</p>
  {:else}
    <!-- ================================================================ -->
    <!-- Header                                                           -->
    <!-- ================================================================ -->
    <header class="print-header">
      <h1>Mycelix Resilience Kit — Community Summary</h1>
      <div class="header-meta">
        <span class="community-name">{communityName}</span>
        <span class="generated-at">Generated: {generatedAt}</span>
      </div>
    </header>

    <hr class="section-rule" />

    <!-- ================================================================ -->
    <!-- TEND Balances                                                    -->
    <!-- ================================================================ -->
    <section class="print-section">
      <h2>TEND Balances</h2>
      {#if balance}
        <table class="data-table">
          <tbody>
            <tr><td class="label-cell">Raw Balance</td><td>{balance.balance} hours</td></tr>
            <tr><td class="label-cell">Effective Balance</td><td>{effectiveBalance} hours</td></tr>
            <tr><td class="label-cell">Demurrage Rate</td><td>{demurrageRate}</td></tr>
            <tr><td class="label-cell">Total Earned</td><td>{balance.total_earned} hours</td></tr>
            <tr><td class="label-cell">Total Spent</td><td>{balance.total_spent} hours</td></tr>
            <tr><td class="label-cell">Exchanges</td><td>{balance.exchange_count}</td></tr>
          </tbody>
        </table>
        {#if oracle}
          <p class="oracle-tier">Oracle Tier: <strong>{oracle.tier}</strong> (vitality: {oracle.vitality}%)</p>
        {/if}
      {:else}
        <p class="empty-note">No balance data available.</p>
      {/if}
    </section>

    <!-- ================================================================ -->
    <!-- Food Production                                                  -->
    <!-- ================================================================ -->
    <section class="print-section">
      <h2>Food Production</h2>
      {#if plots.length > 0}
        <p class="section-summary">{plots.length} plot{plots.length !== 1 ? 's' : ''} — Total area: {totalPlotArea} m&sup2;</p>
        <table class="data-table full-width">
          <thead>
            <tr>
              <th>Plot Name</th>
              <th>Location</th>
              <th>Type</th>
              <th class="num-cell">Area (m&sup2;)</th>
            </tr>
          </thead>
          <tbody>
            {#each plots as plot}
              <tr>
                <td>{plot.name}</td>
                <td>{plot.location}</td>
                <td>{plot.plot_type}</td>
                <td class="num-cell">{plot.area_sqm}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else}
        <p class="empty-note">No food plots registered.</p>
      {/if}
    </section>

    <!-- ================================================================ -->
    <!-- Emergency Contacts                                               -->
    <!-- ================================================================ -->
    <section class="print-section">
      <h2>Emergency Contacts</h2>
      {#if emergencyPlan && emergencyPlan.contacts.length > 0}
        <table class="data-table full-width">
          <thead>
            <tr>
              <th>Name</th>
              <th>Phone</th>
              <th>Relationship</th>
            </tr>
          </thead>
          <tbody>
            {#each emergencyPlan.contacts as contact}
              <tr>
                <td>{contact.name}</td>
                <td class="phone-cell">{contact.phone}</td>
                <td>{contact.relationship}</td>
              </tr>
            {/each}
          </tbody>
        </table>
        {#if emergencyPlan.meeting_points.length > 0}
          <div class="meeting-points">
            <strong>Meeting Points:</strong>
            <ol>
              {#each emergencyPlan.meeting_points as point}
                <li>{point}</li>
              {/each}
            </ol>
          </div>
        {/if}
      {:else}
        <p class="empty-note">No emergency plan found. Create one in the Household section.</p>
      {/if}
    </section>

    <!-- ================================================================ -->
    <!-- Water Systems                                                    -->
    <!-- ================================================================ -->
    <section class="print-section">
      <h2>Water Systems</h2>
      {#if waterSystems.length > 0}
        <p class="section-summary">
          {waterSystems.length} system{waterSystems.length !== 1 ? 's' : ''} —
          Total capacity: {totalWaterCapacity.toLocaleString()} litres
        </p>
        <table class="data-table full-width">
          <thead>
            <tr>
              <th>System</th>
              <th>Type</th>
              <th class="num-cell">Capacity (L)</th>
              <th class="num-cell">Efficiency</th>
            </tr>
          </thead>
          <tbody>
            {#each waterSystems as sys}
              <tr>
                <td>{sys.name}</td>
                <td>{sys.system_type}</td>
                <td class="num-cell">{sys.capacity_liters.toLocaleString()}</td>
                <td class="num-cell">{sys.efficiency_percent}%</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else}
        <p class="empty-note">No water systems registered.</p>
      {/if}
    </section>

    <!-- ================================================================ -->
    <!-- Low Stock Alerts                                                 -->
    <!-- ================================================================ -->
    <section class="print-section">
      <h2>Low Stock Alerts</h2>
      {#if lowStock.length > 0}
        <table class="data-table full-width">
          <thead>
            <tr>
              <th>Item</th>
              <th>Category</th>
              <th class="num-cell">Current Stock</th>
              <th class="num-cell">Reorder Point</th>
              <th class="num-cell">Shortfall</th>
            </tr>
          </thead>
          <tbody>
            {#each lowStock as ls}
              <tr class="alert-row">
                <td>{ls.item.name}</td>
                <td>{ls.item.category}</td>
                <td class="num-cell">{ls.total_stock} {ls.item.unit}</td>
                <td class="num-cell">{ls.item.reorder_point} {ls.item.unit}</td>
                <td class="num-cell shortfall">{ls.item.reorder_point - ls.total_stock} {ls.item.unit}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else}
        <p class="empty-note">All supplies above reorder points.</p>
      {/if}
    </section>

    <!-- ================================================================ -->
    <!-- Footer                                                           -->
    <!-- ================================================================ -->
    <footer class="print-footer">
      <hr class="section-rule" />
      {#if communityContact}
        <p class="contact-line">Operator: {communityContact.name}{communityContact.phone ? ` — ${communityContact.phone}` : ''}</p>
      {/if}
      <p>Mycelix Observatory — {communityName} — Printed for offline use</p>
    </footer>
  {/if}
</div>

<style>
  /* ====================================================================
   * Print-friendly summary — light background, dark text, compact layout
   * Designed to fit on 1-2 A4 pages when printed.
   * ==================================================================== */

  .print-page {
    max-width: 800px;
    margin: 0 auto;
    padding: 16px 24px;
    font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-size: 12px;
    line-height: 1.4;
    color: #111;
    background: #fff;
  }

  /* ---- Header ---- */

  .print-header {
    text-align: center;
    margin-bottom: 4px;
  }

  .print-header h1 {
    font-size: 18px;
    font-weight: 700;
    margin: 0 0 4px 0;
    color: #111;
  }

  .header-meta {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #555;
  }

  .community-name {
    font-weight: 600;
  }

  /* ---- Sections ---- */

  .print-section {
    margin-bottom: 12px;
  }

  .print-section h2 {
    font-size: 14px;
    font-weight: 700;
    margin: 0 0 4px 0;
    padding-bottom: 2px;
    border-bottom: 1px solid #ccc;
    color: #222;
  }

  .section-summary {
    margin: 2px 0 4px 0;
    font-size: 11px;
    color: #555;
  }

  .section-rule {
    border: none;
    border-top: 2px solid #333;
    margin: 8px 0;
  }

  /* ---- Tables ---- */

  .data-table {
    border-collapse: collapse;
    font-size: 11px;
  }

  .data-table.full-width {
    width: 100%;
  }

  .data-table th,
  .data-table td {
    padding: 3px 8px;
    text-align: left;
    border-bottom: 1px solid #ddd;
  }

  .data-table th {
    font-weight: 600;
    background: #f5f5f5;
    border-bottom: 2px solid #bbb;
  }

  .data-table .num-cell {
    text-align: right;
  }

  .data-table .label-cell {
    font-weight: 600;
    width: 160px;
    color: #333;
  }

  .phone-cell {
    font-family: 'Courier New', monospace;
    letter-spacing: 0.3px;
  }

  .alert-row {
    background: #fff8f0;
  }

  .shortfall {
    font-weight: 700;
    color: #b91c1c;
  }

  /* ---- Meeting points ---- */

  .meeting-points {
    margin-top: 6px;
    font-size: 11px;
  }

  .meeting-points ol {
    margin: 2px 0 0 20px;
    padding: 0;
  }

  .meeting-points li {
    margin-bottom: 1px;
  }

  /* ---- Oracle tier ---- */

  .oracle-tier {
    margin: 4px 0 0 0;
    font-size: 11px;
    color: #555;
  }

  /* ---- Footer ---- */

  .print-footer {
    margin-top: 8px;
    text-align: center;
    font-size: 10px;
    color: #888;
  }

  .contact-line {
    margin: 0 0 2px 0;
    font-weight: 600;
    color: #555;
  }

  /* ---- Utility ---- */

  .loading-msg {
    text-align: center;
    padding: 40px;
    color: #666;
  }

  .error-msg {
    text-align: center;
    padding: 40px;
    color: #b91c1c;
  }

  .empty-note {
    margin: 2px 0;
    font-size: 11px;
    color: #888;
    font-style: italic;
  }

  /* ---- No-print controls ---- */

  .no-print {
    display: flex;
    gap: 12px;
    justify-content: flex-end;
    align-items: center;
  }

  .print-btn {
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
    color: #fff;
    background: #1e293b;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }

  .print-btn:hover {
    background: #334155;
  }

  .back-link {
    font-size: 12px;
    color: #555;
    text-decoration: underline;
  }

  /* ---- Print media ---- */

  @media print {
    .no-print {
      display: none !important;
    }

    .print-page {
      padding: 0;
      max-width: none;
    }

    .alert-row {
      background: none;
    }

    .shortfall {
      /* Ensure bold stands out without color on B&W printers */
      text-decoration: underline;
    }
  }
</style>
