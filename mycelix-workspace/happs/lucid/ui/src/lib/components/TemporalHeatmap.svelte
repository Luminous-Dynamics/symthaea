<script lang="ts">
  /**
   * Temporal Heatmap Visualization
   *
   * Displays belief evolution over time as a calendar heatmap,
   * showing activity patterns, confidence changes, and epistemic shifts.
   */

  import { onMount } from 'svelte';
  import type { Thought } from '@mycelix/lucid-client';

  // Props
  export let thoughts: Thought[] = [];
  export let days: number = 365;
  export let onDayClick: ((date: Date, thoughts: Thought[]) => void) | undefined = undefined;

  // State
  let container: HTMLElement;
  let heatmapData: Map<string, DayData> = new Map();
  let maxActivity = 1;
  let weeks: Week[] = [];

  interface DayData {
    date: Date;
    count: number;
    thoughts: Thought[];
    avgConfidence: number;
    dominantType: string;
  }

  interface Week {
    days: (DayData | null)[];
  }

  // Colors for heatmap intensity
  const COLORS = {
    empty: 'var(--color-surface-200)',
    level1: '#9be9a8',
    level2: '#40c463',
    level3: '#30a14e',
    level4: '#216e39',
  };

  // Type colors for the mini-indicator
  const TYPE_COLORS: Record<string, string> = {
    belief: '#8b5cf6',
    question: '#f59e0b',
    evidence: '#10b981',
    hypothesis: '#3b82f6',
    note: '#6b7280',
    idea: '#ec4899',
  };

  function getDateKey(date: Date): string {
    return date.toISOString().split('T')[0];
  }

  function processThoughts() {
    heatmapData.clear();
    maxActivity = 1;

    // Group thoughts by date
    for (const thought of thoughts) {
      const date = new Date(thought.created_at);
      const key = getDateKey(date);

      if (!heatmapData.has(key)) {
        heatmapData.set(key, {
          date,
          count: 0,
          thoughts: [],
          avgConfidence: 0,
          dominantType: 'note',
        });
      }

      const day = heatmapData.get(key)!;
      day.count++;
      day.thoughts.push(thought);
      maxActivity = Math.max(maxActivity, day.count);
    }

    // Calculate averages and dominant types
    for (const [_, day] of heatmapData) {
      const totalConfidence = day.thoughts.reduce((sum, t) => sum + (t.confidence || 0.5), 0);
      day.avgConfidence = totalConfidence / day.thoughts.length;

      // Find dominant thought type
      const typeCounts = new Map<string, number>();
      for (const t of day.thoughts) {
        const type = t.thought_type || 'note';
        typeCounts.set(type, (typeCounts.get(type) || 0) + 1);
      }
      let maxCount = 0;
      for (const [type, count] of typeCounts) {
        if (count > maxCount) {
          maxCount = count;
          day.dominantType = type;
        }
      }
    }

    // Build weeks grid
    buildWeeksGrid();
  }

  function buildWeeksGrid() {
    weeks = [];
    const today = new Date();
    const startDate = new Date(today);
    startDate.setDate(startDate.getDate() - days);

    // Align to start of week (Sunday)
    while (startDate.getDay() !== 0) {
      startDate.setDate(startDate.getDate() - 1);
    }

    let currentWeek: (DayData | null)[] = [];

    for (let d = new Date(startDate); d <= today; d.setDate(d.getDate() + 1)) {
      const key = getDateKey(d);
      const dayData = heatmapData.get(key) || null;

      currentWeek.push(
        dayData || {
          date: new Date(d),
          count: 0,
          thoughts: [],
          avgConfidence: 0,
          dominantType: 'note',
        }
      );

      if (currentWeek.length === 7) {
        weeks.push({ days: currentWeek });
        currentWeek = [];
      }
    }

    if (currentWeek.length > 0) {
      // Pad the last week
      while (currentWeek.length < 7) {
        currentWeek.push(null);
      }
      weeks.push({ days: currentWeek });
    }
  }

  function getColorForCount(count: number): string {
    if (count === 0) return COLORS.empty;
    const ratio = count / maxActivity;
    if (ratio <= 0.25) return COLORS.level1;
    if (ratio <= 0.5) return COLORS.level2;
    if (ratio <= 0.75) return COLORS.level3;
    return COLORS.level4;
  }

  function handleDayClick(day: DayData | null) {
    if (day && day.count > 0 && onDayClick) {
      onDayClick(day.date, day.thoughts);
    }
  }

  function formatDate(date: Date): string {
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
  }

  function getMonthLabels(): { label: string; column: number }[] {
    const labels: { label: string; column: number }[] = [];
    let lastMonth = -1;

    for (let i = 0; i < weeks.length; i++) {
      const firstDay = weeks[i].days.find((d) => d !== null);
      if (firstDay) {
        const month = firstDay.date.getMonth();
        if (month !== lastMonth) {
          labels.push({
            label: firstDay.date.toLocaleDateString('en-US', { month: 'short' }),
            column: i,
          });
          lastMonth = month;
        }
      }
    }

    return labels;
  }

  $: if (thoughts) {
    processThoughts();
  }

  onMount(() => {
    processThoughts();
  });
</script>

<div class="temporal-heatmap" bind:this={container}>
  <div class="heatmap-header">
    <h3>Thought Activity</h3>
    <div class="legend">
      <span class="legend-label">Less</span>
      <div class="legend-colors">
        <div class="legend-box" style="background: {COLORS.empty}"></div>
        <div class="legend-box" style="background: {COLORS.level1}"></div>
        <div class="legend-box" style="background: {COLORS.level2}"></div>
        <div class="legend-box" style="background: {COLORS.level3}"></div>
        <div class="legend-box" style="background: {COLORS.level4}"></div>
      </div>
      <span class="legend-label">More</span>
    </div>
  </div>

  <div class="heatmap-container">
    <!-- Month labels -->
    <div class="month-labels">
      {#each getMonthLabels() as { label, column }}
        <span class="month-label" style="grid-column: {column + 2}">
          {label}
        </span>
      {/each}
    </div>

    <!-- Day labels -->
    <div class="day-labels">
      <span></span>
      <span>Mon</span>
      <span></span>
      <span>Wed</span>
      <span></span>
      <span>Fri</span>
      <span></span>
    </div>

    <!-- Heatmap grid -->
    <div class="heatmap-grid" style="grid-template-columns: repeat({weeks.length}, 12px)">
      {#each weeks as week, weekIndex}
        {#each week.days as day, dayIndex}
          {#if day}
            <button
              class="day-cell"
              class:clickable={day.count > 0}
              style="background: {getColorForCount(day.count)}; grid-row: {dayIndex + 1}; grid-column: {weekIndex + 1}"
              on:click={() => handleDayClick(day)}
              title="{formatDate(day.date)}: {day.count} thought{day.count !== 1 ? 's' : ''}"
            >
              {#if day.count > 0}
                <span
                  class="type-indicator"
                  style="background: {TYPE_COLORS[day.dominantType] || TYPE_COLORS.note}"
                ></span>
              {/if}
            </button>
          {:else}
            <div class="day-cell empty" style="grid-row: {dayIndex + 1}; grid-column: {weekIndex + 1}"></div>
          {/if}
        {/each}
      {/each}
    </div>
  </div>

  <!-- Stats summary -->
  <div class="heatmap-stats">
    <div class="stat">
      <span class="stat-value">{thoughts.length}</span>
      <span class="stat-label">Total thoughts</span>
    </div>
    <div class="stat">
      <span class="stat-value">{heatmapData.size}</span>
      <span class="stat-label">Active days</span>
    </div>
    <div class="stat">
      <span class="stat-value">{maxActivity}</span>
      <span class="stat-label">Max/day</span>
    </div>
    <div class="stat">
      <span class="stat-value">
        {(thoughts.length / Math.max(1, heatmapData.size)).toFixed(1)}
      </span>
      <span class="stat-label">Avg/day</span>
    </div>
  </div>
</div>

<style>
  .temporal-heatmap {
    padding: 1rem;
    background: var(--color-surface-100);
    border-radius: 8px;
  }

  .heatmap-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .heatmap-header h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .legend {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }

  .legend-colors {
    display: flex;
    gap: 2px;
  }

  .legend-box {
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }

  .heatmap-container {
    overflow-x: auto;
  }

  .month-labels {
    display: grid;
    grid-template-columns: 30px repeat(53, 12px);
    gap: 2px;
    margin-bottom: 4px;
    font-size: 0.65rem;
    color: var(--color-text-muted);
  }

  .day-labels {
    display: flex;
    flex-direction: column;
    gap: 2px;
    position: absolute;
    font-size: 0.6rem;
    color: var(--color-text-muted);
  }

  .day-labels span {
    height: 12px;
    line-height: 12px;
    width: 24px;
  }

  .heatmap-grid {
    display: grid;
    grid-template-rows: repeat(7, 12px);
    gap: 2px;
    margin-left: 28px;
  }

  .day-cell {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    border: none;
    padding: 0;
    cursor: default;
    position: relative;
    transition: transform 0.1s;
  }

  .day-cell.clickable {
    cursor: pointer;
  }

  .day-cell.clickable:hover {
    transform: scale(1.3);
    z-index: 1;
  }

  .day-cell.empty {
    visibility: hidden;
  }

  .type-indicator {
    position: absolute;
    bottom: 1px;
    right: 1px;
    width: 3px;
    height: 3px;
    border-radius: 50%;
  }

  .heatmap-stats {
    display: flex;
    gap: 1.5rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--color-border);
  }

  .stat {
    display: flex;
    flex-direction: column;
  }

  .stat-value {
    font-size: 1.25rem;
    font-weight: 600;
  }

  .stat-label {
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }
</style>
