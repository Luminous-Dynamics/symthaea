<script lang="ts">
  /**
   * LoadingSkeleton - Animated placeholder for loading states
   *
   * Usage:
   * <LoadingSkeleton variant="text" />
   * <LoadingSkeleton variant="circle" size={40} />
   * <LoadingSkeleton variant="rect" width="100%" height={200} />
   */

  export let variant: 'text' | 'circle' | 'rect' | 'card' = 'text';
  export let width: string | number = '100%';
  export let height: string | number = variant === 'text' ? 16 : 100;
  export let size: number | undefined = undefined; // For circle variant
  export let lines: number = 1; // For text variant
  export let animated: boolean = true;

  $: computedWidth = typeof width === 'number' ? `${width}px` : width;
  $: computedHeight = typeof height === 'number' ? `${height}px` : height;
  $: circleSize = size ? `${size}px` : '40px';
</script>

{#if variant === 'text'}
  <div class="skeleton-text" class:animated>
    {#each Array(lines) as _, i}
      <div
        class="skeleton-line"
        style="width: {i === lines - 1 && lines > 1 ? '70%' : computedWidth}; height: {computedHeight}"
      ></div>
    {/each}
  </div>
{:else if variant === 'circle'}
  <div
    class="skeleton-circle"
    class:animated
    style="width: {circleSize}; height: {circleSize}"
  ></div>
{:else if variant === 'rect'}
  <div
    class="skeleton-rect"
    class:animated
    style="width: {computedWidth}; height: {computedHeight}"
  ></div>
{:else if variant === 'card'}
  <div class="skeleton-card" class:animated>
    <div class="skeleton-card-header">
      <div class="skeleton-circle" style="width: 40px; height: 40px"></div>
      <div class="skeleton-card-title">
        <div class="skeleton-line" style="width: 60%; height: 14px"></div>
        <div class="skeleton-line" style="width: 40%; height: 12px"></div>
      </div>
    </div>
    <div class="skeleton-card-body">
      <div class="skeleton-line" style="width: 100%; height: 12px"></div>
      <div class="skeleton-line" style="width: 100%; height: 12px"></div>
      <div class="skeleton-line" style="width: 80%; height: 12px"></div>
    </div>
  </div>
{/if}

<style>
  .skeleton-line,
  .skeleton-circle,
  .skeleton-rect,
  .skeleton-card {
    background: linear-gradient(
      90deg,
      var(--bg-tertiary, #0f172a) 25%,
      var(--bg-secondary, #1e293b) 50%,
      var(--bg-tertiary, #0f172a) 75%
    );
    background-size: 200% 100%;
    border-radius: 4px;
  }

  .animated {
    animation: shimmer 1.5s infinite;
  }

  @keyframes shimmer {
    0% {
      background-position: 200% 0;
    }
    100% {
      background-position: -200% 0;
    }
  }

  .skeleton-text {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .skeleton-line {
    border-radius: 4px;
  }

  .skeleton-circle {
    border-radius: 50%;
    flex-shrink: 0;
  }

  .skeleton-rect {
    border-radius: 8px;
  }

  .skeleton-card {
    padding: 16px;
    border-radius: 12px;
    background: var(--bg-secondary, #1e293b);
  }

  .skeleton-card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }

  .skeleton-card-title {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .skeleton-card-body {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .skeleton-card .skeleton-line,
  .skeleton-card .skeleton-circle {
    background: linear-gradient(
      90deg,
      var(--bg-tertiary, #0f172a) 25%,
      rgba(148, 163, 184, 0.1) 50%,
      var(--bg-tertiary, #0f172a) 75%
    );
    background-size: 200% 100%;
  }

  .skeleton-card.animated .skeleton-line,
  .skeleton-card.animated .skeleton-circle {
    animation: shimmer 1.5s infinite;
  }
</style>
