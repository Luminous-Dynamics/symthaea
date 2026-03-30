<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  /**
   * EmptyState - Display when no data is available
   *
   * Usage:
   * <EmptyState
   *   icon="search"
   *   title="No results found"
   *   description="Try adjusting your search terms"
   * />
   */

  export let icon: 'search' | 'inbox' | 'chart' | 'users' | 'lightbulb' | 'shield' | 'star' = 'inbox';
  export let title: string;
  export let description: string = '';
  export let actionLabel: string | undefined = undefined;
  export let onAction: (() => void) | undefined = undefined;

  const icons = {
    search: 'Search icon',
    inbox: 'Inbox icon',
    chart: 'Chart icon',
    users: 'Users icon',
    lightbulb: 'Lightbulb icon',
    shield: 'Shield icon',
    star: 'Star icon',
  };
</script>

<div class="empty-state">
  <div class="empty-icon">
    <span class="icon-placeholder">{icons[icon]}</span>
  </div>
  <h3 class="empty-title">{title}</h3>
  {#if description}
    <p class="empty-description">{description}</p>
  {/if}
  {#if actionLabel && onAction}
    <button class="empty-action" on:click={onAction}>
      {actionLabel}
    </button>
  {/if}
</div>

<style>
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40px 20px;
    min-height: 200px;
  }

  .empty-icon {
    width: 64px;
    height: 64px;
    border-radius: 16px;
    background: var(--bg-tertiary, #0f172a);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
  }

  .icon-placeholder {
    font-size: 12px;
    color: var(--text-muted, #94a3b8);
    opacity: 0.5;
  }

  .empty-title {
    margin: 0 0 8px 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary, #f8fafc);
  }

  .empty-description {
    margin: 0 0 20px 0;
    font-size: 13px;
    color: var(--text-muted, #94a3b8);
    max-width: 300px;
    line-height: 1.5;
  }

  .empty-action {
    padding: 10px 20px;
    background: var(--color-primary, #3b82f6);
    border: none;
    border-radius: 8px;
    color: white;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }

  .empty-action:hover {
    background: var(--color-primary-hover, #2563eb);
  }
</style>
