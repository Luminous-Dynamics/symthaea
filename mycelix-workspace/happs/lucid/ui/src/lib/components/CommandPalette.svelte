<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import { thoughts, searchQuery, selectedThought } from '../stores/thoughts';
  import { isConnected } from '../stores/holochain';

  export let isOpen = false;

  const dispatch = createEventDispatcher();

  let inputRef: HTMLInputElement;
  let query = '';
  let selectedIndex = 0;

  interface Command {
    id: string;
    label: string;
    shortcut?: string;
    icon: string;
    action: () => void;
    category: 'action' | 'navigation' | 'thought' | 'view';
  }

  // Static commands
  const staticCommands: Command[] = [
    {
      id: 'new-thought',
      label: 'New Thought',
      shortcut: 'n',
      icon: '✨',
      action: () => dispatch('command', { type: 'new-thought' }),
      category: 'action',
    },
    {
      id: 'search',
      label: 'Search Thoughts',
      shortcut: '/',
      icon: '🔍',
      action: () => dispatch('command', { type: 'focus-search' }),
      category: 'action',
    },
    {
      id: 'view-list',
      label: 'List View',
      shortcut: '1',
      icon: '📋',
      action: () => dispatch('command', { type: 'view', value: 'list' }),
      category: 'view',
    },
    {
      id: 'view-graph',
      label: 'Graph View',
      shortcut: '2',
      icon: '🕸️',
      action: () => dispatch('command', { type: 'view', value: 'graph' }),
      category: 'view',
    },
    {
      id: 'view-split',
      label: 'Split View',
      shortcut: '3',
      icon: '📊',
      action: () => dispatch('command', { type: 'view', value: 'split' }),
      category: 'view',
    },
    {
      id: 'toggle-theme',
      label: 'Toggle Theme',
      icon: '🌓',
      action: () => dispatch('command', { type: 'toggle-theme' }),
      category: 'action',
    },
    {
      id: 'export',
      label: 'Export Thoughts',
      icon: '📤',
      action: () => dispatch('command', { type: 'export' }),
      category: 'action',
    },
    {
      id: 'import',
      label: 'Import Thoughts',
      icon: '📥',
      action: () => dispatch('command', { type: 'import' }),
      category: 'action',
    },
    {
      id: 'help',
      label: 'Keyboard Shortcuts',
      shortcut: '?',
      icon: '⌨️',
      action: () => dispatch('command', { type: 'show-shortcuts' }),
      category: 'navigation',
    },
  ];

  // Dynamic commands based on thoughts
  $: thoughtCommands = $thoughts.slice(0, 10).map((t): Command => ({
    id: `thought-${t.id}`,
    label: t.content.slice(0, 60) + (t.content.length > 60 ? '...' : ''),
    icon: getThoughtIcon(t.thought_type),
    action: () => {
      selectedThought.set(t);
      dispatch('command', { type: 'select-thought', value: t });
    },
    category: 'thought',
  }));

  // Filter commands based on query
  $: allCommands = [...staticCommands, ...thoughtCommands];
  $: filteredCommands = query
    ? allCommands.filter((cmd) =>
        cmd.label.toLowerCase().includes(query.toLowerCase())
      )
    : staticCommands;

  // Group by category
  $: groupedCommands = filteredCommands.reduce((groups, cmd) => {
    if (!groups[cmd.category]) groups[cmd.category] = [];
    groups[cmd.category].push(cmd);
    return groups;
  }, {} as Record<string, Command[]>);

  function getThoughtIcon(type: string): string {
    const icons: Record<string, string> = {
      Claim: '💭',
      Question: '❓',
      Observation: '👁️',
      Belief: '🙏',
      Hypothesis: '🔬',
      Definition: '📖',
      Argument: '⚔️',
      Evidence: '📊',
      Intuition: '✨',
      Memory: '🧠',
      Goal: '🎯',
      Plan: '📝',
      Reflection: '🪞',
      Quote: '💬',
      Note: '📌',
    };
    return icons[type] || '💡';
  }

  const categoryLabels: Record<string, string> = {
    action: 'Actions',
    navigation: 'Navigation',
    thought: 'Thoughts',
    view: 'Views',
  };

  function executeCommand(command: Command) {
    command.action();
    close();
  }

  function close() {
    isOpen = false;
    query = '';
    selectedIndex = 0;
    dispatch('close');
  }

  function handleKeydown(event: KeyboardEvent) {
    if (!isOpen) {
      // Open palette with Cmd+K or Ctrl+K
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault();
        isOpen = true;
        return;
      }
      return;
    }

    switch (event.key) {
      case 'Escape':
        event.preventDefault();
        close();
        break;

      case 'ArrowDown':
        event.preventDefault();
        selectedIndex = Math.min(selectedIndex + 1, filteredCommands.length - 1);
        break;

      case 'ArrowUp':
        event.preventDefault();
        selectedIndex = Math.max(selectedIndex - 1, 0);
        break;

      case 'Enter':
        event.preventDefault();
        if (filteredCommands[selectedIndex]) {
          executeCommand(filteredCommands[selectedIndex]);
        }
        break;
    }
  }

  // Reset selection when query changes
  $: if (query) selectedIndex = 0;

  // Focus input when opened
  $: if (isOpen && inputRef) {
    setTimeout(() => inputRef?.focus(), 10);
  }

  onMount(() => {
    window.addEventListener('keydown', handleKeydown);
  });

  onDestroy(() => {
    window.removeEventListener('keydown', handleKeydown);
  });
</script>

{#if isOpen}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="overlay" role="presentation" on:click={close} on:keydown={(e) => e.key === 'Escape' && close()}>
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div class="palette" role="dialog" aria-modal="true" aria-label="Command palette" tabindex="-1" on:click|stopPropagation>
      <div class="search-box">
        <span class="search-icon">⌘</span>
        <input
          bind:this={inputRef}
          bind:value={query}
          type="text"
          placeholder="Type a command or search..."
          spellcheck="false"
          aria-label="Search commands"
          aria-autocomplete="list"
          aria-controls="command-list"
          aria-activedescendant={filteredCommands[selectedIndex]?.id || ''}
        />
        <kbd class="esc-hint">ESC</kbd>
      </div>

      <div class="commands" role="listbox" aria-label="Available commands">
        {#each Object.entries(groupedCommands) as [category, commands]}
          <div class="category" role="group" aria-label={categoryLabels[category]}>
            <div class="category-label" id="category-{category}">{categoryLabels[category]}</div>
            {#each commands as command, i}
              {@const globalIndex = filteredCommands.indexOf(command)}
              <button
                class="command"
                class:selected={globalIndex === selectedIndex}
                on:click={() => executeCommand(command)}
                on:mouseenter={() => (selectedIndex = globalIndex)}
                role="option"
                aria-selected={globalIndex === selectedIndex}
                aria-label="{command.label}{command.shortcut ? `, shortcut ${command.shortcut}` : ''}"
              >
                <span class="icon" aria-hidden="true">{command.icon}</span>
                <span class="label">{command.label}</span>
                {#if command.shortcut}
                  <kbd class="shortcut" aria-hidden="true">{command.shortcut}</kbd>
                {/if}
              </button>
            {/each}
          </div>
        {/each}

        {#if filteredCommands.length === 0}
          <div class="no-results">
            <span>No commands found for "{query}"</span>
          </div>
        {/if}
      </div>

      <div class="footer">
        <span><kbd>↑↓</kbd> navigate</span>
        <span><kbd>↵</kbd> select</span>
        <span><kbd>esc</kbd> close</span>
      </div>
    </div>
  </div>
{/if}

<style>
  .overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding-top: 15vh;
    z-index: 1000;
  }

  .palette {
    width: 100%;
    max-width: 560px;
    background: #1e1e2e;
    border: 1px solid #3a3a5e;
    border-radius: 12px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    overflow: hidden;
  }

  .search-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    border-bottom: 1px solid #2a2a4e;
  }

  .search-icon {
    font-size: 1.2rem;
    color: #7c3aed;
  }

  input {
    flex: 1;
    background: none;
    border: none;
    color: #e5e5e5;
    font-size: 1rem;
    outline: none;
  }

  input::placeholder {
    color: #666;
  }

  .esc-hint {
    padding: 2px 6px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 4px;
    font-size: 0.7rem;
    color: #666;
  }

  .commands {
    max-height: 400px;
    overflow-y: auto;
  }

  .category {
    padding: 8px 0;
  }

  .category-label {
    padding: 4px 16px;
    font-size: 0.7rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .command {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    padding: 10px 16px;
    background: none;
    border: none;
    color: #e5e5e5;
    font-size: 0.9rem;
    text-align: left;
    cursor: pointer;
    transition: background 0.1s;
  }

  .command:hover,
  .command.selected {
    background: #252540;
  }

  .command.selected {
    background: #7c3aed22;
  }

  .command .icon {
    font-size: 1.1rem;
  }

  .command .label {
    flex: 1;
  }

  .command .shortcut {
    padding: 2px 6px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 4px;
    font-size: 0.7rem;
    color: #888;
  }

  .no-results {
    padding: 24px 16px;
    text-align: center;
    color: #666;
  }

  .footer {
    display: flex;
    gap: 16px;
    padding: 12px 16px;
    border-top: 1px solid #2a2a4e;
    font-size: 0.75rem;
    color: #555;
  }

  .footer kbd {
    padding: 1px 4px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 3px;
    font-size: 0.7rem;
  }
</style>
