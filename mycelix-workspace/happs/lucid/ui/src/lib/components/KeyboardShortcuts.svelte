<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let isOpen = false;

  const dispatch = createEventDispatcher();

  const shortcuts = [
    { category: 'General', items: [
      { keys: ['⌘', 'K'], description: 'Open command palette' },
      { keys: ['⌘', 'N'], description: 'New thought' },
      { keys: ['⌘', '/'], description: 'Focus search' },
      { keys: ['?'], description: 'Show keyboard shortcuts' },
      { keys: ['Esc'], description: 'Close modal / deselect' },
    ]},
    { category: 'Views', items: [
      { keys: ['1'], description: 'Switch to list view' },
      { keys: ['2'], description: 'Switch to graph view' },
      { keys: ['3'], description: 'Switch to split view' },
    ]},
    { category: 'Navigation', items: [
      { keys: ['J'], description: 'Next thought' },
      { keys: ['K'], description: 'Previous thought' },
      { keys: ['Enter'], description: 'Open selected thought' },
      { keys: ['G', 'G'], description: 'Go to top' },
      { keys: ['Shift', 'G'], description: 'Go to bottom' },
    ]},
    { category: 'Graph View', items: [
      { keys: ['+'], description: 'Zoom in' },
      { keys: ['-'], description: 'Zoom out' },
      { keys: ['0'], description: 'Reset zoom' },
      { keys: ['F'], description: 'Fit to screen' },
    ]},
    { category: 'Actions', items: [
      { keys: ['⌘', 'E'], description: 'Export thoughts' },
      { keys: ['⌘', 'I'], description: 'Import thoughts' },
      { keys: ['D'], description: 'Delete selected thought' },
      { keys: ['⌘', 'S'], description: 'Save (if editing)' },
    ]},
  ];

  function close() {
    isOpen = false;
    dispatch('close');
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      close();
    }
  }
</script>

{#if isOpen}
  <div
    class="overlay"
    on:click={close}
    on:keydown={handleKeydown}
    tabindex="-1"
    role="dialog"
  >
    <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
    <div class="modal" on:click|stopPropagation>
      <header>
        <h2>Keyboard Shortcuts</h2>
        <button class="close-btn" on:click={close}>×</button>
      </header>

      <div class="shortcuts-grid">
        {#each shortcuts as section}
          <div class="section">
            <h3>{section.category}</h3>
            <ul>
              {#each section.items as shortcut}
                <li>
                  <div class="keys">
                    {#each shortcut.keys as key, i}
                      {#if i > 0}<span class="plus">+</span>{/if}
                      <kbd>{key}</kbd>
                    {/each}
                  </div>
                  <span class="description">{shortcut.description}</span>
                </li>
              {/each}
            </ul>
          </div>
        {/each}
      </div>

      <footer>
        <p>Tip: Most shortcuts work when focused on the main content area</p>
      </footer>
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
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    width: 100%;
    max-width: 700px;
    max-height: 80vh;
    background: #1e1e2e;
    border: 1px solid #3a3a5e;
    border-radius: 12px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid #2a2a4e;
  }

  h2 {
    margin: 0;
    font-size: 1.1rem;
    color: #fff;
  }

  .close-btn {
    background: none;
    border: none;
    color: #666;
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0;
    line-height: 1;
  }

  .close-btn:hover {
    color: #fff;
  }

  .shortcuts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 24px;
    padding: 20px;
    overflow-y: auto;
  }

  .section h3 {
    margin: 0 0 12px;
    font-size: 0.8rem;
    color: #7c3aed;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  li {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
  }

  .keys {
    display: flex;
    align-items: center;
    gap: 4px;
    min-width: 80px;
  }

  kbd {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 24px;
    height: 24px;
    padding: 0 6px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: inherit;
    color: #e5e5e5;
  }

  .plus {
    color: #555;
    font-size: 0.7rem;
  }

  .description {
    font-size: 0.85rem;
    color: #888;
  }

  footer {
    padding: 12px 20px;
    border-top: 1px solid #2a2a4e;
    background: #1a1a2e;
  }

  footer p {
    margin: 0;
    font-size: 0.75rem;
    color: #555;
  }
</style>
