<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { thoughts, createThought } from '../stores/thoughts';
  import {
    exportAsJSON,
    exportAsMarkdown,
    exportAsJSONLD,
    downloadExport,
    autoImport,
    generateBookmarklet,
    type ImportResult,
  } from '../services/import-export';

  export let isOpen = false;

  const dispatch = createEventDispatcher();

  let activeTab: 'export' | 'import' | 'clipper' = 'export';
  let importFile: FileList | null = null;
  let importResult: ImportResult | null = null;
  let isImporting = false;
  let importProgress = 0;

  // Export
  function handleExportJSON() {
    const content = exportAsJSON($thoughts);
    const date = new Date().toISOString().split('T')[0];
    downloadExport(content, `lucid-export-${date}.json`, 'application/json');
  }

  function handleExportMarkdown() {
    const content = exportAsMarkdown($thoughts);
    const date = new Date().toISOString().split('T')[0];
    downloadExport(content, `lucid-export-${date}.md`, 'text/markdown');
  }

  function handleExportJSONLD() {
    const content = exportAsJSONLD($thoughts);
    const date = new Date().toISOString().split('T')[0];
    downloadExport(content, `lucid-export-${date}.jsonld`, 'application/ld+json');
  }

  // Import
  async function handleFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;

    const file = input.files[0];
    const content = await file.text();
    importResult = autoImport(content, file.name);
  }

  async function handleConfirmImport() {
    if (!importResult?.thoughts.length) return;

    isImporting = true;
    importProgress = 0;

    const total = importResult.thoughts.length;

    for (let i = 0; i < total; i++) {
      await createThought(importResult.thoughts[i] as any);
      importProgress = ((i + 1) / total) * 100;
      // Small delay to avoid overwhelming the system
      await new Promise((r) => setTimeout(r, 50));
    }

    isImporting = false;
    importResult = null;
    dispatch('imported', { count: total });
    close();
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    const files = event.dataTransfer?.files;
    if (files?.length) {
      const file = files[0];
      file.text().then((content) => {
        importResult = autoImport(content, file.name);
      });
    }
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
  }

  // Clipper
  const bookmarklet = generateBookmarklet('http://localhost:8888/api');

  function close() {
    isOpen = false;
    importResult = null;
    dispatch('close');
  }
</script>

{#if isOpen}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="overlay" role="presentation" on:click={close} on:keydown={(e) => e.key === 'Escape' && close()}>
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div class="modal" on:click|stopPropagation role="dialog" aria-modal="true" tabindex="-1" aria-label="Import and Export dialog">
      <header>
        <h2>Import / Export</h2>
        <button class="close-btn" on:click={close}>×</button>
      </header>

      <nav class="tabs">
        <button class:active={activeTab === 'export'} on:click={() => (activeTab = 'export')}>
          Export
        </button>
        <button class:active={activeTab === 'import'} on:click={() => (activeTab = 'import')}>
          Import
        </button>
        <button class:active={activeTab === 'clipper'} on:click={() => (activeTab = 'clipper')}>
          Web Clipper
        </button>
      </nav>

      <div class="content">
        {#if activeTab === 'export'}
          <div class="export-section">
            <p class="description">
              Export your {$thoughts.length} thoughts in various formats.
            </p>

            <div class="export-options">
              <button class="export-btn" on:click={handleExportJSON}>
                <span class="icon">📄</span>
                <span class="format">JSON</span>
                <span class="desc">Full data, can be re-imported</span>
              </button>

              <button class="export-btn" on:click={handleExportMarkdown}>
                <span class="icon">📝</span>
                <span class="format">Markdown</span>
                <span class="desc">Human-readable, organized by domain</span>
              </button>

              <button class="export-btn" on:click={handleExportJSONLD}>
                <span class="icon">🔗</span>
                <span class="format">JSON-LD</span>
                <span class="desc">Linked Data for semantic web</span>
              </button>
            </div>
          </div>
        {:else if activeTab === 'import'}
          <div class="import-section">
            {#if !importResult}
              <div
                class="drop-zone"
                on:drop={handleDrop}
                on:dragover={handleDragOver}
                role="button"
                tabindex="0"
              >
                <span class="icon">📥</span>
                <p>Drop a file here or click to select</p>
                <span class="formats">
                  Supports: JSON, Markdown, Obsidian, Roam Research
                </span>
                <input type="file" accept=".json,.md,.txt" on:change={handleFileSelect} />
              </div>
            {:else}
              <div class="import-preview">
                <h3>Import Preview</h3>

                {#if importResult.errors.length > 0}
                  <div class="messages errors">
                    <h4>Errors:</h4>
                    <ul>
                      {#each importResult.errors as error}
                        <li>{error}</li>
                      {/each}
                    </ul>
                  </div>
                {/if}

                {#if importResult.warnings.length > 0}
                  <div class="messages warnings">
                    <h4>Warnings:</h4>
                    <ul>
                      {#each importResult.warnings as warning}
                        <li>{warning}</li>
                      {/each}
                    </ul>
                  </div>
                {/if}

                <div class="preview-summary">
                  <span class="count">{importResult.thoughts.length}</span>
                  <span class="label">thoughts will be imported</span>
                </div>

                {#if importResult.thoughts.length > 0}
                  <div class="preview-list">
                    {#each importResult.thoughts.slice(0, 5) as thought}
                      <div class="preview-item">
                        <span class="type">{thought.thought_type}</span>
                        <span class="text">{thought.content?.slice(0, 60)}...</span>
                      </div>
                    {/each}
                    {#if importResult.thoughts.length > 5}
                      <div class="preview-more">
                        +{importResult.thoughts.length - 5} more
                      </div>
                    {/if}
                  </div>
                {/if}

                {#if isImporting}
                  <div class="progress-bar">
                    <div class="progress-fill" style="width: {importProgress}%"></div>
                  </div>
                  <p class="progress-text">Importing... {Math.round(importProgress)}%</p>
                {:else}
                  <div class="import-actions">
                    <button class="primary" on:click={handleConfirmImport} disabled={importResult.thoughts.length === 0}>
                      Import {importResult.thoughts.length} Thoughts
                    </button>
                    <button class="secondary" on:click={() => (importResult = null)}>
                      Cancel
                    </button>
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        {:else if activeTab === 'clipper'}
          <div class="clipper-section">
            <h3>Web Clipper Bookmarklet</h3>
            <p>
              Drag this button to your bookmarks bar to clip content from any webpage:
            </p>

            <a class="bookmarklet" href={bookmarklet} on:click|preventDefault>
              📎 Clip to LUCID
            </a>

            <div class="instructions">
              <h4>How to use:</h4>
              <ol>
                <li>Drag the button above to your bookmarks bar</li>
                <li>Select text on any webpage</li>
                <li>Click the bookmarklet to save it to LUCID</li>
              </ol>
            </div>

            <div class="manual-clip">
              <h4>Manual Clip</h4>
              <p>Paste content or a URL to create a thought:</p>
              <textarea placeholder="Paste content here..."></textarea>
              <button>Create Thought</button>
            </div>
          </div>
        {/if}
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
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    width: 100%;
    max-width: 600px;
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
  }

  .tabs {
    display: flex;
    border-bottom: 1px solid #2a2a4e;
  }

  .tabs button {
    flex: 1;
    padding: 12px;
    background: none;
    border: none;
    color: #888;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
    border-bottom: 2px solid transparent;
  }

  .tabs button:hover {
    color: #aaa;
  }

  .tabs button.active {
    color: #7c3aed;
    border-bottom-color: #7c3aed;
  }

  .content {
    padding: 20px;
    overflow-y: auto;
  }

  .description {
    color: #888;
    margin-bottom: 20px;
  }

  .export-options {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .export-btn {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
  }

  .export-btn:hover {
    border-color: #5a5a8e;
    background: #2a2a50;
  }

  .export-btn .icon {
    font-size: 1.5rem;
  }

  .export-btn .format {
    font-size: 1rem;
    color: #fff;
    font-weight: 600;
  }

  .export-btn .desc {
    font-size: 0.8rem;
    color: #888;
    margin-left: auto;
  }

  /* Import */
  .drop-zone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    background: #252540;
    border: 2px dashed #3a3a5e;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }

  .drop-zone:hover {
    border-color: #7c3aed;
    background: rgba(124, 58, 237, 0.05);
  }

  .drop-zone .icon {
    font-size: 2.5rem;
    margin-bottom: 12px;
  }

  .drop-zone p {
    margin: 0 0 8px;
    color: #e5e5e5;
  }

  .drop-zone .formats {
    font-size: 0.8rem;
    color: #666;
  }

  .drop-zone input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }

  .import-preview h3 {
    margin: 0 0 16px;
    color: #fff;
  }

  .messages {
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 16px;
  }

  .messages h4 {
    margin: 0 0 8px;
    font-size: 0.85rem;
  }

  .messages ul {
    margin: 0;
    padding-left: 20px;
    font-size: 0.85rem;
  }

  .messages.errors {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid #ef4444;
    color: #fca5a5;
  }

  .messages.warnings {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid #f59e0b;
    color: #fcd34d;
  }

  .preview-summary {
    text-align: center;
    padding: 20px;
    background: rgba(124, 58, 237, 0.1);
    border-radius: 8px;
    margin-bottom: 16px;
  }

  .preview-summary .count {
    display: block;
    font-size: 2rem;
    font-weight: 700;
    color: #7c3aed;
  }

  .preview-summary .label {
    color: #888;
  }

  .preview-list {
    margin-bottom: 16px;
  }

  .preview-item {
    display: flex;
    gap: 8px;
    padding: 8px;
    background: #252540;
    border-radius: 4px;
    margin-bottom: 4px;
  }

  .preview-item .type {
    padding: 2px 6px;
    background: #3a3a5e;
    border-radius: 3px;
    font-size: 0.7rem;
    color: #888;
  }

  .preview-item .text {
    font-size: 0.85rem;
    color: #a5a5c5;
  }

  .preview-more {
    text-align: center;
    font-size: 0.8rem;
    color: #666;
    padding: 8px;
  }

  .progress-bar {
    height: 4px;
    background: #252540;
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .progress-fill {
    height: 100%;
    background: #7c3aed;
    transition: width 0.2s;
  }

  .progress-text {
    text-align: center;
    color: #888;
    font-size: 0.85rem;
  }

  .import-actions {
    display: flex;
    gap: 12px;
  }

  .import-actions button {
    flex: 1;
    padding: 12px;
    border-radius: 6px;
    font-size: 0.9rem;
    cursor: pointer;
  }

  .import-actions .primary {
    background: #7c3aed;
    border: none;
    color: white;
  }

  .import-actions .primary:disabled {
    background: #4a4a6e;
    cursor: not-allowed;
  }

  .import-actions .secondary {
    background: none;
    border: 1px solid #3a3a5e;
    color: #888;
  }

  /* Clipper */
  .clipper-section h3 {
    margin: 0 0 12px;
    color: #fff;
  }

  .clipper-section > p {
    color: #888;
    margin-bottom: 16px;
  }

  .bookmarklet {
    display: inline-block;
    padding: 12px 24px;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    border-radius: 8px;
    color: white;
    font-weight: 600;
    text-decoration: none;
    cursor: grab;
    margin-bottom: 20px;
  }

  .instructions {
    background: #252540;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
  }

  .instructions h4 {
    margin: 0 0 12px;
    color: #fff;
    font-size: 0.9rem;
  }

  .instructions ol {
    margin: 0;
    padding-left: 20px;
    color: #888;
    font-size: 0.85rem;
  }

  .instructions li {
    margin-bottom: 8px;
  }

  .manual-clip h4 {
    margin: 0 0 8px;
    color: #fff;
    font-size: 0.9rem;
  }

  .manual-clip p {
    color: #888;
    font-size: 0.85rem;
    margin-bottom: 12px;
  }

  .manual-clip textarea {
    width: 100%;
    height: 100px;
    padding: 12px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 6px;
    color: #e5e5e5;
    font-family: inherit;
    resize: vertical;
    margin-bottom: 12px;
  }

  .manual-clip button {
    padding: 10px 20px;
    background: #7c3aed;
    border: none;
    border-radius: 6px;
    color: white;
    cursor: pointer;
  }
</style>
