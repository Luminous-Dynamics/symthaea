<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { initHolochainClient } from '$lib/holochain/client';
  import { createDispute } from '$lib/holochain/disputes';
  import { notifications, isConnected } from '$lib/stores';
  import ConnectionNotice from '$lib/components/ConnectionNotice.svelte';
  import { uploadFiles } from '$lib/ipfs/ipfsClient';
  import type { CreateDisputeInput, DisputeReason } from '$types';

  // URL parameters
  let transactionHash = '';
  let listingTitle = '';
  let sellerName = '';

  // Form state
  let reason: DisputeReason = 'not_as_described';
  let description = '';
  let evidenceFiles: File[] = [];
  let evidencePreviews: string[] = [];

  // UI state
  let submitting = false;
  let uploadingEvidence = false;

  // Dispute reasons for dropdown
  const reasons: { value: DisputeReason; label: string }[] = [
    { value: 'not_as_described', label: 'Item Not As Described' },
    { value: 'defective_product', label: 'Defective or Broken Product' },
    { value: 'wrong_item', label: 'Wrong Item Received' },
    { value: 'non_delivery', label: 'Item Not Delivered' },
    { value: 'damaged_in_shipping', label: 'Damaged During Shipping' },
    { value: 'seller_unresponsive', label: 'Seller Not Responding' },
    { value: 'other', label: 'Other Issue' },
  ];

  /**
   * Load URL parameters
   */
  onMount(() => {
    transactionHash = $page.url.searchParams.get('transaction') || '';
    listingTitle = $page.url.searchParams.get('title') || 'Product';
    sellerName = $page.url.searchParams.get('seller') || 'Seller';

    if (!transactionHash) {
      notifications.error('Invalid Request', 'Missing transaction information');
      setTimeout(() => {
        goto('/transactions');
      }, 2000);
    }
  });

  /**
   * Handle file selection
   */
  function handleFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    const files = input.files;

    if (!files || files.length === 0) return;

    // Convert FileList to array and add to existing evidence
    const newFiles = Array.from(files);
    evidenceFiles = [...evidenceFiles, ...newFiles];

    // Generate previews
    newFiles.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        evidencePreviews = [...evidencePreviews, e.target?.result as string];
      };
      reader.readAsDataURL(file);
    });

    // Reset input
    input.value = '';
  }

  /**
   * Remove evidence at index
   */
  function removeEvidence(index: number) {
    evidenceFiles = evidenceFiles.filter((_, i) => i !== index);
    evidencePreviews = evidencePreviews.filter((_, i) => i !== index);
  }

  /**
   * Validate form
   */
  function validateForm(): boolean {
    if (!description.trim()) {
      notifications.error('Description Required', 'Please describe the issue');
      return false;
    }

    if (description.trim().length < 20) {
      notifications.error('Description Too Short', 'Please write at least 20 characters');
      return false;
    }

    if (evidenceFiles.length === 0) {
      notifications.warning('No Evidence', 'Consider uploading photos or documents to support your claim');
      // Don't block submission, just warn
    }

    return true;
  }

  /**
   * Submit dispute
   */
  async function handleSubmit() {
    if (!validateForm()) return;

    submitting = true;

    try {
      // Upload evidence to IPFS
      let evidenceCids: string[] = [];
      if (evidenceFiles.length > 0) {
        notifications.info('Uploading Evidence', 'Uploading evidence to IPFS...');
        uploadingEvidence = true;
        try {
          evidenceCids = await uploadFiles(evidenceFiles);
        } finally {
          uploadingEvidence = false;
        }
      }

      // Connect to Holochain
      const client = await initHolochainClient();

      // Create dispute input
      const disputeInput: CreateDisputeInput = {
        transaction_hash: transactionHash,
        reason,
        description: description.trim(),
        evidence_ipfs_cids: evidenceCids,
      };

      // File dispute
      notifications.info('Filing Dispute', 'Submitting dispute to MRC...');
      await createDispute(client, disputeInput);

      notifications.success(
        'Dispute Filed',
        'Your dispute has been submitted to the Multi-Resonance Council for review'
      );

      // Redirect to transactions page
      setTimeout(() => {
        goto('/transactions');
      }, 2000);
    } catch (e: any) {
      console.error('Failed to file dispute:', e);
      notifications.error('Submission Failed', e.message || 'Failed to file dispute');
    } finally {
      uploadingEvidence = false;
      submitting = false;
    }
  }

  /**
   * Cancel and go back
   */
  function handleCancel() {
    goto('/transactions');
  }
</script>

<main class="file-dispute">
  <div class="container">
    <!-- Header -->
    <header class="page-header">
      <h1>File a Dispute</h1>
      <p class="subtitle">Submit your issue to the Multi-Resonance Council (MRC)</p>
    </header>

    <ConnectionNotice />

    <!-- Form -->
    <form on:submit|preventDefault={handleSubmit} class="dispute-form">
      <!-- Transaction Context -->
      <div class="context-card">
        <div class="context-label">Disputing Transaction</div>
        <div class="context-title">{listingTitle}</div>
        <div class="context-seller">Sold by {sellerName}</div>
      </div>

      <!-- What is MRC? -->
      <div class="info-box">
        <h3>About the Multi-Resonance Council</h3>
        <p>
          The MRC is a decentralized panel of trusted arbitrators who review disputes and make fair
          decisions based on evidence. Your case will be reviewed by multiple arbitrators, weighted
          by their Proof of Generalized Quality (PoGQ) scores.
        </p>
      </div>

      <!-- Reason -->
      <div class="form-group">
        <label for="reason-input">
          Dispute Reason <span class="required">*</span>
        </label>
        <select id="reason-input" bind:value={reason} required>
          {#each reasons as reasonOption}
            <option value={reasonOption.value}>{reasonOption.label}</option>
          {/each}
        </select>
      </div>

      <!-- Description -->
      <div class="form-group">
        <label for="description-input">
          Detailed Description <span class="required">*</span>
        </label>
        <textarea
          id="description-input"
          bind:value={description}
          placeholder="Please provide a detailed explanation of the issue. Include what you expected, what you received, and any communication with the seller..."
          rows="8"
          maxlength="2000"
          required
        />
        <small class="help-text">{description.length}/2000 characters (minimum 20)</small>
      </div>

      <!-- Evidence Upload -->
      <div class="form-group">
        <label for="evidence-input">Supporting Evidence</label>

        <!-- Evidence Upload Button -->
        <div class="evidence-upload-area">
          <input
            id="evidence-input"
            type="file"
            accept="image/*,application/pdf"
            multiple
            on:change={handleFileSelect}
            style="display: none;"
          />
          <button
            type="button"
            class="btn btn-upload"
            on:click={() => document.getElementById('evidence-input')?.click()}
            disabled={evidenceFiles.length >= 10}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Upload Evidence ({evidenceFiles.length}/10)
          </button>
        </div>

        <!-- Evidence Previews -->
        {#if evidencePreviews.length > 0}
          <div class="evidence-preview-grid">
            {#each evidencePreviews as preview, i}
              <div class="evidence-preview-item">
                <img src={preview} alt="Evidence {i + 1}" />
                <button
                  type="button"
                  class="remove-evidence-btn"
                  on:click={() => removeEvidence(i)}
                  title="Remove evidence"
                >
                  ×
                </button>
              </div>
            {/each}
          </div>
        {/if}

        <small class="help-text">
          Upload photos showing the issue, screenshots of communication, or any other relevant
          documents. Maximum 10 files.
        </small>
      </div>

      <!-- Guidelines -->
      <div class="guidelines">
        <h3>Filing Guidelines</h3>
        <ul>
          <li>Only file disputes for legitimate issues with your transaction</li>
          <li>Provide clear evidence to support your claim</li>
          <li>Be honest and thorough in your description</li>
          <li>False or malicious disputes may affect your trust score</li>
          <li>The MRC decision is final and binding</li>
        </ul>
      </div>

      <!-- Form Actions -->
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" on:click={handleCancel} disabled={submitting}>
          Cancel
        </button>
        <button
          type="submit"
          class="btn btn-primary"
          disabled={submitting || uploadingEvidence || !$isConnected}
          title={$isConnected ? '' : 'Connect to Holochain to file a dispute'}
        >
          {#if submitting}
            Filing Dispute...
          {:else if uploadingEvidence}
            Uploading Evidence...
          {:else}
            Submit to MRC
          {/if}
        </button>
      </div>
    </form>
  </div>
</main>

<style>
  .file-dispute {
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 1rem;
  }

  .container {
    max-width: 800px;
    margin: 0 auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    padding: 2rem;
  }

  .page-header {
    margin-bottom: 2rem;
    text-align: center;
  }

  .page-header h1 {
    font-size: 2rem;
    color: #2d3748;
    margin: 0 0 0.5rem 0;
  }

  .subtitle {
    color: #718096;
    font-size: 1rem;
    margin: 0;
  }

  .dispute-form {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  /* Context Card */
  .context-card {
    background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
    padding: 1.5rem;
    border-radius: 8px;
    color: white;
    text-align: center;
  }

  .context-label {
    font-size: 0.875rem;
    opacity: 0.9;
    margin-bottom: 0.5rem;
  }

  .context-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }

  .context-seller {
    font-size: 0.95rem;
    opacity: 0.9;
  }

  /* Info Box */
  .info-box {
    background: #ebf8ff;
    border-left: 4px solid #4299e1;
    padding: 1.5rem;
    border-radius: 8px;
  }

  .info-box h3 {
    font-size: 1rem;
    color: #2d3748;
    margin: 0 0 0.75rem 0;
  }

  .info-box p {
    color: #4a5568;
    margin: 0;
    line-height: 1.6;
  }

  /* Form Group */
  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  label {
    font-weight: 600;
    color: #2d3748;
    font-size: 0.95rem;
  }

  .required {
    color: #e53e3e;
  }

  select,
  textarea {
    padding: 0.75rem;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    font-size: 1rem;
    font-family: inherit;
    transition: border-color 0.2s;
  }

  select:focus,
  textarea:focus {
    outline: none;
    border-color: #667eea;
  }

  textarea {
    resize: vertical;
    min-height: 150px;
  }

  .help-text {
    color: #718096;
    font-size: 0.875rem;
  }

  /* Evidence Upload */
  .evidence-upload-area {
    display: flex;
    justify-content: center;
    padding: 1rem;
    border: 2px dashed #cbd5e0;
    border-radius: 8px;
    background: #f7fafc;
  }

  .btn-upload {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: white;
    border: 2px solid #e53e3e;
    color: #e53e3e;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-upload:hover:not(:disabled) {
    background: #e53e3e;
    color: white;
  }

  .btn-upload:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .evidence-preview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }

  .evidence-preview-item {
    position: relative;
    aspect-ratio: 1;
    border-radius: 8px;
    overflow: hidden;
    border: 2px solid #e2e8f0;
  }

  .evidence-preview-item img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .remove-evidence-btn {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.7);
    color: white;
    border: none;
    font-size: 1.5rem;
    line-height: 1;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.2s;
  }

  .remove-evidence-btn:hover {
    background: rgba(229, 62, 62, 0.9);
  }

  /* Guidelines */
  .guidelines {
    background: #fffaf0;
    border-left: 4px solid #ed8936;
    padding: 1.5rem;
    border-radius: 8px;
  }

  .guidelines h3 {
    font-size: 1rem;
    color: #2d3748;
    margin: 0 0 1rem 0;
  }

  .guidelines ul {
    margin: 0;
    padding-left: 1.5rem;
    color: #4a5568;
  }

  .guidelines li {
    margin-bottom: 0.5rem;
  }

  .guidelines li:last-child {
    margin-bottom: 0;
  }

  /* Form Actions */
  .form-actions {
    display: flex;
    gap: 1rem;
    justify-content: flex-end;
    padding-top: 1rem;
    border-top: 2px solid #e2e8f0;
  }

  .btn {
    padding: 0.875rem 2rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
  }

  .btn-primary {
    background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(229, 62, 62, 0.4);
  }

  .btn-secondary {
    background: white;
    color: #2d3748;
    border: 2px solid #e2e8f0;
  }

  .btn-secondary:hover:not(:disabled) {
    background: #f7fafc;
  }

  .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  /* Responsive */
  @media (max-width: 768px) {
    .container {
      padding: 1.5rem;
    }

    .evidence-preview-grid {
      grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    }

    .form-actions {
      flex-direction: column;
    }

    .btn {
      width: 100%;
    }
  }
</style>
