<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { get } from 'svelte/store';
  import { initHolochainClient } from '$lib/holochain/client';
  import { createListing } from '$lib/holochain/listings';
  import { notifications, isConnected } from '$lib/stores';
  import { uploadFiles } from '$lib/ipfs/ipfsClient';
  import { userAgentId } from '$lib/stores/auth';
  import { submitListingClaim } from '$lib/knowledge/market-knowledge-bridge';
  import ConnectionNotice from '$lib/components/ConnectionNotice.svelte';
  import type { CreateListingInput, ListingCategory } from '$types';

  // Form state
  let title = '';
  let description = '';
  let price = 0;
  let category: ListingCategory = 'Other';
  let quantityAvailable = 1;
  let photoFiles: File[] = [];
  let photoPreviews: string[] = [];

  // UI state
  let submitting = false;
  let uploadingPhotos = false;

  // Categories for dropdown
  const categories: ListingCategory[] = [
    'Electronics',
    'Fashion',
    'Home & Garden',
    'Sports & Outdoors',
    'Books & Media',
    'Toys & Games',
    'Health & Beauty',
    'Automotive',
    'Art & Collectibles',
    'Other',
  ];

  /**
   * Handle file selection
   */
  function handleFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    const files = input.files;

    if (!files || files.length === 0) return;

    // Convert FileList to array and add to existing photos
    const newFiles = Array.from(files);
    photoFiles = [...photoFiles, ...newFiles];

    // Generate previews
    newFiles.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        photoPreviews = [...photoPreviews, e.target?.result as string];
      };
      reader.readAsDataURL(file);
    });

    // Reset input
    input.value = '';
  }

  /**
   * Remove photo at index
   */
  function removePhoto(index: number) {
    photoFiles = photoFiles.filter((_, i) => i !== index);
    photoPreviews = photoPreviews.filter((_, i) => i !== index);
  }

  /**
   * Validate form
   */
  function validateForm(): boolean {
    if (!title.trim()) {
      notifications.error('Validation Error', 'Title is required');
      return false;
    }

    if (title.length < 5) {
      notifications.error('Validation Error', 'Title must be at least 5 characters');
      return false;
    }

    if (!description.trim()) {
      notifications.error('Validation Error', 'Description is required');
      return false;
    }

    if (description.length < 20) {
      notifications.error('Validation Error', 'Description must be at least 20 characters');
      return false;
    }

    if (price <= 0) {
      notifications.error('Validation Error', 'Price must be greater than $0');
      return false;
    }

    if (price > 1000000) {
      notifications.error('Validation Error', 'Price cannot exceed $1,000,000');
      return false;
    }

    if (quantityAvailable < 1) {
      notifications.error('Validation Error', 'Quantity must be at least 1');
      return false;
    }

    if (photoFiles.length === 0) {
      notifications.error('Validation Error', 'At least one photo is required');
      return false;
    }

    if (photoFiles.length > 10) {
      notifications.error('Validation Error', 'Maximum 10 photos allowed');
      return false;
    }

    return true;
  }

  /**
   * Submit form
   */
  async function handleSubmit() {
    if (!validateForm()) return;

    submitting = true;

    try {
      // Upload photos to IPFS
      notifications.info('Uploading Photos', 'Uploading photos to IPFS...');
      uploadingPhotos = true;
      const photoCids = await uploadFiles(photoFiles);
      uploadingPhotos = false;

      // Connect to Holochain
      const client = await initHolochainClient();

      // Create listing input
      const listingInput: CreateListingInput = {
        title: title.trim(),
        description: description.trim(),
        price,
        category,
        photos_ipfs_cids: photoCids,
        quantity_available: quantityAvailable,
      };

      // Create listing
      notifications.info('Creating Listing', 'Creating listing on blockchain...');
      const createdListing = await createListing(client, listingInput);

      // Fire-and-forget: register claim in Mycelix Knowledge
      const sellerId = get(userAgentId);
      if (sellerId) {
        submitListingClaim(client, {
          listing: createdListing,
          sellerAgentId: sellerId,
        });
      }

      // Success!
      notifications.success('Listing Created', 'Your listing has been published!');

      // Redirect to listing detail page
      setTimeout(() => {
        goto(`/listing/${createdListing.listing_hash || createdListing.id}`);
      }, 1500);
    } catch (e: any) {
      console.error('Failed to create listing:', e);
      notifications.error('Creation Failed', e.message || 'Failed to create listing');
    } finally {
      uploadingPhotos = false;
      submitting = false;
    }
  }

  /**
   * Cancel and go back
   */
  function handleCancel() {
    goto('/dashboard');
  }

  onMount(() => {
    // Focus title input on mount
    const titleInput = document.getElementById('title-input') as HTMLInputElement;
    titleInput?.focus();
  });
</script>

<main class="create-listing">
  <div class="container">
    <!-- Header -->
    <header class="page-header">
      <h1>Create New Listing</h1>
      <p class="subtitle">List your item for sale on the marketplace</p>
    </header>

    <ConnectionNotice />

    <!-- Form -->
    <form on:submit|preventDefault={handleSubmit} class="listing-form">
      <!-- Title -->
      <div class="form-group">
        <label for="title-input">
          Title <span class="required">*</span>
        </label>
        <input
          id="title-input"
          type="text"
          bind:value={title}
          placeholder="E.g., iPhone 13 Pro - Like New"
          maxlength="100"
          required
        />
        <small class="help-text">{title.length}/100 characters</small>
      </div>

      <!-- Description -->
      <div class="form-group">
        <label for="description-input">
          Description <span class="required">*</span>
        </label>
        <textarea
          id="description-input"
          bind:value={description}
          placeholder="Provide a detailed description of your item, including condition, features, and any defects..."
          rows="6"
          maxlength="2000"
          required
        />
        <small class="help-text">{description.length}/2000 characters</small>
      </div>

      <!-- Price & Quantity Row -->
      <div class="form-row">
        <!-- Price -->
        <div class="form-group">
          <label for="price-input">
            Price <span class="required">*</span>
          </label>
          <div class="input-with-prefix">
            <span class="prefix">$</span>
            <input
              id="price-input"
              type="number"
              bind:value={price}
              min="0.01"
              step="0.01"
              placeholder="0.00"
              required
            />
          </div>
        </div>

        <!-- Quantity -->
        <div class="form-group">
          <label for="quantity-input">Quantity Available</label>
          <input
            id="quantity-input"
            type="number"
            bind:value={quantityAvailable}
            min="1"
            max="10000"
            required
          />
        </div>
      </div>

      <!-- Category -->
      <div class="form-group">
        <label for="category-input">
          Category <span class="required">*</span>
        </label>
        <select id="category-input" bind:value={category} required>
          {#each categories as cat}
            <option value={cat}>{cat}</option>
          {/each}
        </select>
      </div>

      <!-- Photos -->
      <div class="form-group">
        <label for="photos-input">
          Photos <span class="required">*</span>
        </label>

        <!-- Photo Upload Button -->
        <div class="photo-upload-area">
          <input
            id="photos-input"
            type="file"
            accept="image/*"
            multiple
            on:change={handleFileSelect}
            style="display: none;"
          />
          <button
            type="button"
            class="btn btn-upload"
            on:click={() => document.getElementById('photos-input')?.click()}
            disabled={photoFiles.length >= 10}
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
            Upload Photos ({photoFiles.length}/10)
          </button>
        </div>

        <!-- Photo Previews -->
        {#if photoPreviews.length > 0}
          <div class="photo-preview-grid">
            {#each photoPreviews as preview, i}
              <div class="photo-preview-item">
                <img src={preview} alt="Preview {i + 1}" />
                <button
                  type="button"
                  class="remove-photo-btn"
                  on:click={() => removePhoto(i)}
                  title="Remove photo"
                >
                  ×
                </button>
                {#if i === 0}
                  <span class="main-photo-badge">Main</span>
                {/if}
              </div>
            {/each}
          </div>
        {/if}

        <small class="help-text">
          First photo will be the main listing image. Maximum 10 photos.
        </small>
      </div>

      <!-- Form Actions -->
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" on:click={handleCancel} disabled={submitting}>
          Cancel
        </button>
        <button
          type="submit"
          class="btn btn-primary"
          disabled={submitting || uploadingPhotos || !$isConnected}
          title={$isConnected ? '' : 'Connect to Holochain to publish'}
        >
          {#if submitting}
            Creating Listing...
          {:else if uploadingPhotos}
            Uploading Photos...
          {:else}
            Create Listing
          {/if}
        </button>
      </div>
    </form>
  </div>
</main>

<style>
  .create-listing {
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

  .listing-form {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }

  label {
    font-weight: 600;
    color: #2d3748;
    font-size: 0.95rem;
  }

  .required {
    color: #e53e3e;
  }

  input[type='text'],
  input[type='number'],
  textarea,
  select {
    padding: 0.75rem;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    font-size: 1rem;
    font-family: inherit;
    transition: border-color 0.2s;
  }

  input:focus,
  textarea:focus,
  select:focus {
    outline: none;
    border-color: #667eea;
  }

  textarea {
    resize: vertical;
    min-height: 120px;
  }

  .input-with-prefix {
    position: relative;
    display: flex;
    align-items: center;
  }

  .prefix {
    position: absolute;
    left: 0.75rem;
    font-weight: 600;
    color: #718096;
    pointer-events: none;
  }

  .input-with-prefix input {
    padding-left: 2rem;
    width: 100%;
  }

  .help-text {
    color: #718096;
    font-size: 0.875rem;
  }

  /* Photo Upload */
  .photo-upload-area {
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
    border: 2px solid #667eea;
    color: #667eea;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-upload:hover:not(:disabled) {
    background: #667eea;
    color: white;
  }

  .btn-upload:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .photo-preview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }

  .photo-preview-item {
    position: relative;
    aspect-ratio: 1;
    border-radius: 8px;
    overflow: hidden;
    border: 2px solid #e2e8f0;
  }

  .photo-preview-item img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .remove-photo-btn {
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

  .remove-photo-btn:hover {
    background: rgba(229, 62, 62, 0.9);
  }

  .main-photo-badge {
    position: absolute;
    bottom: 0.5rem;
    left: 0.5rem;
    background: rgba(102, 126, 234, 0.9);
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
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
    padding: 0.75rem 2rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
  }

  .btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
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

    .form-row {
      grid-template-columns: 1fr;
      gap: 1rem;
    }

    .photo-preview-grid {
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
