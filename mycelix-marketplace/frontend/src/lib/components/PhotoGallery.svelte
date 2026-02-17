<script lang="ts">
  /**
   * Photo Gallery Component
   *
   * Displays a grid of photos from IPFS CIDs
   */

  export let cids: string[] = [];
  export let alt: string = 'Photo';
  export let layout: 'grid' | 'carousel' = 'grid';
  export let maxHeight: string = '400px';

  let selectedIndex = 0;

  function getIpfsUrl(cid: string): string {
    return `https://ipfs.io/ipfs/${cid}`;
  }

  function selectImage(index: number) {
    selectedIndex = index;
  }
</script>

<div class="photo-gallery" class:grid-layout={layout === 'grid'}>
  {#if cids.length === 0}
    <div class="no-photos">
      <span>📷</span>
      <p>No photos available</p>
    </div>
  {:else if layout === 'grid'}
    <div class="photo-grid" style="max-height: {maxHeight}">
      {#each cids as cid, index}
        <div class="photo-item">
          <img src={getIpfsUrl(cid)} alt="{alt} {index + 1}" />
        </div>
      {/each}
    </div>
  {:else}
    <!-- Carousel layout -->
    <div class="photo-carousel">
      <div class="main-photo" style="max-height: {maxHeight}">
        <img src={getIpfsUrl(cids[selectedIndex])} alt="{alt} {selectedIndex + 1}" />
      </div>

      {#if cids.length > 1}
        <div class="thumbnails">
          {#each cids as cid, index}
            <button
              class="thumbnail"
              class:active={index === selectedIndex}
              on:click={() => selectImage(index)}
            >
              <img src={getIpfsUrl(cid)} alt="{alt} thumbnail {index + 1}" />
            </button>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .photo-gallery {
    width: 100%;
  }

  .no-photos {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    background: var(--gray-100);
    border-radius: var(--radius-md);
    color: var(--gray-500);
  }

  .no-photos span {
    font-size: 3rem;
    margin-bottom: 0.5rem;
  }

  /* Grid Layout */
  .photo-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 0.5rem;
    overflow-y: auto;
    padding: 0.5rem;
  }

  .photo-item {
    position: relative;
    width: 100%;
    padding-bottom: 100%; /* Square aspect ratio */
    background: var(--gray-200);
    border-radius: var(--radius-sm);
    overflow: hidden;
  }

  .photo-item img {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  /* Carousel Layout */
  .photo-carousel {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .main-photo {
    width: 100%;
    background: var(--gray-200);
    border-radius: var(--radius-md);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .main-photo img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  .thumbnails {
    display: flex;
    gap: 0.5rem;
    overflow-x: auto;
    padding: 0.25rem;
  }

  .thumbnail {
    flex-shrink: 0;
    width: 60px;
    height: 60px;
    border: 2px solid transparent;
    border-radius: var(--radius-sm);
    overflow: hidden;
    cursor: pointer;
    background: var(--gray-200);
    padding: 0;
    transition: all 0.2s;
  }

  .thumbnail:hover {
    border-color: var(--primary);
    transform: scale(1.05);
  }

  .thumbnail.active {
    border-color: var(--primary);
    box-shadow: 0 0 0 2px var(--primary-dark);
  }

  .thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  /* Scrollbar styling */
  .photo-grid::-webkit-scrollbar,
  .thumbnails::-webkit-scrollbar {
    height: 6px;
    width: 6px;
  }

  .photo-grid::-webkit-scrollbar-track,
  .thumbnails::-webkit-scrollbar-track {
    background: var(--gray-200);
    border-radius: 3px;
  }

  .photo-grid::-webkit-scrollbar-thumb,
  .thumbnails::-webkit-scrollbar-thumb {
    background: var(--gray-400);
    border-radius: 3px;
  }

  .photo-grid::-webkit-scrollbar-thumb:hover,
  .thumbnails::-webkit-scrollbar-thumb:hover {
    background: var(--gray-500);
  }

  /* Responsive */
  @media (max-width: 768px) {
    .photo-grid {
      grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    }

    .thumbnail {
      width: 50px;
      height: 50px;
    }
  }
</style>
