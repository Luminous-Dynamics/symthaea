<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    getAllInventoryItems,
    getLowStockItems,
    addInventoryItem,
    updateStockLevel,
    type InventoryItem,
    type LowStockItem,
  } from '$lib/resilience-client';
  import { toasts } from '$lib/toast';
  import { exportInventoryCsv } from '$lib/data-export';
  import { createFreshness } from '$lib/freshness';
  import FreshnessBar from '$lib/components/FreshnessBar.svelte';

  let items: InventoryItem[] = [];
  let lowStock: LowStockItem[] = [];
  let loading = true;
  let error = '';
  let filterCategory = '';

  // Add Item form state
  let showAddForm = false;
  let addSubmitting = false;
  let addError = '';
  let newItem = resetNewItem();

  function resetNewItem() {
    return {
      sku: '',
      name: '',
      description: '' as string | null,
      category: 'Food',
      unit: '',
      reorder_point: 10,
      reorder_quantity: 50,
      created_at: Date.now(),
    };
  }

  // Update Stock form state — keyed by item id
  let stockFormOpen: Record<string, boolean> = {};
  let stockQuantity: Record<string, number> = {};
  let stockNotes: Record<string, string> = {};
  let stockSubmitting: Record<string, boolean> = {};
  let stockError: Record<string, string> = {};

  const CATEGORIES = ['Food', 'Water', 'Medical', 'Fuel', 'Hygiene', 'Shelter'];

  // Freshness — 2min polling
  async function fetchData() {
    [items, lowStock] = await Promise.all([
      getAllInventoryItems(),
      getLowStockItems(),
    ]);
  }

  const freshness = createFreshness(fetchData, 120_000);
  const { lastUpdated, loadError, refreshing, startPolling, stopPolling, refresh } = freshness;

  onMount(async () => {
    await refresh();
    loading = false;
    startPolling();
  });

  onDestroy(() => stopPolling());

  $: categories = [...new Set(items.map(i => i.category))].sort();

  $: filteredItems = filterCategory
    ? items.filter(i => i.category === filterCategory)
    : items;

  $: lowStockIds = new Set(lowStock.map(ls => ls.item.id));

  function categoryColor(cat: string): string {
    const colors: Record<string, string> = {
      Food: 'bg-green-900/50 text-green-300',
      Water: 'bg-cyan-900/50 text-cyan-300',
      Medical: 'bg-red-900/50 text-red-300',
      Fuel: 'bg-orange-900/50 text-orange-300',
      Hygiene: 'bg-blue-900/50 text-blue-300',
      Shelter: 'bg-yellow-900/50 text-yellow-300',
    };
    return colors[cat] ?? 'bg-gray-800 text-gray-300';
  }

  function lowStockInfo(itemId: string): LowStockItem | undefined {
    return lowStock.find(ls => ls.item.id === itemId);
  }

  async function handleAddItem() {
    addError = '';
    addSubmitting = true;
    try {
      const created = await addInventoryItem({
        ...newItem,
        description: newItem.description || undefined,
      });
      items = [created, ...items];
      newItem = resetNewItem();
      showAddForm = false;
      toasts.success('Item added');
    } catch (e) {
      addError = e instanceof Error ? e.message : 'Failed to add item';
      toasts.error(addError);
    } finally {
      addSubmitting = false;
    }
  }

  function toggleStockForm(itemId: string) {
    stockFormOpen[itemId] = !stockFormOpen[itemId];
    if (stockFormOpen[itemId]) {
      stockQuantity[itemId] = stockQuantity[itemId] ?? 0;
      stockNotes[itemId] = stockNotes[itemId] ?? '';
      stockError[itemId] = '';
    }
  }

  async function handleUpdateStock(itemId: string) {
    stockError[itemId] = '';
    stockSubmitting[itemId] = true;
    try {
      await updateStockLevel(itemId, stockQuantity[itemId], stockNotes[itemId]);
      // Refresh low stock data after update
      lowStock = await getLowStockItems();
      stockFormOpen[itemId] = false;
      stockQuantity[itemId] = 0;
      stockNotes[itemId] = '';
      toasts.success('Stock updated');
    } catch (e) {
      stockError[itemId] = e instanceof Error ? e.message : 'Failed to update stock';
      toasts.error(stockError[itemId]);
    } finally {
      stockSubmitting[itemId] = false;
    }
  }
</script>

<div class="min-h-screen bg-gray-950 text-gray-100 p-6">
  <div class="container mx-auto max-w-6xl">
    <FreshnessBar {lastUpdated} {loadError} {refreshing} {refresh} />
    <div class="flex items-center justify-between mb-1">
      <h1 class="text-2xl font-bold">Community Supplies</h1>
      <button
        class="px-4 py-2 rounded-lg text-sm font-semibold transition-colors bg-amber-700 hover:bg-amber-600 text-white"
        on:click={() => { showAddForm = !showAddForm; addError = ''; }}
        aria-label={showAddForm ? 'Cancel adding new item' : 'Add new inventory item'}
      >
        {showAddForm ? 'Cancel' : '+ Add Item'}
      </button>
    </div>
    <p class="text-gray-400 mb-6">Emergency inventory tracking — food, water, medical, fuel, and shelter supplies.</p>

    {#if loading}
      <div class="text-gray-400">Loading supply data...</div>
    {:else if error}
      <div class="bg-red-900/30 border border-red-500 rounded p-4 text-red-200">{error}</div>
    {:else}

      <!-- Add Item Form -->
      {#if showAddForm}
        <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
          <h2 class="text-lg font-semibold mb-4 text-amber-400">New Inventory Item</h2>
          {#if addError}
            <div class="bg-red-900/30 border border-red-500 rounded p-3 text-red-200 text-sm mb-4">{addError}</div>
          {/if}
          <form on:submit|preventDefault={handleAddItem} class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label class="block text-sm text-gray-400 mb-1" for="add-name">Name</label>
              <input
                id="add-name"
                type="text"
                required
                bind:value={newItem.name}
                class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-gray-100 focus:border-amber-500 focus:outline-none"
                placeholder="e.g. Bottled Water 500ml"
              />
            </div>
            <div>
              <label class="block text-sm text-gray-400 mb-1" for="add-sku">SKU</label>
              <input
                id="add-sku"
                type="text"
                required
                bind:value={newItem.sku}
                class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-gray-100 focus:border-amber-500 focus:outline-none"
                placeholder="e.g. WTR-500"
              />
            </div>
            <div class="sm:col-span-2">
              <label class="block text-sm text-gray-400 mb-1" for="add-desc">Description</label>
              <input
                id="add-desc"
                type="text"
                bind:value={newItem.description}
                class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-gray-100 focus:border-amber-500 focus:outline-none"
                placeholder="Optional description"
              />
            </div>
            <div>
              <label class="block text-sm text-gray-400 mb-1" for="add-category">Category</label>
              <select
                id="add-category"
                bind:value={newItem.category}
                class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-gray-100 focus:border-amber-500 focus:outline-none"
              >
                {#each CATEGORIES as cat}
                  <option value={cat}>{cat}</option>
                {/each}
              </select>
            </div>
            <div>
              <label class="block text-sm text-gray-400 mb-1" for="add-unit">Unit</label>
              <input
                id="add-unit"
                type="text"
                required
                bind:value={newItem.unit}
                class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-gray-100 focus:border-amber-500 focus:outline-none"
                placeholder="e.g. bottles, kg, packs"
              />
            </div>
            <div>
              <label class="block text-sm text-gray-400 mb-1" for="add-reorder-point">Reorder Point</label>
              <input
                id="add-reorder-point"
                type="number"
                required
                min="0"
                bind:value={newItem.reorder_point}
                class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-gray-100 focus:border-amber-500 focus:outline-none"
              />
            </div>
            <div>
              <label class="block text-sm text-gray-400 mb-1" for="add-reorder-qty">Reorder Quantity</label>
              <input
                id="add-reorder-qty"
                type="number"
                required
                min="1"
                bind:value={newItem.reorder_quantity}
                class="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-gray-100 focus:border-amber-500 focus:outline-none"
              />
            </div>
            <div class="sm:col-span-2 flex justify-end">
              <button
                type="submit"
                disabled={addSubmitting}
                class="px-6 py-2 rounded-lg text-sm font-semibold transition-colors bg-amber-700 hover:bg-amber-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Submit new inventory item"
              >
                {addSubmitting ? 'Adding...' : 'Add Item'}
              </button>
            </div>
          </form>
        </div>
      {/if}

      <!-- Low Stock Alerts -->
      {#if lowStock.length > 0}
        <div class="mb-6">
          <h2 class="text-lg font-semibold mb-3 text-yellow-400">Low Stock Alerts</h2>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {#each lowStock as ls}
              <div class="bg-yellow-900/20 border border-yellow-700 rounded-lg p-4">
                <div class="flex items-center justify-between">
                  <span class="font-semibold text-yellow-200">{ls.item.name}</span>
                  <span class="text-yellow-400 font-bold">{ls.total_stock} / {ls.item.reorder_point}</span>
                </div>
                <p class="text-sm text-yellow-300/70 mt-1">
                  Below reorder point — need {ls.item.reorder_quantity} {ls.item.unit}s
                </p>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Summary -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold text-amber-400">{items.length}</div>
          <div class="text-sm text-gray-400">Item Types</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold text-amber-400">{categories.length}</div>
          <div class="text-sm text-gray-400">Categories</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold {lowStock.length > 0 ? 'text-yellow-400' : 'text-green-400'}">{lowStock.length}</div>
          <div class="text-sm text-gray-400">Low Stock Items</div>
        </div>
      </div>

      <!-- Category Filter -->
      <div class="flex flex-wrap gap-2 mb-6">
        <button
          class="px-3 py-1 rounded-full text-sm transition-colors {!filterCategory ? 'bg-amber-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
          on:click={() => filterCategory = ''}
          aria-label="Show all categories"
        >All</button>
        {#each categories as cat}
          <button
            class="px-3 py-1 rounded-full text-sm transition-colors {filterCategory === cat ? 'bg-amber-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
            on:click={() => filterCategory = cat}
            aria-label="Filter by {cat}"
          >{cat}</button>
        {/each}
      </div>

      <!-- Inventory List -->
      <div class="space-y-3">
        {#each filteredItems as item (item.id)}
          <div class="bg-gray-900 rounded-lg border p-4 {lowStockIds.has(item.id) ? 'border-yellow-700' : 'border-gray-800'}">
            <div class="flex items-start justify-between">
              <div>
                <div class="flex items-center gap-2">
                  <h3 class="font-semibold text-white">{item.name}</h3>
                  {#if lowStockIds.has(item.id)}
                    <span class="text-xs px-2 py-0.5 rounded-full bg-yellow-900/50 text-yellow-300">LOW</span>
                  {/if}
                </div>
                {#if item.description}
                  <p class="text-sm text-gray-400 mt-1">{item.description}</p>
                {/if}
              </div>
              <div class="flex items-center gap-2">
                <button
                  class="px-3 py-1 rounded text-xs font-medium transition-colors bg-amber-800/60 hover:bg-amber-700 text-amber-200"
                  on:click={() => toggleStockForm(item.id)}
                  aria-label={stockFormOpen[item.id] ? `Cancel stock update for ${item.name}` : `Update stock for ${item.name}`}
                >
                  {stockFormOpen[item.id] ? 'Cancel' : 'Update Stock'}
                </button>
                <span class="text-xs px-2 py-1 rounded-full {categoryColor(item.category)}">{item.category}</span>
              </div>
            </div>
            <div class="flex gap-4 mt-2 text-xs text-gray-500">
              <span>SKU: {item.sku}</span>
              <span>Unit: {item.unit}</span>
              <span>Reorder at: {item.reorder_point}</span>
              <span>Reorder qty: {item.reorder_quantity}</span>
            </div>

            <!-- Inline Stock Update Form -->
            {#if stockFormOpen[item.id]}
              <div class="mt-3 pt-3 border-t border-gray-700">
                {#if stockError[item.id]}
                  <div class="bg-red-900/30 border border-red-500 rounded p-2 text-red-200 text-xs mb-3">{stockError[item.id]}</div>
                {/if}
                <form on:submit|preventDefault={() => handleUpdateStock(item.id)} class="flex flex-col sm:flex-row flex-wrap items-stretch sm:items-end gap-2 sm:gap-3">
                  <div>
                    <label class="block text-xs text-gray-400 mb-1" for="stock-qty-{item.id}">Quantity</label>
                    <input
                      id="stock-qty-{item.id}"
                      type="number"
                      required
                      min="0"
                      bind:value={stockQuantity[item.id]}
                      class="w-28 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-gray-100 focus:border-amber-500 focus:outline-none"
                    />
                  </div>
                  <div class="flex-1 min-w-[200px]">
                    <label class="block text-xs text-gray-400 mb-1" for="stock-notes-{item.id}">Notes</label>
                    <input
                      id="stock-notes-{item.id}"
                      type="text"
                      bind:value={stockNotes[item.id]}
                      class="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-gray-100 focus:border-amber-500 focus:outline-none"
                      placeholder="e.g. Monthly restock delivery"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={stockSubmitting[item.id]}
                    class="px-4 py-1 rounded text-sm font-semibold transition-colors bg-amber-700 hover:bg-amber-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                    aria-label="Save stock update for {item.name}"
                  >
                    {stockSubmitting[item.id] ? 'Saving...' : 'Save'}
                  </button>
                </form>
              </div>
            {/if}
          </div>
        {/each}
      </div>

      {#if filteredItems.length === 0}
        <div class="text-center text-gray-500 py-12">
          <p class="text-lg">No items in this category.</p>
        </div>
      {/if}

      <div class="mt-6 flex justify-end">
        <button on:click={() => exportInventoryCsv(items)}
          class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-300 transition-colors"
          aria-label="Export inventory data as CSV file">
          Export Inventory CSV
        </button>
      </div>
    {/if}
  </div>
</div>
