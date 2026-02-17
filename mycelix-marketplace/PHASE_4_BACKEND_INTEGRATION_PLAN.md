# 🔗 Phase 4: Backend Integration Plan

**Created**: November 11, 2025
**Status**: 📋 Planning Complete - Ready to Execute
**Estimated Duration**: 2-3 weeks
**Prerequisites**: ✅ Phase 3 Frontend Complete (100%)

---

## 🎯 Objective

Connect all 8 frontend pages to the Holochain backend by replacing TODO placeholders with real zome calls, implementing real-time updates, adding proper state management, and beginning automated testing.

---

## 📊 Current State Analysis

### What's Complete ✅
- **8 frontend pages**: All UI components built and tested with demo data
- **IPFS integration**: PhotoGallery and PhotoUpload working
- **Design system**: Consistent UI/UX across all pages
- **TODO placeholders**: Clear integration points in every file
- **Demo data**: Rich test data for all scenarios

### What's Needed 🚧
- **Holochain client**: Connection to DHT via zome calls
- **Real-time updates**: WebSocket for live data
- **State management**: Svelte stores for global state
- **Error handling**: Network failure scenarios
- **Authentication**: User session management
- **Testing**: Automated E2E and unit tests

---

## 🏗️ Integration Architecture

### Holochain Connection Pattern
```typescript
// Create Holochain client (once at app startup)
import { AppWebsocket } from '@holochain/client';

const client = await AppWebsocket.connect('ws://localhost:8888');
const appInfo = await client.appInfo();

// Zome call pattern (used throughout)
const result = await client.callZome({
  cap_secret: null,
  cell_id: [appInfo.cell_data[0].cell_id[0], appInfo.cell_data[0].cell_id[1]],
  zome_name: 'listings', // or 'transactions', 'disputes', etc.
  fn_name: 'get_listing',
  payload: { listing_hash: 'uhC0k...' },
});
```

### State Management Pattern
```typescript
// stores/auth.ts - User authentication
import { writable } from 'svelte/store';

export const currentUser = writable(null);
export const isAuthenticated = writable(false);
export const userProfile = writable(null);

// stores/cart.ts - Shopping cart (migrate from localStorage)
export const cartItems = writable([]);

// stores/holochain.ts - Connection status
export const holochainClient = writable(null);
export const connectionStatus = writable('disconnected');
```

### Real-Time Updates Pattern
```typescript
// Signal subscription for live updates
client.on('signal', (signal) => {
  if (signal.zome_name === 'transactions') {
    // Handle transaction status updates
    updateTransactionStatus(signal.payload);
  } else if (signal.zome_name === 'disputes') {
    // Handle dispute updates
    updateDisputeStatus(signal.payload);
  }
});
```

---

## 📋 Implementation Roadmap

### Week 1: Core Integration (8 pages)

#### Day 1-2: Setup & ListingDetail
**Files**: `frontend/src/lib/holochain/client.ts`, `ListingDetail.svelte`

**Tasks**:
1. Create Holochain client wrapper:
   ```bash
   npm install @holochain/client
   ```

2. Implement connection logic:
   ```typescript
   // lib/holochain/client.ts
   export async function initHolochainClient() {
     const client = await AppWebsocket.connect('ws://localhost:8888');
     return client;
   }

   export async function getListing(client, listingHash) {
     return await client.callZome({
       zome_name: 'listings',
       fn_name: 'get_listing',
       payload: { listing_hash: listingHash },
     });
   }
   ```

3. Replace TODO in ListingDetail.svelte:
   ```typescript
   // Replace demo data loading
   onMount(async () => {
     loading = true;
     try {
       const client = await initHolochainClient();
       const result = await getListing(client, listingHash);

       listing = result.listing;
       seller = result.seller;
       reviews = result.reviews || [];

       // Check IPFS availability (keep existing code)
       if (listing.photos_ipfs_cids.length > 0) {
         photosAvailable = await isAvailable(listing.photos_ipfs_cids[0]);
       }
     } catch (e) {
       error = e.message;
     } finally {
       loading = false;
     }
   });
   ```

4. Test with real Holochain data
5. Handle edge cases (network errors, missing data)

**Acceptance Criteria**:
- ✅ Can fetch listing from DHT
- ✅ IPFS photos load correctly
- ✅ Seller info displays with trust score
- ✅ Reviews load from DHT
- ✅ Error handling for network failures

---

#### Day 3-4: Browse & Cart
**Files**: `Browse.svelte`, `Cart.svelte`

**Browse.svelte Integration**:
```typescript
// Replace demo getAllListings
export async function getAllListings(client) {
  return await client.callZome({
    zome_name: 'listings',
    fn_name: 'get_all_listings',
    payload: {},
  });
}

onMount(async () => {
  loading = true;
  try {
    const client = await initHolochainClient();
    const result = await getAllListings(client);
    allListings = result.listings;
    applyFilters(); // Keep existing client-side filtering
  } catch (e) {
    error = e.message;
  } finally {
    loading = false;
  }
});
```

**Cart.svelte Integration**:
1. Create Svelte store for cart (migrate from localStorage):
   ```typescript
   // stores/cart.ts
   import { writable } from 'svelte/store';
   import { browser } from '$app/environment';

   const stored = browser ? localStorage.getItem('mycelix_cart') : null;
   export const cartItems = writable(stored ? JSON.parse(stored) : []);

   cartItems.subscribe((value) => {
     if (browser) {
       localStorage.setItem('mycelix_cart', JSON.stringify(value));
     }
   });
   ```

2. Update Cart.svelte to use store:
   ```typescript
   import { cartItems } from '$lib/stores/cart';

   // Remove local cartItems array, use $cartItems instead
   $: subtotal = $cartItems.reduce((sum, item) => sum + item.price * item.quantity, 0);
   ```

**Acceptance Criteria**:
- ✅ Browse loads all listings from DHT
- ✅ Filtering/sorting works with real data
- ✅ Cart uses Svelte store
- ✅ Cart persists across page reloads

---

#### Day 5: Dashboard
**Files**: `Dashboard.svelte`

**Integration**:
```typescript
// Create multiple zome call functions
export async function getMyProfile(client) {
  return await client.callZome({
    zome_name: 'users',
    fn_name: 'get_my_profile',
    payload: {},
  });
}

export async function getMyListings(client) {
  return await client.callZome({
    zome_name: 'listings',
    fn_name: 'get_my_listings',
    payload: {},
  });
}

export async function getMyPurchases(client) {
  return await client.callZome({
    zome_name: 'transactions',
    fn_name: 'get_my_purchases',
    payload: {},
  });
}

export async function getMySales(client) {
  return await client.callZome({
    zome_name: 'transactions',
    fn_name: 'get_my_sales',
    payload: {},
  });
}

export async function getMyReviews(client) {
  return await client.callZome({
    zome_name: 'reviews',
    fn_name: 'get_reviews_for_seller',
    payload: {},
  });
}

// Replace demo data with parallel loading
onMount(async () => {
  loading = true;
  try {
    const client = await initHolochainClient();

    const [userData, listings, purchases, sales, reviews] = await Promise.all([
      getMyProfile(client),
      getMyListings(client),
      getMyPurchases(client),
      getMySales(client),
      getMyReviews(client),
    ]);

    user = userData;
    myListings = listings;
    myPurchases = purchases;
    mySales = sales;
    myReviews = reviews;
  } catch (e) {
    error = e.message;
  } finally {
    loading = false;
  }
});
```

**Acceptance Criteria**:
- ✅ User profile loads from DHT
- ✅ All 4 tabs load real data
- ✅ Statistics calculated from real data
- ✅ Parallel loading for performance

---

### Week 2: Advanced Features (4 pages)

#### Day 6-7: MRCArbitration
**Files**: `MRCArbitration.svelte`

**Integration**:
```typescript
export async function getMyArbitrationCases(client) {
  return await client.callZome({
    zome_name: 'disputes',
    fn_name: 'get_my_arbitration_cases',
    payload: {},
  });
}

export async function castArbitratorVote(client, claimId, vote, reasoning) {
  return await client.callZome({
    zome_name: 'disputes',
    fn_name: 'cast_arbitrator_vote',
    payload: {
      claim_id: claimId,
      vote: vote, // 'Approve' or 'Reject'
      reasoning: reasoning,
    },
  });
}

// Implement real weighted voting
async function castVote(approve: boolean) {
  if (!voteReasoning.trim()) {
    alert('Please provide reasoning for your vote');
    return;
  }

  votingInProgress = true;
  try {
    const client = await initHolochainClient();
    const result = await castArbitratorVote(
      client,
      selectedDispute.claim_id,
      approve ? 'Approve' : 'Reject',
      voteReasoning
    );

    // Update UI with result
    selectedTransaction.my_vote = approve ? 'Approve' : 'Reject';
    selectedTransaction.votes_cast += 1;

    // Check consensus (requires fetching updated dispute state)
    const updatedDispute = await getDisputeDetails(client, selectedDispute.claim_id);
    selectedTransaction = updatedDispute;

    voteSuccess = 'Vote cast successfully!';

    // Reload cases to reflect changes
    await loadArbitrationCases();
  } catch (e) {
    voteError = e.message;
  } finally {
    votingInProgress = false;
  }
}
```

**Real-Time Updates**:
```typescript
// Subscribe to dispute updates
client.on('signal', (signal) => {
  if (signal.zome_name === 'disputes' && signal.data.type === 'vote_cast') {
    // Update dispute in real-time when other arbitrators vote
    updateDisputeVoteCount(signal.data.claim_id, signal.data.new_vote_count);
  }
});
```

**Acceptance Criteria**:
- ✅ Loads real arbitration cases from DHT
- ✅ Votes recorded on-chain
- ✅ Weighted voting calculated correctly
- ✅ Consensus tracked in real-time
- ✅ Real-time updates when other arbitrators vote

---

#### Day 8: Checkout & Transactions
**Files**: `Checkout.svelte`, `Transactions.svelte`

**Checkout.svelte Integration**:
```typescript
export async function createTransaction(client, listingHash, quantity, shippingAddress, paymentMethod) {
  return await client.callZome({
    zome_name: 'transactions',
    fn_name: 'create_transaction',
    payload: {
      listing_hash: listingHash,
      quantity: quantity,
      shipping_address: shippingAddress,
      payment_method: paymentMethod,
    },
  });
}

async function completeCheckout() {
  processingCheckout = true;
  try {
    const client = await initHolochainClient();

    // Create transactions for all cart items
    const transactions = await Promise.all(
      $cartItems.map((item) =>
        createTransaction(
          client,
          item.listing_hash,
          item.quantity,
          shippingAddress,
          paymentMethod
        )
      )
    );

    // Clear cart
    cartItems.set([]);

    // Redirect to transactions page
    window.location.href = '/transactions';
  } catch (e) {
    checkoutError = e.message;
  } finally {
    processingCheckout = false;
  }
}
```

**Transactions.svelte Integration**:
```typescript
export async function getMyTransactions(client) {
  return await client.callZome({
    zome_name: 'transactions',
    fn_name: 'get_my_transactions',
    payload: {},
  });
}

export async function updateTransactionStatus(client, transactionHash, status, trackingNumber = null) {
  return await client.callZome({
    zome_name: 'transactions',
    fn_name: 'update_transaction_status',
    payload: {
      transaction_hash: transactionHash,
      status: status,
      tracking_number: trackingNumber,
    },
  });
}

// Real-time transaction updates
client.on('signal', (signal) => {
  if (signal.zome_name === 'transactions' && signal.data.type === 'status_changed') {
    // Update transaction status in real-time
    updateTransactionInList(signal.data.transaction_hash, signal.data.new_status);
  }
});
```

**Acceptance Criteria**:
- ✅ Checkout creates real DHT transactions
- ✅ Transaction list loads from DHT
- ✅ Status updates recorded on-chain
- ✅ Real-time status notifications
- ✅ Buyer and seller actions work

---

### Week 3: Testing & Polish

#### Day 9-10: State Management & Real-Time
**Files**: `stores/*.ts`, WebSocket setup

**Tasks**:
1. Create Svelte stores:
   - `stores/auth.ts` - User authentication state
   - `stores/holochain.ts` - Client connection state
   - `stores/cart.ts` - Shopping cart (already done)
   - `stores/notifications.ts` - Real-time notifications

2. Implement WebSocket signal handling:
   ```typescript
   // lib/holochain/signals.ts
   import { notifications } from '$lib/stores/notifications';

   export function setupSignalHandlers(client) {
     client.on('signal', (signal) => {
       switch (signal.zome_name) {
         case 'transactions':
           handleTransactionSignal(signal);
           break;
         case 'disputes':
           handleDisputeSignal(signal);
           break;
         case 'listings':
           handleListingSignal(signal);
           break;
       }
     });
   }

   function handleTransactionSignal(signal) {
     if (signal.data.type === 'status_changed') {
       notifications.add({
         type: 'info',
         message: `Transaction ${signal.data.transaction_id} status: ${signal.data.new_status}`,
       });
     }
   }
   ```

3. Add notification UI component:
   ```svelte
   <!-- lib/components/Notifications.svelte -->
   <script>
     import { notifications } from '$lib/stores/notifications';
   </script>

   {#if $notifications.length > 0}
     <div class="notifications">
       {#each $notifications as notification}
         <div class="notification {notification.type}">
           {notification.message}
         </div>
       {/each}
     </div>
   {/if}
   ```

**Acceptance Criteria**:
- ✅ Global state management with stores
- ✅ Real-time notifications working
- ✅ Connection status displayed
- ✅ Graceful connection loss handling

---

#### Day 11-12: Automated Testing
**Files**: `tests/e2e/*.spec.ts`, `tests/unit/*.test.ts`

**E2E Tests with Playwright**:
```bash
npm install -D @playwright/test
```

```typescript
// tests/e2e/marketplace-flow.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Complete Marketplace Flow', () => {
  test('should complete purchase flow', async ({ page }) => {
    // Navigate to browse
    await page.goto('/browse');

    // Search for item
    await page.fill('input[type="search"]', 'keyboard');
    await page.click('.listing-card:first-child');

    // Add to cart
    await page.click('button:has-text("Add to Cart")');

    // Go to cart
    await page.click('a[href="/cart"]');
    expect(await page.locator('.cart-items').count()).toBeGreaterThan(0);

    // Checkout
    await page.click('button:has-text("Proceed to Checkout")');

    // Fill shipping
    await page.fill('input[name="name"]', 'Test User');
    await page.fill('input[name="address"]', '123 Test St');
    await page.click('button:has-text("Next")');

    // Select payment
    await page.click('input[value="crypto"]');
    await page.click('button:has-text("Next")');

    // Complete
    await page.click('button:has-text("Complete Purchase")');

    // Verify redirect to transactions
    await expect(page).toHaveURL('/transactions');
  });
});
```

**Unit Tests with Vitest**:
```bash
npm install -D vitest @testing-library/svelte
```

```typescript
// tests/unit/TrustBadge.test.ts
import { render } from '@testing-library/svelte';
import { expect, test } from 'vitest';
import TrustBadge from '$lib/components/TrustBadge.svelte';

test('should display correct trust tier', () => {
  const { getByText } = render(TrustBadge, {
    props: { trustScore: 87.5, showLabel: true },
  });

  expect(getByText('87.5%')).toBeTruthy();
  expect(getByText('Excellent')).toBeTruthy();
});

test('should show tooltip on hover', async () => {
  const { container, getByText } = render(TrustBadge, {
    props: {
      trustScore: 92,
      breakdown: {
        transactionCount: 45,
        positiveReviews: 38,
      },
    },
  });

  const badge = container.querySelector('.trust-badge');
  await badge.dispatchEvent(new MouseEvent('mouseenter'));

  expect(getByText('Transactions:')).toBeTruthy();
  expect(getByText('45')).toBeTruthy();
});
```

**Acceptance Criteria**:
- ✅ E2E tests for all user flows
- ✅ Unit tests for all components
- ✅ CI/CD pipeline configured
- ✅ Test coverage > 80%

---

#### Day 13-15: Performance & Production Readiness

**Tasks**:
1. **Code Splitting**:
   ```typescript
   // vite.config.ts
   export default defineConfig({
     build: {
       rollupOptions: {
         output: {
           manualChunks: {
             'holochain': ['@holochain/client'],
             'ipfs': ['ipfs-http-client'],
             'vendor': ['svelte', 'svelte/store'],
           },
         },
       },
     },
   });
   ```

2. **Lazy Loading**:
   ```typescript
   // Lazy load heavy components
   const MRCArbitration = () => import('./routes/MRCArbitration.svelte');
   const Checkout = () => import('./routes/Checkout.svelte');
   ```

3. **Error Boundaries**:
   ```svelte
   <!-- lib/components/ErrorBoundary.svelte -->
   <script>
     import { onMount } from 'svelte';

     let error = null;

     onMount(() => {
       window.addEventListener('error', (e) => {
         error = e.error;
       });

       window.addEventListener('unhandledrejection', (e) => {
         error = e.reason;
       });
     });
   </script>

   {#if error}
     <div class="error-boundary">
       <h2>Something went wrong</h2>
       <p>{error.message}</p>
       <button on:click={() => window.location.reload()}>Reload</button>
     </div>
   {:else}
     <slot />
   {/if}
   ```

4. **Accessibility Audit**:
   - Run axe-core audits
   - Add ARIA labels
   - Ensure keyboard navigation
   - Test with screen readers
   - Fix color contrast issues

5. **Performance Optimization**:
   - Lighthouse audit (target 90+ score)
   - Image optimization (WebP format)
   - CSS purging (remove unused styles)
   - Bundle size optimization (< 500KB)

**Acceptance Criteria**:
- ✅ Lighthouse score > 90
- ✅ Bundle size < 500KB
- ✅ WCAG 2.1 AA compliant
- ✅ Error boundaries catch all errors
- ✅ Lazy loading implemented

---

## 🎯 Success Metrics

### Functional
- ✅ All 8 pages connected to Holochain
- ✅ Real-time updates working
- ✅ State management with Svelte stores
- ✅ No demo data remaining
- ✅ All TODO placeholders replaced

### Quality
- ✅ E2E test coverage > 80%
- ✅ Unit test coverage > 80%
- ✅ Zero console errors
- ✅ Lighthouse score > 90
- ✅ WCAG 2.1 AA compliant

### Performance
- ✅ Page load < 2 seconds
- ✅ Bundle size < 500KB
- ✅ Time to interactive < 3 seconds
- ✅ Real-time updates < 500ms latency

---

## 🚧 Potential Challenges

### Challenge 1: Holochain Connection Management
**Issue**: WebSocket connection drops, needs reconnection logic
**Solution**:
```typescript
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

async function connectWithRetry() {
  try {
    const client = await AppWebsocket.connect('ws://localhost:8888');
    reconnectAttempts = 0;
    return client;
  } catch (e) {
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
      reconnectAttempts++;
      await new Promise((resolve) => setTimeout(resolve, 1000 * reconnectAttempts));
      return connectWithRetry();
    }
    throw new Error('Failed to connect to Holochain after multiple attempts');
  }
}
```

### Challenge 2: Data Migration from Demo to Real
**Issue**: Demo data structure might not match DHT structure exactly
**Solution**:
- Create type interfaces for all data structures
- Add validation layer between DHT and UI
- Use adapters to transform DHT data to UI format

### Challenge 3: Real-Time Update Synchronization
**Issue**: Multiple tabs open might show stale data
**Solution**:
- Use BroadcastChannel API for cross-tab communication
- Implement optimistic updates with rollback
- Add refresh button as fallback

### Challenge 4: Large Listing Sets
**Issue**: Loading all listings might be slow
**Solution**:
- Implement pagination at DHT level
- Add virtual scrolling for long lists
- Cache frequently accessed listings

---

## 📋 Pre-Integration Checklist

### Environment Setup
- [ ] Holochain conductor running on localhost:8888
- [ ] IPFS node running on localhost:5001
- [ ] `@holochain/client` installed
- [ ] All zomes compiled and DNA deployed
- [ ] Test agents created in conductor

### Development Tools
- [ ] Playwright installed for E2E tests
- [ ] Vitest installed for unit tests
- [ ] axe-core installed for accessibility
- [ ] Lighthouse CLI for performance

### Documentation
- [ ] Holochain zome API documentation available
- [ ] Data structure schemas documented
- [ ] Signal/notification format documented
- [ ] Error code reference created

---

## 🎉 Phase 4 Completion Criteria

Phase 4 will be considered complete when:

1. **All 8 pages connected**: Every page loads real data from Holochain DHT
2. **Real-time updates**: WebSocket signals update UI without refresh
3. **State management**: Svelte stores manage global state
4. **Testing**: E2E and unit tests passing with >80% coverage
5. **Performance**: Lighthouse score >90, bundle <500KB
6. **Accessibility**: WCAG 2.1 AA compliant
7. **Production ready**: No console errors, all edge cases handled
8. **Documentation**: Integration guide complete

**Target Completion**: 2-3 weeks from Phase 4 start

---

## 📄 Related Documentation

- **Phase 3 Complete**: [PHASE_3_COMPLETE.md](./PHASE_3_COMPLETE.md)
- **Frontend Pages**: [PHASE_3_FRONTEND_PAGES_SUMMARY.md](./PHASE_3_FRONTEND_PAGES_SUMMARY.md)
- **IPFS Integration**: [IPFS_INTEGRATION_COMPLETE.md](./IPFS_INTEGRATION_COMPLETE.md)
- **Project Status**: [README.md](./README.md)

---

**Status**: 📋 Planning Complete - Ready to Execute Phase 4

**Next Action**: Begin Day 1-2 tasks (Holochain client setup and ListingDetail integration)

🚀 **Phase 4: Let's connect the frontend to the decentralized backend!** 🚀
