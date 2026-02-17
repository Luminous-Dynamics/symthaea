# Manual Testing Checklist - Phase 4 Pages

**Purpose**: Verify all 10 pages work correctly before automated testing
**When to Use**: After Phase 4 completion, before Phase 5 begins
**How to Test**: Run dev server and follow each test case

---

## 🚀 Setup

```bash
cd /srv/luminous-dynamics/Mycelix-Core/mycelix-marketplace/frontend
npm install
npm run dev
```

**Prerequisites**:
- Holochain conductor running (backend)
- IPFS node running (or mock mode)
- Test data seeded in Holochain DHT

---

## 📋 Test Case 1: Browse Marketplace

**Page**: `/browse`
**Priority**: Critical
**Estimated Time**: 5 minutes

### Test Steps:

1. **Navigate to Browse**
   - [ ] Page loads without errors
   - [ ] Gradient background displays correctly
   - [ ] Header shows "Browse Marketplace"

2. **Listing Display**
   - [ ] At least one listing card displays
   - [ ] Each card shows: image, title, price, category, seller
   - [ ] Trust score badge displays (even if placeholder)
   - [ ] Cards are responsive (test mobile view)

3. **Search Functionality**
   - [ ] Type in search box → listings filter in real-time
   - [ ] Search matches title and description
   - [ ] No results message appears when no matches

4. **Category Filter**
   - [ ] Dropdown shows 10 categories + "All Categories"
   - [ ] Selecting category filters listings
   - [ ] "All Categories" shows all listings

5. **Price Range Filter**
   - [ ] Min/max price sliders work
   - [ ] Listings filter to price range
   - [ ] Price values update as sliders move

6. **Sort Options**
   - [ ] "Newest" sorts by created_at descending
   - [ ] "Price: Low to High" sorts correctly
   - [ ] "Price: High to Low" sorts correctly
   - [ ] "Trust Score" sorts by trust (even if placeholder)

7. **View Toggle**
   - [ ] Grid view displays by default
   - [ ] List view button switches to list layout
   - [ ] Grid view button switches back

8. **Navigation**
   - [ ] Click listing card → redirects to `/listing/[hash]`
   - [ ] Browser back button works

### Expected Errors:
- If Holochain not running: "Failed to connect" notification
- If no listings: Empty state with "No listings found"

---

## 📋 Test Case 2: Listing Detail

**Page**: `/listing/[listing_hash]`
**Priority**: Critical
**Estimated Time**: 5 minutes

### Test Steps:

1. **Page Load**
   - [ ] Page loads with listing data
   - [ ] Breadcrumbs show: Browse › Category › Title
   - [ ] Main photo displays prominently

2. **Photo Gallery**
   - [ ] Main photo shows first image
   - [ ] Thumbnails display below (if multiple photos)
   - [ ] Click thumbnail → changes main photo
   - [ ] Thumbnails scroll horizontally on mobile

3. **Product Information**
   - [ ] Title displays correctly
   - [ ] Price shows with $ formatting
   - [ ] Category displays
   - [ ] Description shows with preserved whitespace
   - [ ] Quantity available displays

4. **Seller Card**
   - [ ] Seller name displays
   - [ ] Trust score shows (placeholder OK)
   - [ ] Member since date shows
   - [ ] Avatar displays (or placeholder)

5. **Quantity Selector**
   - [ ] Default quantity is 1
   - [ ] + button increments quantity
   - [ ] - button decrements (min 1, button disabled at 1)
   - [ ] Manual input works
   - [ ] Invalid input (0, negative, letters) handled

6. **Add to Cart**
   - [ ] Click "Add to Cart" → success notification
   - [ ] Cart icon updates with item count (if visible)
   - [ ] Can add multiple times (quantity increases)

7. **Buy Now**
   - [ ] Click "Buy Now" → redirects to `/checkout`
   - [ ] Item added to cart before redirect

8. **Reviews Section**
   - [ ] Reviews display if present
   - [ ] Star rating shows for each review
   - [ ] Reviewer name and date show
   - [ ] Comment text displays
   - [ ] "No reviews yet" if empty

9. **Responsive Design**
   - [ ] Mobile: Photos stack vertically
   - [ ] Mobile: Seller card moves below description
   - [ ] Mobile: Buttons stack vertically

### Expected Errors:
- Invalid listing hash: "Listing not found" notification
- Holochain error: "Failed to load listing" notification

---

## 📋 Test Case 3: Shopping Cart

**Page**: `/cart`
**Priority**: Critical
**Estimated Time**: 5 minutes

### Test Steps:

1. **Empty Cart**
   - [ ] Empty state shows cart icon
   - [ ] "Your cart is empty" message
   - [ ] "Browse Marketplace" button present
   - [ ] Click button → redirects to `/browse`

2. **Cart with Items** (Add items via Browse → ListingDetail first)
   - [ ] Cart items display in list
   - [ ] Each item shows: image, title, seller, price, quantity
   - [ ] Order summary shows on right (sticky on desktop)

3. **Quantity Controls**
   - [ ] + button increments quantity
   - [ ] - button decrements (disabled at 1)
   - [ ] Manual input works
   - [ ] Totals update immediately

4. **Remove Item**
   - [ ] Click trash icon → confirmation
   - [ ] Item removed from cart
   - [ ] Success notification
   - [ ] Totals update
   - [ ] If last item removed → empty state

5. **Order Summary**
   - [ ] Subtotal = sum(price × quantity)
   - [ ] Tax = subtotal × 0.08
   - [ ] Shipping = $5.99 (or $0 if empty)
   - [ ] Total = subtotal + tax + shipping
   - [ ] All values formatted as currency

6. **Navigation**
   - [ ] "Continue Shopping" → back to `/browse`
   - [ ] "Proceed to Checkout" → to `/checkout`

7. **Persistence**
   - [ ] Refresh page → cart items persist (localStorage)
   - [ ] Close browser → reopen → items still there

### Expected Behavior:
- Cart persists across page navigation
- Totals always correct
- Responsive on mobile

---

## 📋 Test Case 4: Checkout

**Page**: `/checkout`
**Priority**: Critical
**Estimated Time**: 7 minutes

### Test Steps:

1. **Page Load**
   - [ ] Page loads with cart items
   - [ ] Multi-step progress indicator shows
   - [ ] Step 1 (Shipping) is active

2. **Step 1: Shipping Information**
   - [ ] All fields present: name, address, city, state, postal code, country
   - [ ] Fields validate on submit
   - [ ] Error messages for empty fields
   - [ ] "Continue to Payment" button

3. **Step 2: Payment Method**
   - [ ] Payment method options display
   - [ ] Can select payment method
   - [ ] "Back" button returns to Step 1
   - [ ] "Continue to Review" button

4. **Step 3: Review Order**
   - [ ] Shipping address displays correctly
   - [ ] Payment method displays
   - [ ] Cart items list shows
   - [ ] Order summary shows totals
   - [ ] "Back" button returns to Step 2
   - [ ] "Complete Order" button

5. **Submit Order**
   - [ ] Click "Complete Order"
   - [ ] Loading state shows
   - [ ] Success notification appears
   - [ ] Redirects to `/transactions` or `/dashboard`
   - [ ] Cart is cleared after success

6. **Validation**
   - [ ] Cannot proceed without filling required fields
   - [ ] Cannot proceed without selecting payment method
   - [ ] Invalid postal code shows error

7. **Batch Transaction Creation**
   - [ ] Multiple items → multiple transactions created
   - [ ] All transactions succeed or all fail (atomic)

### Expected Errors:
- Empty cart: Redirect to cart with message
- Holochain error: Transaction failed notification
- Network error: Retry prompt

---

## 📋 Test Case 5: Dashboard

**Page**: `/dashboard`
**Priority**: High
**Estimated Time**: 5 minutes

### Test Steps:

1. **Page Load**
   - [ ] Page loads without errors
   - [ ] Gradient background displays
   - [ ] Welcome message with username

2. **Profile Section**
   - [ ] Avatar displays (or placeholder)
   - [ ] Username displays
   - [ ] Trust score shows (PoGQ)
   - [ ] Verified badge (if applicable)
   - [ ] Member since date

3. **Stats Grid**
   - [ ] Total Listings count
   - [ ] Sales Completed count
   - [ ] Purchases Made count
   - [ ] Average Rating (stars)
   - [ ] All stats formatted correctly

4. **Recent Transactions**
   - [ ] Last 5 transactions display
   - [ ] Each shows: title, amount, date, status
   - [ ] Status badge has correct color
   - [ ] "View All" button → `/transactions`

5. **Active Listings**
   - [ ] Last 5 listings display
   - [ ] Each shows: image, title, price, views
   - [ ] "View All" button → `/browse` (filtered to my listings)

6. **Quick Actions**
   - [ ] "Create Listing" button → `/create-listing`
   - [ ] "Browse Marketplace" button → `/browse`
   - [ ] "View Transactions" button → `/transactions`

7. **Empty States**
   - [ ] No transactions: "No recent activity"
   - [ ] No listings: "Create your first listing"

8. **Parallel Data Loading**
   - [ ] Page loads quickly (Promise.all)
   - [ ] Loading states show for each section
   - [ ] Errors handled gracefully

### Expected Behavior:
- Data loads in < 2 seconds
- Stats update in real-time
- Responsive on mobile

---

## 📋 Test Case 6: Create Listing

**Page**: `/create-listing`
**Priority**: High
**Estimated Time**: 7 minutes

### Test Steps:

1. **Page Load**
   - [ ] Form displays with all fields
   - [ ] Title input focused on load

2. **Form Fields**
   - [ ] Title input (max 100 chars)
   - [ ] Description textarea (max 2000 chars)
   - [ ] Price input (currency format)
   - [ ] Quantity input (integer, min 1)
   - [ ] Category dropdown (10 options)
   - [ ] Character counters update in real-time

3. **Photo Upload**
   - [ ] Click "Upload Photos" → file picker opens
   - [ ] Select image → preview displays
   - [ ] Multiple images show in grid
   - [ ] First image marked as "Main"
   - [ ] Remove button on each photo
   - [ ] Click remove → photo disappears
   - [ ] Max 10 photos enforced

4. **Form Validation**
   - [ ] Submit empty form → error notifications
   - [ ] Title < 5 chars → error
   - [ ] Description < 20 chars → error
   - [ ] Price ≤ $0 → error
   - [ ] No photos → error
   - [ ] Valid form → success

5. **Submit Listing**
   - [ ] Click "Create Listing"
   - [ ] "Uploading Photos..." notification
   - [ ] "Creating listing..." notification
   - [ ] Success notification
   - [ ] Redirects to listing detail page

6. **Cancel**
   - [ ] Click "Cancel" → redirects to `/dashboard`
   - [ ] No listing created

7. **Mock IPFS Upload**
   - [ ] Photos "upload" (mock)
   - [ ] Mock CIDs generated (start with "Qm")
   - [ ] 1-second delay simulated

### Expected Errors:
- No photos: Warning (allow but warn)
- File too large: Error
- Invalid image format: Error

---

## 📋 Test Case 7: Transactions

**Page**: `/transactions`
**Priority**: High
**Estimated Time**: 7 minutes

### Test Steps:

1. **Page Load**
   - [ ] Transactions list displays
   - [ ] Purchases and sales combined
   - [ ] Sorted by date (newest first)

2. **Filter Tabs**
   - [ ] "All" tab shows purchases + sales
   - [ ] "Purchases" tab shows only purchases
   - [ ] "Sales" tab shows only sales
   - [ ] Active tab highlighted

3. **Transaction Card Display**
   - [ ] Each card shows: photo, title, amount, date, status
   - [ ] Status badge color-coded (pending, shipped, delivered, completed)
   - [ ] Buyer/seller name shows
   - [ ] Transaction date formatted

4. **Buyer Actions** (on purchases)
   - [ ] "Confirm Delivery" button shows when status = shipped
   - [ ] Click "Confirm Delivery" → status updates to delivered
   - [ ] Success notification
   - [ ] "Leave Review" button appears after delivery
   - [ ] Click "Leave Review" → redirects to `/submit-review?...`
   - [ ] "File Dispute" button shows for shipped/delivered
   - [ ] Click "File Dispute" → redirects to `/file-dispute?...`

5. **Seller Actions** (on sales)
   - [ ] "Mark as Shipped" button shows when status = pending
   - [ ] Click "Mark as Shipped" → tracking number prompt
   - [ ] Enter tracking number → status updates to shipped
   - [ ] Success notification

6. **Status Filter**
   - [ ] Filter by status dropdown
   - [ ] Selecting status filters transactions
   - [ ] "All" shows everything

7. **Empty States**
   - [ ] No transactions: "No transactions yet" message
   - [ ] No purchases: "Browse to make your first purchase"
   - [ ] No sales: "Create a listing to make sales"

8. **Parallel Data Fetching**
   - [ ] Purchases and sales loaded in parallel
   - [ ] Fast page load (<2 seconds)

### Expected Behavior:
- Real-time status updates
- Actions only show when appropriate
- Responsive on mobile

---

## 📋 Test Case 8: Submit Review

**Page**: `/submit-review?transaction=...&listing=...&title=...&seller=...`
**Priority**: Medium
**Estimated Time**: 5 minutes

### Test Steps:

1. **Page Load**
   - [ ] Page loads with URL parameters
   - [ ] Context card shows listing title and seller
   - [ ] If no params → error notification + redirect

2. **Star Rating**
   - [ ] 5 star buttons display
   - [ ] Stars are empty by default
   - [ ] Hover over star → fills star + all before it
   - [ ] Click star → sets rating
   - [ ] Rating description updates (Poor, Fair, Good, Very Good, Excellent)

3. **Comment Field**
   - [ ] Textarea displays
   - [ ] Character counter (1000 max)
   - [ ] Counter updates as typing

4. **Review Guidelines**
   - [ ] Guidelines section displays
   - [ ] Clear expectations listed

5. **Form Validation**
   - [ ] Submit without rating → error "Rating required"
   - [ ] Submit without comment → error "Comment required"
   - [ ] Comment < 10 chars → error "Too short"
   - [ ] Valid form → success

6. **Submit Review**
   - [ ] Click "Submit Review"
   - [ ] Loading state shows
   - [ ] Success notification
   - [ ] Redirects to `/transactions`

7. **Cancel**
   - [ ] Click "Cancel" → back to `/transactions`

### Expected Errors:
- Missing URL params: Redirect with error
- Holochain error: Review failed notification

---

## 📋 Test Case 9: File Dispute

**Page**: `/file-dispute?transaction=...&title=...&seller=...`
**Priority**: Medium
**Estimated Time**: 7 minutes

### Test Steps:

1. **Page Load**
   - [ ] Page loads with red gradient (serious context)
   - [ ] Context card shows transaction details
   - [ ] MRC information box explains arbitration
   - [ ] If no params → error + redirect

2. **Dispute Reason**
   - [ ] Dropdown shows 7 reasons
   - [ ] Default selected (not_as_described)
   - [ ] Selecting reason updates dropdown

3. **Description Field**
   - [ ] Textarea displays
   - [ ] Character counter (2000 max)
   - [ ] Counter updates as typing

4. **Evidence Upload**
   - [ ] Click "Upload Evidence" → file picker
   - [ ] Select image → preview displays
   - [ ] Multiple files show in grid
   - [ ] Remove button on each file
   - [ ] Max 10 files enforced
   - [ ] Supports images and PDFs

5. **Filing Guidelines**
   - [ ] Guidelines section displays
   - [ ] Warnings about false disputes

6. **Form Validation**
   - [ ] Submit without description → error
   - [ ] Description < 20 chars → error
   - [ ] No evidence → warning (but allow)
   - [ ] Valid form → success

7. **Submit Dispute**
   - [ ] Click "Submit to MRC"
   - [ ] "Uploading Evidence..." if files present
   - [ ] "Filing Dispute..." notification
   - [ ] Success notification
   - [ ] Redirects to `/transactions`

8. **Mock IPFS Upload**
   - [ ] Evidence "uploads" (mock)
   - [ ] Mock CIDs generated

### Expected Errors:
- Missing params: Error + redirect
- Holochain error: Dispute failed notification

---

## 📋 Test Case 10: MRC Arbitration

**Page**: `/mrc-arbitration`
**Priority**: Medium
**Estimated Time**: 10 minutes

### Test Steps:

1. **Access Control**
   - [ ] If not arbitrator → "Not Authorized" message
   - [ ] If arbitrator → dashboard loads

2. **Dashboard**
   - [ ] Arbitrator profile displays
   - [ ] Stats show: PoGQ, cases arbitrated, approval rate, active cases
   - [ ] Stats are real numbers (not placeholders)

3. **Dispute Tabs**
   - [ ] Three tabs: Pending, Active, Resolved
   - [ ] Badge counts on each tab
   - [ ] Active tab highlighted

4. **Pending Disputes Tab**
   - [ ] New disputes display
   - [ ] Each shows: title, photo, buyer/seller, reason, amount
   - [ ] Status badge: "Pending Assignment"
   - [ ] Click dispute → expands details

5. **Active Disputes Tab**
   - [ ] Assigned disputes display
   - [ ] "Vote Now" button on each
   - [ ] Click "Vote Now" → voting modal opens

6. **Voting Modal**
   - [ ] Dispute details display
   - [ ] Evidence section shows photos (if any)
   - [ ] Description shows
   - [ ] Reasoning textarea displays
   - [ ] "Approve" and "Reject" buttons
   - [ ] Both sides' arguments visible

7. **Cast Vote**
   - [ ] Select "Approve" or "Reject"
   - [ ] Write reasoning (required)
   - [ ] Submit vote
   - [ ] Success notification
   - [ ] Consensus progress updates

8. **Consensus Reached**
   - [ ] When 66% threshold met → dispute moves to Resolved
   - [ ] Success notification shows final decision
   - [ ] Dispute removed from Active tab

9. **Resolved Disputes Tab**
   - [ ] Completed disputes display
   - [ ] Final decision shows (Approved/Rejected)
   - [ ] Vote breakdown shows
   - [ ] Timestamp shows

10. **Weighted Voting**
    - [ ] Vote weight shown (based on PoGQ)
    - [ ] Higher PoGQ → more voting power
    - [ ] Consensus calculation accurate

### Expected Behavior:
- Only arbitrators can access
- Real-time updates
- Consensus calculation correct

---

## 📋 Cross-Cutting Tests

### Responsive Design (All Pages)

**Desktop** (1920x1080):
- [ ] All pages render correctly
- [ ] No horizontal scrolling
- [ ] Sidebars/grids use full width appropriately

**Tablet** (768x1024):
- [ ] Layouts adapt gracefully
- [ ] Grids collapse to 2 columns
- [ ] Navigation still accessible

**Mobile** (375x667):
- [ ] Single column layouts
- [ ] Touch targets ≥ 44x44px
- [ ] Forms fill screen width
- [ ] Images scale appropriately

### Navigation (All Pages)

- [ ] Logo/site name links to homepage
- [ ] Browser back button works
- [ ] Direct URL access works
- [ ] Invalid URLs → 404 page

### Notifications (All Pages)

- [ ] Success notifications are green
- [ ] Error notifications are red
- [ ] Info notifications are blue
- [ ] Notifications auto-dismiss after 5 seconds
- [ ] Can manually dismiss
- [ ] Stack multiple notifications

### TypeScript (All Pages)

- [ ] No console errors
- [ ] No TypeScript errors in build
- [ ] Autocomplete works in IDE
- [ ] Type inference correct

### Performance (All Pages)

- [ ] Initial load < 3 seconds
- [ ] No memory leaks
- [ ] Smooth scrolling
- [ ] No janky animations

---

## 🐛 Bug Tracking

### Issues Found:

| Page | Issue | Severity | Status |
|------|-------|----------|--------|
|      |       |          |        |

### Severity Levels:
- **Critical**: Blocks core functionality
- **High**: Major feature broken
- **Medium**: Feature partially works
- **Low**: UI/UX issue

---

## ✅ Sign-Off

**Tester Name**: _______________
**Date**: _______________
**Environment**: Development / Staging / Production
**Holochain Version**: _______________
**Frontend Version**: _______________

### Overall Assessment:

- [ ] All critical flows work
- [ ] No critical bugs found
- [ ] Ready for Phase 5
- [ ] Need fixes before proceeding

### Notes:

---

**Status**: Ready for Manual Testing
**Next Action**: Run dev server and test all pages

🌊 Testing with thoroughness and care!
