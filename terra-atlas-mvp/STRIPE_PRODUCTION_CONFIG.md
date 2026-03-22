# Stripe Production Configuration for Terra Atlas

## Current Status
- ✅ Test keys configured in .env.local
- 🔧 Production keys need to be obtained from Stripe Dashboard
- 🔒 Keys should be stored in BWS (Bitwarden Secrets Manager) for security

## Required Stripe Configuration

### 1. Production Keys Needed
```bash
# Store these in BWS when obtained:
bws create stripe-prod-publishable "pk_live_..."
bws create stripe-prod-secret "sk_live_..."
bws create stripe-prod-webhook "YOUR_STRIPE_WEBHOOK_SECRET..."
```

### 2. Stripe Products to Create

#### Investment Tiers
1. **Seed Investment** ($10 - $99)
   - Product ID: `prod_seed_investment`
   - One-time payment
   - Instant access to portfolio

2. **Growth Investment** ($100 - $999)
   - Product ID: `prod_growth_investment`
   - One-time payment
   - Priority project updates

3. **Impact Investment** ($1,000+)
   - Product ID: `prod_impact_investment`
   - One-time payment
   - VIP investor benefits

#### Subscription Plans (Future)
1. **Terra Atlas Pro** ($9.99/month)
   - Advanced analytics
   - Priority support
   - Early access to projects

2. **Terra Atlas Enterprise** (Custom pricing)
   - White-label solution
   - API access
   - Dedicated support

### 3. Webhook Endpoints to Configure

```javascript
// Production webhook endpoints
https://atlas.luminousdynamics.io/api/webhooks/stripe

// Events to listen for:
- payment_intent.succeeded
- payment_intent.failed
- customer.subscription.created
- customer.subscription.updated
- customer.subscription.deleted
- invoice.payment_succeeded
- checkout.session.completed
```

### 4. Payment Methods to Enable
- ✅ Credit/Debit Cards
- ✅ Bank Transfers (ACH)
- ✅ Wire Transfers (for large investments)
- 🔲 Crypto payments (future via Stripe)
- 🔲 International payments

### 5. Security & Compliance

#### PCI Compliance
- Use Stripe Elements or Checkout for card collection
- Never store card details directly
- All sensitive data handled by Stripe

#### KYC/AML Requirements
- Collect investor information via Stripe Identity
- Verify accredited investor status for large investments
- Maintain audit trail of all transactions

### 6. Integration Checklist

- [ ] Create Stripe account for Terra Atlas
- [ ] Complete business verification
- [ ] Set up bank account for payouts
- [ ] Configure tax settings
- [ ] Create products and prices
- [ ] Set up webhook endpoints
- [ ] Test payment flows
- [ ] Enable production mode
- [ ] Store keys in BWS
- [ ] Update .env.local with production keys
- [ ] Deploy to Vercel with production environment variables

### 7. Test Card Numbers (Development)

```
# Successful payment
4242 4242 4242 4242

# Requires authentication
4000 0025 0000 3155

# Declined
4000 0000 0000 9995

# Insufficient funds
4000 0000 0000 9995
```

### 8. Revenue Model

#### Transaction Fees
- Platform fee: 2% of investment amount
- Minimum fee: $0.50
- Maximum fee: $500

#### Example Calculations
- $100 investment → $2 platform fee
- $10,000 investment → $200 platform fee
- $50,000 investment → $500 platform fee (capped)

### 9. Payout Schedule
- Weekly payouts to project developers
- Monthly investor returns distribution
- Quarterly platform revenue reconciliation

## Next Steps

1. **Create Stripe Account**
   - Sign up at https://dashboard.stripe.com/register
   - Complete business verification
   - Add bank account details

2. **Configure Products**
   - Create investment tier products
   - Set up pricing plans
   - Configure payment methods

3. **Implement Payment Flow**
   - Build checkout page
   - Create success/failure pages
   - Implement webhook handlers

4. **Store Production Keys**
   ```bash
   # After obtaining production keys:
   bws create stripe-prod-publishable "pk_live_..."
   bws create stripe-prod-secret "sk_live_..."
   bws create stripe-prod-webhook "YOUR_STRIPE_WEBHOOK_SECRET..."
   ```

5. **Update Environment Variables**
   ```bash
   # In Vercel dashboard, add:
   NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_...
   STRIPE_SECRET_KEY=sk_live_...
   STRIPE_WEBHOOK_SECRET=YOUR_STRIPE_WEBHOOK_SECRET...
   ```

## Contact
For Stripe setup assistance:
- Stripe Support: https://support.stripe.com
- Terra Atlas Team: payments@terra-atlas.earth