# Stripe Payment Processing Integration

## Overview
This document describes the Stripe payment integration for the Terra Atlas MVP, supporting both standard renewable energy investments and high-value SMR (Small Modular Reactor) investments.

## Implementation Status ✅

### Completed Features
- ✅ Stripe SDK integration (server and client)
- ✅ Payment Intent API endpoints
- ✅ React payment form component with Stripe Elements
- ✅ Investment confirmation workflow
- ✅ SMR-specific validation ($100K minimum)
- ✅ Customer creation and management
- ✅ Webhook handler for payment events
- ✅ Database integration with Supabase
- ✅ Failed payment recovery logging

### Components Created

#### 1. Server-Side Configuration (`/lib/stripe.ts`)
- Stripe server SDK initialization
- Helper functions for amount conversion
- API version: 2024-12-18.acacia

#### 2. Client-Side Configuration (`/lib/stripe-client.ts`)
- Stripe.js lazy loading
- Client-side SDK for browser

#### 3. API Endpoints

##### `/api/stripe/create-payment-intent`
- Creates Stripe payment intent
- Validates investment amounts
- SMR minimum: $100,000
- Standard minimum: $10
- Creates/retrieves Stripe customers
- Adds metadata for tracking

##### `/api/stripe/confirm-payment`
- Confirms successful payments
- Creates investment records in database
- Updates project funding totals
- Handles failed database writes gracefully

##### `/api/stripe/webhook`
- Handles Stripe webhook events
- Verifies webhook signatures
- Processes payment status updates
- Handles refunds and subscriptions

#### 4. React Components

##### `PaymentForm` Component (`/components/PaymentForm.tsx`)
- Stripe Elements integration
- Card input with real-time validation
- Loading and error states
- Success confirmation
- PCI-compliant card handling

#### 5. Integration Pages
- `/app/invest/[id]/page.tsx` - Standard investments
- `/app/invest/smr/[id]/page.tsx` - SMR investments

## Setup Instructions

### 1. Environment Variables
Copy `.env.stripe.example` to `.env.local` and add your Stripe keys:

```bash
# Required Stripe keys
STRIPE_SECRET_KEY=YOUR_STRIPE_SECRET_KEY...
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=YOUR_STRIPE_PUBLISHABLE_KEY...
STRIPE_WEBHOOK_SECRET=YOUR_STRIPE_WEBHOOK_SECRET...
```

### 2. Stripe Dashboard Setup

1. **Create Stripe Account**: https://dashboard.stripe.com/register
2. **Get API Keys**: https://dashboard.stripe.com/apikeys
3. **Create Webhook Endpoint**: 
   - Go to https://dashboard.stripe.com/webhooks
   - Add endpoint: `https://your-domain.com/api/stripe/webhook`
   - Select events:
     - `payment_intent.succeeded`
     - `payment_intent.payment_failed`
     - `charge.refunded`
   - Copy the webhook secret

### 3. Database Tables Required

The integration expects these Supabase tables:

```sql
-- Investments table
CREATE TABLE investments (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR NOT NULL,
  user_email VARCHAR NOT NULL,
  project_id INTEGER NOT NULL,
  project_name VARCHAR,
  project_type VARCHAR,
  amount DECIMAL(10,2) NOT NULL,
  investment_term_months INTEGER DEFAULT 12,
  payment_method VARCHAR DEFAULT 'stripe',
  payment_status VARCHAR DEFAULT 'pending',
  stripe_payment_intent_id VARCHAR UNIQUE,
  stripe_customer_id VARCHAR,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Failed investment records (for recovery)
CREATE TABLE failed_investment_records (
  id SERIAL PRIMARY KEY,
  payment_intent_id VARCHAR,
  investment_data JSONB,
  error_details JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Projects table (should have these columns)
ALTER TABLE projects ADD COLUMN IF NOT EXISTS total_raised DECIMAL(12,2) DEFAULT 0;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS investors_count INTEGER DEFAULT 0;
```

## Testing Guide

### Test Mode
The integration is configured for Stripe test mode. Use test cards:

#### Standard Test Cards
- **Success**: `4242 4242 4242 4242`
- **Declined**: `4000 0000 0000 0002`
- **Requires Auth**: `4000 0025 0000 3155`

#### Testing Flow
1. Navigate to an investment page
2. Enter investment amount
3. Click "Invest Now"
4. Review details in confirmation modal
5. Click "Proceed to Payment"
6. Enter test card details
7. Complete payment
8. Verify success message and redirect

### Testing SMR Investments
1. Navigate to `/smr` and select a project
2. Minimum investment is $100,000
3. Uses same payment flow with higher validation

### Testing Webhooks Locally
Use Stripe CLI for local webhook testing:

```bash
# Install Stripe CLI
brew install stripe/stripe-cli/stripe

# Login to Stripe
stripe login

# Forward webhooks to local server
stripe listen --forward-to localhost:3000/api/stripe/webhook

# In another terminal, trigger test events
stripe trigger payment_intent.succeeded
```

## Payment Flow Diagram

```
User Flow:
1. User selects investment amount
2. Clicks "Invest Now"
3. Reviews investment details
4. Clicks "Proceed to Payment"
5. Enters card details (Stripe Elements)
6. Stripe processes payment
7. Success: Investment recorded, redirect to portfolio
8. Failure: Error shown, user can retry

Technical Flow:
1. Frontend → create-payment-intent API
2. API → Stripe PaymentIntent creation
3. API → Returns client secret
4. Frontend → Stripe.confirmCardPayment()
5. Stripe → Processes payment
6. Frontend → confirm-payment API
7. API → Creates investment record
8. API → Updates project totals
9. Webhook → Updates payment status
```

## Security Considerations

### Implemented Security
- ✅ Server-side payment intent creation
- ✅ Client secret never exposed in code
- ✅ Webhook signature verification
- ✅ Authentication required for investments
- ✅ PCI-compliant card handling via Stripe Elements
- ✅ Amount validation on server
- ✅ SMR accredited investor checks

### Best Practices
1. Never log full card details
2. Always verify webhook signatures
3. Use HTTPS in production
4. Keep Stripe keys secure
5. Implement rate limiting for API endpoints
6. Monitor for suspicious payment patterns

## Monitoring & Analytics

### Stripe Dashboard
Monitor payments at: https://dashboard.stripe.com/payments

### Key Metrics to Track
- Payment success rate
- Average investment amount
- Failed payment reasons
- Customer lifetime value
- Payment method distribution

### Database Queries
```sql
-- Total investments by project
SELECT 
  project_id, 
  project_name,
  COUNT(*) as investor_count,
  SUM(amount) as total_raised
FROM investments
WHERE payment_status = 'completed'
GROUP BY project_id, project_name;

-- Failed payments needing attention
SELECT * FROM failed_investment_records
WHERE created_at > NOW() - INTERVAL '24 hours';
```

## Troubleshooting

### Common Issues

#### Payment Intent Creation Fails
- Check STRIPE_SECRET_KEY is set correctly
- Verify API key has correct permissions
- Check network connectivity to Stripe

#### Card Declined
- Use different test card numbers
- Check amount is within limits
- Verify customer country restrictions

#### Webhook Not Received
- Verify webhook URL is publicly accessible
- Check webhook secret is correct
- Review Stripe webhook logs

#### Database Write Fails
- Check Supabase connection
- Verify table schema matches
- Review failed_investment_records table

## Future Enhancements

### Planned Features
- [ ] Subscription/recurring investments
- [ ] Multiple payment methods (ACH, wire)
- [ ] Investment installment plans
- [ ] Automated invoice generation
- [ ] Tax document generation
- [ ] Refund processing UI
- [ ] Payment retry logic
- [ ] Fraud detection rules
- [ ] Multi-currency support
- [ ] Payment link generation

## Support & Resources

### Stripe Documentation
- [Stripe Docs](https://stripe.com/docs)
- [Payment Intents Guide](https://stripe.com/docs/payments/payment-intents)
- [Stripe Elements](https://stripe.com/docs/payments/elements)
- [Testing Guide](https://stripe.com/docs/testing)
- [Webhook Events](https://stripe.com/docs/webhooks)

### Internal Contacts
- Frontend Integration: `/components/PaymentForm.tsx`
- API Endpoints: `/app/api/stripe/`
- Configuration: `/lib/stripe.ts`

## Deployment Checklist

- [ ] Set production Stripe keys in environment
- [ ] Configure production webhook endpoint
- [ ] Test payment flow in staging
- [ ] Verify database migrations
- [ ] Set up monitoring alerts
- [ ] Configure rate limiting
- [ ] Review security settings
- [ ] Test error handling
- [ ] Document support procedures
- [ ] Train support team