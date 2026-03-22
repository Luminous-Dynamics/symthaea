export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import { headers } from 'next/headers'
import { stripe } from '@/lib/stripe'
import { createClient } from '@supabase/supabase-js'
import { requireServerEnv, serverEnv } from '@/lib/env.server'
import logger from '@/lib/logger'

// Initialize Supabase client
const supabaseUrl = serverEnv.NEXT_PUBLIC_SUPABASE_URL
const supabaseServiceKey = requireServerEnv('SUPABASE_SERVICE_ROLE_KEY')
const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey)

// Stripe webhook secret for verifying the webhook signature
const webhookSecret = requireServerEnv('STRIPE_WEBHOOK_SECRET')

export async function POST(request: NextRequest) {
  try {
    const body = await request.text()
    const headersList = await headers()
    const signature = headersList.get('stripe-signature')

    if (!signature) {
      return NextResponse.json(
        { error: 'No signature found' },
        { status: 400 }
      )
    }

    // Verify webhook signature
    let event
    try {
      event = stripe.webhooks.constructEvent(body, signature, webhookSecret)
    } catch (err: any) {
      logger.error('Webhook signature verification failed', err)
      return NextResponse.json(
        { error: 'Invalid signature' },
        { status: 400 }
      )
    }

    // Handle different event types
    switch (event.type) {
      case 'payment_intent.succeeded': {
        const paymentIntent = event.data.object as any
        logger.info({ message: 'Payment succeeded', context: { paymentIntentId: paymentIntent.id } })
        
        // Update investment record status if needed
        await supabaseAdmin
          .from('investments')
          .update({ 
            payment_status: 'completed',
            stripe_status: 'succeeded',
            updated_at: new Date().toISOString()
          })
          .eq('stripe_payment_intent_id', paymentIntent.id)
        
        break
      }

      case 'payment_intent.payment_failed': {
        const paymentIntent = event.data.object as any
        logger.warn({ message: 'Payment failed', context: { paymentIntentId: paymentIntent.id } })
        
        // Update investment record with failure
        await supabaseAdmin
          .from('investments')
          .update({ 
            payment_status: 'failed',
            stripe_status: 'failed',
            failure_reason: paymentIntent.last_payment_error?.message,
            updated_at: new Date().toISOString()
          })
          .eq('stripe_payment_intent_id', paymentIntent.id)
        
        break
      }

      case 'charge.refunded': {
        const charge = event.data.object as any
        logger.info({ message: 'Charge refunded', context: { chargeId: charge.id } })
        
        // Handle refund - update investment record
        await supabaseAdmin
          .from('investments')
          .update({ 
            payment_status: 'refunded',
            refund_amount: charge.amount_refunded / 100,
            refunded_at: new Date().toISOString()
          })
          .eq('stripe_payment_intent_id', charge.payment_intent)
        
        break
      }

      case 'customer.subscription.created': {
        // Handle subscription creation for recurring investments
        const subscription = event.data.object as any
        logger.info({ message: 'Subscription created', context: { subscriptionId: subscription.id } })
        
        // Create subscription record
        await supabaseAdmin
          .from('investment_subscriptions')
          .insert([{
            stripe_subscription_id: subscription.id,
            stripe_customer_id: subscription.customer,
            status: subscription.status,
            current_period_start: new Date(subscription.current_period_start * 1000).toISOString(),
            current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
            metadata: subscription.metadata
          }])
        
        break
      }

      case 'customer.subscription.deleted': {
        // Handle subscription cancellation
        const subscription = event.data.object as any
        logger.warn({ message: 'Subscription cancelled', context: { subscriptionId: subscription.id } })
        
        // Update subscription record
        await supabaseAdmin
          .from('investment_subscriptions')
          .update({ 
            status: 'cancelled',
            cancelled_at: new Date().toISOString()
          })
          .eq('stripe_subscription_id', subscription.id)
        
        break
      }

      case 'invoice.payment_succeeded': {
        // Handle successful recurring payment
        const invoice = event.data.object as any
        logger.info({ message: 'Invoice paid', context: { invoiceId: invoice.id } })
        
        // Record recurring payment
        if (invoice.subscription) {
          await supabaseAdmin
            .from('recurring_payments')
            .insert([{
              stripe_invoice_id: invoice.id,
              stripe_subscription_id: invoice.subscription,
              amount: invoice.amount_paid / 100,
              paid_at: new Date(invoice.status_transitions.paid_at * 1000).toISOString()
            }])
        }
        
        break
      }

      default:
        logger.warn({ message: 'Unhandled Stripe event type', context: { eventType: event.type } })
    }

    // Return success response
    return NextResponse.json({ received: true })
  } catch (error: any) {
    logger.error('Stripe webhook error', error)
    return NextResponse.json(
      { error: 'Webhook processing failed' },
      { status: 500 }
    )
  }
}
