import { loadStripe } from '@stripe/stripe-js'
import logger from '@/lib/logger'

// Client-side Stripe instance
const stripePublishableKey = process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY || ''

if (!stripePublishableKey && typeof window !== 'undefined') {
  logger.warn('NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY is not set')
}

// Lazy-load Stripe on client side
let stripePromise: ReturnType<typeof loadStripe> | null = null

export const getStripe = () => {
  if (!stripePromise) {
    stripePromise = loadStripe(stripePublishableKey || 'YOUR_STRIPE_PUBLISHABLE_KEY')
  }
  return stripePromise
}