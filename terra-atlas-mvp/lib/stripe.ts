import Stripe from 'stripe'
import { requireServerEnv } from './env.server'

// Server-side Stripe instance
const stripeSecretKey = requireServerEnv('STRIPE_SECRET_KEY')

export const stripe = new Stripe(stripeSecretKey, {
  apiVersion: '2025-08-27.basil',
  typescript: true,
})

// Helper function to format amount for Stripe (converts dollars to cents)
export const formatAmountForStripe = (amount: number): number => {
  return Math.round(amount * 100)
}

// Helper function to format amount from Stripe (converts cents to dollars)
export const formatAmountFromStripe = (amount: number): number => {
  return amount / 100
}
