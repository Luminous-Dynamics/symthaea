import { NextResponse } from 'next/server'

export async function GET() {
  return NextResponse.json({ 
    status: 'ok',
    service: 'Terra Atlas MVP',
    timestamp: new Date().toISOString(),
    endpoints: {
      projects: '/api/projects',
      smr: '/api/smr',
      stripe: {
        createPaymentIntent: '/api/stripe/create-payment-intent',
        confirmPayment: '/api/stripe/confirm-payment',
        webhook: '/api/stripe/webhook'
      }
    }
  })
}