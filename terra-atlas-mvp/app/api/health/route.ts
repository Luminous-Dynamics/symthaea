// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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