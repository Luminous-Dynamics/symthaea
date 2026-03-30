// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { stripe, formatAmountForStripe } from '@/lib/stripe'
import { requireServerEnv, serverEnv } from '@/lib/env.server'
import logger from '@/lib/logger'

const supabaseUrl = serverEnv.NEXT_PUBLIC_SUPABASE_URL
const supabaseServiceKey = requireServerEnv('SUPABASE_SERVICE_ROLE_KEY')
const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey)

async function getAuthenticatedUser(request: NextRequest) {
  const authHeader = request.headers.get('authorization')
  if (!authHeader?.startsWith('Bearer ')) return null

  const token = authHeader.replace('Bearer ', '')
  const { data, error } = await supabaseAdmin.auth.getUser(token)
  if (error || !data.user) return null
  return data.user
}

export async function POST(request: NextRequest) {
  try {
    const user = await getAuthenticatedUser(request)
    if (!user?.email) {
      return NextResponse.json(
        { error: 'Unauthorized - please sign in' },
        { status: 401 }
      )
    }

    const body = await request.json()
    const { 
      amount, 
      projectId, 
      projectName, 
      projectType,
      investmentTermMonths,
      metadata = {}
    } = body

    // Validate input
    if (!amount || amount < 10) {
      return NextResponse.json(
        { error: 'Invalid amount. Minimum investment is $10' },
        { status: 400 }
      )
    }

    if (!projectId || !projectName) {
      return NextResponse.json(
        { error: 'Project information is required' },
        { status: 400 }
      )
    }

    // Equal opportunity: Same minimum for ALL projects including SMR
    // Everyone deserves access to clean energy investment, regardless of wealth

    // Create or retrieve Stripe customer
    let customerId = null
    
    // Search for existing customer by email
    const existingCustomers = await stripe.customers.list({
      email: user.email,
      limit: 1
    })

    if (existingCustomers.data.length > 0) {
      customerId = existingCustomers.data[0].id
    } else {
      // Create new customer
      const customer = await stripe.customers.create({
        email: user.email,
        name: (user.user_metadata as Record<string, any> | null)?.full_name || undefined,
        metadata: {
          userId: user.id || '',
          platform: 'terra-atlas'
        }
      })
      customerId = customer.id
    }

    // Create payment intent
    const paymentIntent = await stripe.paymentIntents.create({
      amount: formatAmountForStripe(amount),
      currency: 'usd',
      customer: customerId,
      metadata: {
        projectId: projectId.toString(),
        projectName,
        projectType: projectType || 'standard',
        investmentTermMonths: investmentTermMonths?.toString() || '12',
        userEmail: user.email,
        userId: user.id || '',
        ...metadata
      },
      description: `Investment in ${projectName}`,
      automatic_payment_methods: {
        enabled: true,
      },
      // For large SMR investments, allow manual confirmation
      confirmation_method: amount >= 100000 ? 'manual' : 'automatic',
    })

    return NextResponse.json({
      clientSecret: paymentIntent.client_secret,
      paymentIntentId: paymentIntent.id,
      amount,
      customerId
    })
  } catch (error: any) {
    logger.error('Stripe payment intent error', error)
    return NextResponse.json(
      { error: error.message || 'Failed to create payment intent' },
      { status: 500 }
    )
  }
}
