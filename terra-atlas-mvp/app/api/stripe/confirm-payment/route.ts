// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import { stripe } from '@/lib/stripe'
import { createClient } from '@supabase/supabase-js'
import { requireServerEnv, serverEnv } from '@/lib/env.server'
import logger from '@/lib/logger'

// Initialize Supabase client for database operations
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
    const { paymentIntentId } = body

    if (!paymentIntentId) {
      return NextResponse.json(
        { error: 'Payment intent ID is required' },
        { status: 400 }
      )
    }

    // Retrieve payment intent from Stripe
    const paymentIntent = await stripe.paymentIntents.retrieve(paymentIntentId)

    // Verify payment belongs to authenticated user
    if (
      paymentIntent.metadata.userEmail !== user.email ||
      paymentIntent.metadata.userId !== user.id
    ) {
      return NextResponse.json(
        { error: 'Unauthorized - Payment does not belong to user' },
        { status: 403 }
      )
    }

    // Check payment status
    if (paymentIntent.status !== 'succeeded') {
      return NextResponse.json(
        { 
          error: 'Payment not completed',
          status: paymentIntent.status,
          requiresAction: paymentIntent.status === 'requires_action'
        },
        { status: 400 }
      )
    }

    // Payment successful - Create investment record in database
    const investmentData = {
      user_id: user.id,
      user_email: user.email,
      project_id: parseInt(paymentIntent.metadata.projectId),
      project_name: paymentIntent.metadata.projectName,
      project_type: paymentIntent.metadata.projectType,
      amount: paymentIntent.amount / 100, // Convert from cents to dollars
      investment_term_months: parseInt(paymentIntent.metadata.investmentTermMonths || '12'),
      payment_method: 'stripe',
      payment_status: 'completed',
      stripe_payment_intent_id: paymentIntent.id,
      stripe_customer_id: paymentIntent.customer as string,
      metadata: {
        ...paymentIntent.metadata,
        confirmedAt: new Date().toISOString()
      }
    }

    // Insert investment record
    const { data: investment, error: dbError } = await supabaseAdmin
      .from('investments')
      .insert([investmentData])
      .select()
      .single()

    if (dbError) {
      logger.error('Stripe confirm payment database error', dbError)
      // Payment succeeded but database failed - log for manual reconciliation
      await logFailedInvestmentRecord(paymentIntent, investmentData, dbError)
      
      return NextResponse.json({
        success: true,
        paymentStatus: 'succeeded',
        investmentRecorded: false,
        message: 'Payment successful. Investment record pending.',
        paymentIntentId: paymentIntent.id
      })
    }

    // Update project funding totals
    await updateProjectFunding(
      parseInt(paymentIntent.metadata.projectId),
      paymentIntent.amount / 100
    )

    // Send confirmation email (async - don't wait)
    sendInvestmentConfirmationEmail(user.email, investmentData).catch((error) =>
      logger.error('Sending investment confirmation email failed', error)
    )

    return NextResponse.json({
      success: true,
      paymentStatus: 'succeeded',
      investmentRecorded: true,
      investmentId: investment.id,
      amount: paymentIntent.amount / 100,
      projectName: paymentIntent.metadata.projectName,
      message: 'Investment successful!'
    })
  } catch (error: any) {
    logger.error('Payment confirmation error', error)
    return NextResponse.json(
      { error: error.message || 'Failed to confirm payment' },
      { status: 500 }
    )
  }
}

// Helper function to update project funding totals
async function updateProjectFunding(projectId: number, amount: number) {
  try {
    // Get current project funding
    const { data: project } = await supabaseAdmin
      .from('projects')
      .select('total_raised, investors_count')
      .eq('id', projectId)
      .single()

    if (project) {
      // Update funding totals
      await supabaseAdmin
        .from('projects')
        .update({
          total_raised: (project.total_raised || 0) + amount,
          investors_count: (project.investors_count || 0) + 1
        })
        .eq('id', projectId)
    }
  } catch (error) {
    logger.error('Failed to update project funding', error)
  }
}

// Helper function to log failed investment records for manual reconciliation
async function logFailedInvestmentRecord(
  paymentIntent: any,
  investmentData: any,
  error: any
) {
  try {
    await supabaseAdmin
      .from('failed_investment_records')
      .insert([{
        payment_intent_id: paymentIntent.id,
        investment_data: investmentData,
        error_details: error,
        created_at: new Date().toISOString()
      }])
  } catch (logError) {
    logger.error('Failed to log investment record failure', logError)
  }
}

// Helper function to send confirmation email
async function sendInvestmentConfirmationEmail(email: string, investmentData: any) {
  // This would integrate with your email service (SendGrid, AWS SES, etc.)
  // For now, just log it
  logger.info({
    message: 'Sending investment confirmation email',
    context: { email, projectName: investmentData?.project_name }
  })
}
