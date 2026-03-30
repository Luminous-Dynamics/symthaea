// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import logger from '@/lib/logger'
import { requireServerEnv, serverEnv } from '@/lib/env.server'

// Initialize Supabase client
const supabaseUrl = serverEnv.NEXT_PUBLIC_SUPABASE_URL
const supabaseServiceKey = requireServerEnv('SUPABASE_SERVICE_ROLE_KEY')
const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey)

export async function GET(request: NextRequest) {
  try {
    // Get auth header
    const authHeader = request.headers.get('authorization')
    if (!authHeader) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // Verify the user's token
    const token = authHeader.replace('Bearer ', '')
    const { data: { user }, error: authError } = await supabaseAdmin.auth.getUser(token)
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 })
    }

    // Get query parameters
    const searchParams = request.nextUrl.searchParams
    const status = searchParams.get('status') || 'all'
    const limit = parseInt(searchParams.get('limit') || '50')
    const offset = parseInt(searchParams.get('offset') || '0')

    // Build query
    let query = supabaseAdmin
      .from('investments')
      .select('*', { count: 'exact' })
      .eq('user_id', user.id)
      .order('created_at', { ascending: false })

    if (status !== 'all') {
      query = query.eq('status', status)
    }

    // Execute query with pagination
    const { data: investments, error, count } = await query
      .range(offset, offset + limit - 1)

    if (error) {
      logger.error('Error fetching investments', error)
      return NextResponse.json({ error: error.message }, { status: 500 })
    }

    // Get portfolio summary
    const { data: summary } = await supabaseAdmin
      .from('portfolio_summary')
      .select('*')
      .eq('user_id', user.id)
      .single()

    return NextResponse.json({
      investments: investments || [],
      summary: summary || {
        total_investments: 0,
        unique_projects: 0,
        total_invested: 0,
        total_expected_return: 0,
        avg_return_percentage: 0,
        active_pledges: 0,
        total_pledged: 0
      },
      pagination: {
        total: count || 0,
        limit,
        offset,
        hasMore: (count || 0) > offset + limit
      }
    })
  } catch (error) {
    logger.error('Investments API error', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    // Get auth header
    const authHeader = request.headers.get('authorization')
    if (!authHeader) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // Verify the user's token
    const token = authHeader.replace('Bearer ', '')
    const { data: { user }, error: authError } = await supabaseAdmin.auth.getUser(token)
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 })
    }

    // Parse request body
    const {
      project_id: projectId,
      project_name: projectName,
      project_type: projectType,
      amount,
      investment_term_months: investmentTermMonths,
      payment_method: paymentMethod
    } = await request.json()

    // Validate amount
    if (!amount || amount < 10) {
      return NextResponse.json(
        { error: 'Minimum investment is $10' },
        { status: 400 }
      )
    }

    // Calculate expected return based on project type
    const returnRates: { [key: string]: number } = {
      'Solar': 0.14,
      'Wind': 0.16,
      'Battery': 0.15,
      'Hydro': 0.11,
      'Nuclear': 0.12,
      'default': 0.13
    }
    
    const returnRate = returnRates[projectType] || returnRates.default
    const expected_return = amount * returnRate * (investmentTermMonths || 12) / 12

    // Create investment record
    const { data: investment, error } = await supabaseAdmin
      .from('investments')
      .insert({
        user_id: user.id,
        project_id: projectId,
        project_name: projectName,
        project_type: projectType,
        amount,
        status: 'pending',
        payment_method: paymentMethod,
        investment_term_months: investmentTermMonths || 12,
        expected_return,
        share_percentage: amount / 1000000, // Simplified calculation
      metadata: {
        ip_address: request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip'),
        user_agent: request.headers.get('user-agent'),
        created_from: 'web_app'
      }
      })
      .select()
      .single()

    if (error) {
      logger.error('Error creating investment', error)
      return NextResponse.json({ error: error.message }, { status: 500 })
    }

    // Create initial transaction record
    await supabaseAdmin
      .from('investment_transactions')
      .insert({
        investment_id: investment.id,
        user_id: user.id,
        type: 'deposit',
        amount,
        balance_after: amount,
        description: `Initial investment in ${projectName}`
      })

    return NextResponse.json({
      success: true,
      investment,
      message: 'Investment created successfully. Please complete payment to confirm.'
    }, { status: 201 })
  } catch (error) {
    logger.error('Investments API error', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function PATCH(request: NextRequest) {
  try {
    // Get auth header
    const authHeader = request.headers.get('authorization')
    if (!authHeader) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // Verify the user's token
    const token = authHeader.replace('Bearer ', '')
    const { data: { user }, error: authError } = await supabaseAdmin.auth.getUser(token)
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 })
    }

    const {
      investment_id: investmentId,
      status,
      payment_id: paymentId,
      transaction_hash: transactionHash
    } = await request.json()

    if (!investmentId) {
      return NextResponse.json(
        { error: 'Investment ID is required' },
        { status: 400 }
      )
    }

    // Get the investment
    const { data: existingInvestment, error: fetchError } = await supabaseAdmin
      .from('investments')
      .select('*')
      .eq('id', investmentId)
      .eq('user_id', user.id)
      .single()

    if (fetchError || !existingInvestment) {
      return NextResponse.json(
        { error: 'Investment not found' },
        { status: 404 }
      )
    }

    // Only allow updates to pending investments
    if (existingInvestment.status !== 'pending') {
      return NextResponse.json(
        { error: 'Can only update pending investments' },
        { status: 400 }
      )
    }

    // Prepare update data
    const updateData: Record<string, any> = { status }

    if (paymentId) updateData.payment_id = paymentId
    if (transactionHash) updateData.transaction_hash = transactionHash
    
    if (status === 'confirmed') {
      updateData.confirmed_at = new Date().toISOString()
    } else if (status === 'completed') {
      updateData.completed_at = new Date().toISOString()
    } else if (status === 'cancelled') {
      updateData.cancelled_at = new Date().toISOString()
    }

    // Update the investment
    const { data: updatedInvestment, error: updateError } = await supabaseAdmin
      .from('investments')
      .update(updateData)
      .eq('id', investmentId)
      .eq('user_id', user.id)
      .select()
      .single()

    if (updateError) {
      logger.error('Error updating investment', updateError)
      return NextResponse.json({ error: updateError.message }, { status: 500 })
    }

    // Log transaction for status changes
    if (status === 'completed') {
      await supabaseAdmin
        .from('investment_transactions')
        .insert({
          investment_id: investmentId,
          user_id: user.id,
          type: 'deposit',
          amount: existingInvestment.amount,
          balance_after: existingInvestment.amount,
          description: `Investment confirmed for ${existingInvestment.project_name}`
        })
    } else if (status === 'cancelled' || status === 'refunded') {
      await supabaseAdmin
        .from('investment_transactions')
        .insert({
          investment_id: investmentId,
          user_id: user.id,
          type: 'refund',
          amount: existingInvestment.amount,
          balance_after: 0,
          description: `Investment ${status} for ${existingInvestment.project_name}`
        })
    }

    return NextResponse.json({
      success: true,
      investment: updatedInvestment,
      message: `Investment ${status} successfully`
    })
  } catch (error) {
    logger.error('Investments API error', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
