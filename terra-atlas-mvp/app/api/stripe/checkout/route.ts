import { NextRequest, NextResponse } from 'next/server'
import Stripe from 'stripe'
import logger from '@/lib/logger'

// Check if we're in demo mode (no real Stripe keys)
const isDemoMode = !process.env.STRIPE_SECRET_KEY || 
                   process.env.STRIPE_SECRET_KEY === 'YOUR_STRIPE_SECRET_KEY' ||
                   process.env.STRIPE_SECRET_KEY.includes('placeholder')

// Initialize Stripe only if not in demo mode
const stripe = !isDemoMode ? new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2025-08-27.basil',
}) : null

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { projectId, projectName, amount, userId, userEmail } = body

    // Validate input
    if (!projectId || !projectName || !amount) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      )
    }

    // Demo mode - create mock session
    if (isDemoMode) {
      const mockSessionId = `cs_demo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      
      // Store session data in a simple way for demo (in production, use database)
      const demoCheckoutUrl = `/invest/demo-checkout?session=${mockSessionId}&amount=${amount}&project=${encodeURIComponent(projectName)}&projectId=${projectId}`
      
      return NextResponse.json({ 
        sessionId: mockSessionId,
        checkoutUrl: demoCheckoutUrl,
        testMode: true,
        demoMode: true,
        message: 'Demo mode active - no real charges will be made'
      })
    }

    // Real Stripe mode
    const session = await stripe!.checkout.sessions.create({
      payment_method_types: ['card'],
      mode: 'payment',
      success_url: `${process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3002'}/invest/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3002'}/invest/cancel`,
      line_items: [
        {
          price_data: {
            currency: 'usd',
            product_data: {
              name: `Investment: ${projectName}`,
              description: `Energy project investment in ${projectName}`,
              metadata: {
                projectId: projectId.toString(),
                type: 'energy_investment'
              }
            },
            unit_amount: Math.round(amount * 100), // Convert to cents
          },
          quantity: 1,
        }
      ],
      metadata: {
        projectId: projectId.toString(),
        userId: userId || 'anonymous',
        userEmail: userEmail || 'not_provided',
        investmentType: 'energy_project'
      },
      customer_email: userEmail,
      // Test mode configuration
      payment_intent_data: {
        metadata: {
          projectId: projectId.toString(),
          environment: 'test'
        }
      }
    })

    return NextResponse.json({ 
      sessionId: session.id,
      checkoutUrl: session.url,
      testMode: true 
    })
    
  } catch (error) {
    logger.error('Stripe checkout error', error)
    return NextResponse.json(
      { error: 'Failed to create checkout session' },
      { status: 500 }
    )
  }
}

export async function GET(req: NextRequest) {
  // Get session status for success page
  const searchParams = req.nextUrl.searchParams
  const sessionId = searchParams.get('session_id')
  
  if (!sessionId) {
    return NextResponse.json(
      { error: 'Missing session ID' },
      { status: 400 }
    )
  }

  // Handle demo mode sessions
  if (isDemoMode || sessionId.startsWith('cs_demo_')) {
    // In demo mode, return mock successful session data
    return NextResponse.json({
      status: 'paid',
      customerEmail: 'demo@terraatlas.io',
      amountTotal: 10000, // $100 in cents
      metadata: {
        projectId: searchParams.get('projectId') || 'demo-project',
        environment: 'demo'
      },
      testMode: true,
      demoMode: true
    })
  }

  try {
    const session = await stripe!.checkout.sessions.retrieve(sessionId)
    
    return NextResponse.json({
      status: session.payment_status,
      customerEmail: session.customer_email,
      amountTotal: session.amount_total,
      metadata: session.metadata,
      testMode: true
    })
    
  } catch (error) {
    logger.error('Error retrieving Stripe session', error)
    return NextResponse.json(
      { error: 'Failed to retrieve session' },
      { status: 500 }
    )
  }
}