import { NextRequest, NextResponse } from 'next/server'
import {
  Pledge,
  PledgeInput,
  validatePledge,
  ProjectPledgeSummary,
  calculateProjectTarget
} from '@/lib/pledge/types'
import logger from '@/lib/logger'

// In-memory pledge store (would be database in production)
const pledges: Map<string, Pledge> = new Map()

// Generate unique ID
function generateId(): string {
  return `pledge_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

// GET - Get pledges for a project
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const projectId = searchParams.get('projectId')

  if (!projectId) {
    return NextResponse.json(
      { error: 'Project ID required' },
      { status: 400 }
    )
  }

  // Get all pledges for this project
  const projectPledges = Array.from(pledges.values())
    .filter(p => p.projectId === projectId && p.status !== 'cancelled')
    .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())

  const totalPledged = projectPledges.reduce((sum, p) => sum + p.amount, 0)

  // Estimate target based on project ID (would come from project data)
  const capacityMatch = projectId.match(/(\d+)/)
  const estimatedCapacity = capacityMatch ? parseInt(capacityMatch[1]) : 100
  const targetAmount = calculateProjectTarget(estimatedCapacity)

  const summary: ProjectPledgeSummary = {
    projectId,
    totalPledged,
    pledgeCount: projectPledges.length,
    targetAmount,
    percentFunded: (totalPledged / targetAmount) * 100,
    topPledgers: projectPledges
      .filter(p => !p.anonymous)
      .slice(0, 5)
      .map(p => ({
        name: p.email.split('@')[0],
        amount: p.amount,
        date: p.createdAt
      })),
    recentActivity: projectPledges
      .slice(0, 10)
      .map(p => ({
        amount: p.amount,
        date: p.createdAt,
        anonymous: p.anonymous
      }))
  }

  return NextResponse.json({
    success: true,
    summary
  })
}

// POST - Create new pledge
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const input: PledgeInput = body

    // Validate
    const validation = validatePledge(input)
    if (!validation.valid) {
      return NextResponse.json(
        { error: 'Invalid pledge', details: validation.errors },
        { status: 400 }
      )
    }

    // Create pledge
    const pledge: Pledge = {
      id: generateId(),
      projectId: input.projectId,
      email: input.email,
      amount: input.amount,
      status: 'pending',
      createdAt: new Date(),
      anonymous: input.anonymous || false,
      message: input.message
    }

    // Store
    pledges.set(pledge.id, pledge)

    logger.info(`New pledge created: ${pledge.id} for project ${pledge.projectId} - $${pledge.amount}`)

    // In production, would send confirmation email here

    return NextResponse.json({
      success: true,
      pledge: {
        id: pledge.id,
        amount: pledge.amount,
        status: pledge.status,
        message: 'Thank you for your pledge! Check your email to confirm.'
      }
    })
  } catch (error) {
    logger.error('Pledge creation error', error)
    return NextResponse.json(
      { error: 'Failed to create pledge' },
      { status: 500 }
    )
  }
}

// PUT - Confirm or update pledge
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    const { pledgeId, action } = body

    const pledge = pledges.get(pledgeId)
    if (!pledge) {
      return NextResponse.json(
        { error: 'Pledge not found' },
        { status: 404 }
      )
    }

    if (action === 'confirm') {
      pledge.status = 'confirmed'
      pledge.confirmedAt = new Date()
      pledges.set(pledgeId, pledge)

      return NextResponse.json({
        success: true,
        message: 'Pledge confirmed!'
      })
    }

    if (action === 'cancel') {
      pledge.status = 'cancelled'
      pledges.set(pledgeId, pledge)

      return NextResponse.json({
        success: true,
        message: 'Pledge cancelled'
      })
    }

    return NextResponse.json(
      { error: 'Invalid action' },
      { status: 400 }
    )
  } catch (error) {
    logger.error('Pledge update error', error)
    return NextResponse.json(
      { error: 'Failed to update pledge' },
      { status: 500 }
    )
  }
}
