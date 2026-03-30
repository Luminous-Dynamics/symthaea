// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import { db } from '../../../lib/drizzle/db'
import { users, dataPoints, validations } from '../../../lib/drizzle/schema'
import { eq, and, desc, sql } from 'drizzle-orm'
import jwt from 'jsonwebtoken'
import logger from '@/lib/logger'

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production'

// Verify JWT token
async function verifyToken(token: string) {
  try {
    const decoded = jwt.verify(token, JWT_SECRET) as any
    return decoded
  } catch (error) {
    return null
  }
}

// GET endpoint - Get validations for a data point or user
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const dataPointId = searchParams.get('dataPointId')
    const userId = searchParams.get('userId')
    const limit = parseInt(searchParams.get('limit') || '100')
    const offset = parseInt(searchParams.get('offset') || '0')

    // Build query conditions
    const conditions = []
    if (dataPointId) conditions.push(eq(validations.dataPointId, dataPointId))
    if (userId) conditions.push(eq(validations.userId, userId))

    // Fetch validations with user and datapoint info
    const validationsList = await db
      .select({
        validation: validations,
        user: {
          id: users.id,
          username: users.username,
          avatarUrl: users.avatarUrl,
          trustLevel: users.trustLevel,
          reputationScore: users.reputationScore
        },
        dataPoint: {
          id: dataPoints.id,
          dataType: dataPoints.dataType,
          trustScore: dataPoints.trustScore,
          confirmsCount: dataPoints.confirmsCount,
          disputesCount: dataPoints.disputesCount,
          flagsCount: dataPoints.flagsCount
        }
      })
      .from(validations)
      .leftJoin(users, eq(validations.userId, users.id))
      .leftJoin(dataPoints, eq(validations.dataPointId, dataPoints.id))
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .orderBy(desc(validations.createdAt))
      .limit(limit)
      .offset(offset)

    // Get total count
    const [countResult] = await db
      .select({ count: sql<number>`count(*)::int` })
      .from(validations)
      .where(conditions.length > 0 ? and(...conditions) : undefined)

    const total = countResult?.count || 0

    return NextResponse.json({
      validations: validationsList,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + validationsList.length < total
      }
    })
  } catch (error) {
    logger.error('Error fetching validations', error)
    return NextResponse.json(
      { error: 'Failed to fetch validations' },
      { status: 500 }
    )
  }
}

// POST endpoint - Create or update a validation
export async function POST(request: NextRequest) {
  try {
    // Get auth token
    const authHeader = request.headers.get('authorization')
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      )
    }

    const token = authHeader.substring(7)
    const decoded = await verifyToken(token)
    
    if (!decoded) {
      return NextResponse.json(
        { error: 'Invalid token' },
        { status: 401 }
      )
    }

    const userId = decoded.userId
    const body = await request.json()
    const { 
      dataPointId, 
      validationType, 
      comment, 
      evidenceUrls 
    }: {
      dataPointId: string
      validationType: string
      comment?: string
      evidenceUrls?: string[]
    } = body

    // Validation
    if (!dataPointId || !validationType) {
      return NextResponse.json(
        { error: 'dataPointId and validationType are required' },
        { status: 400 }
      )
    }

    // Check if validation type is valid
    if (!['confirm', 'dispute', 'flag'].includes(validationType)) {
      return NextResponse.json(
        { error: 'Invalid validation type. Must be: confirm, dispute, or flag' },
        { status: 400 }
      )
    }

    // Check if data point exists
    const [dataPoint] = await db
      .select()
      .from(dataPoints)
      .where(eq(dataPoints.id, dataPointId))
      .limit(1)

    if (!dataPoint) {
      return NextResponse.json(
        { error: 'Data point not found' },
        { status: 404 }
      )
    }

    // Check for existing validation
    const [existing] = await db
      .select()
      .from(validations)
      .where(
        and(
          eq(validations.userId, userId),
          eq(validations.dataPointId, dataPointId)
        )
      )
      .limit(1)

    let validation
    const ipAddress = request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || null
    const userAgent = request.headers.get('user-agent') || null
    
    if (existing) {
      // Update existing validation
      [validation] = await db
        .update(validations)
        .set({
          previousType: existing.validationType,
          validationType,
          comment,
          evidenceUrls: evidenceUrls || [],
          ipAddress,
          userAgent,
          updatedAt: new Date()
        })
        .where(eq(validations.id, existing.id))
        .returning()
    } else {
      // Create new validation
      [validation] = await db
        .insert(validations)
        .values({
          userId,
          dataPointId,
          validationType,
          comment,
          evidenceUrls: evidenceUrls || [],
          ipAddress,
          userAgent,
          clientFingerprint: request.headers.get('x-client-fingerprint') || null,
          createdAt: new Date(),
          updatedAt: new Date()
        })
        .returning()
    }

    // Update data point trust score based on validation type
    const countUpdates: any = {}
    if (validationType === 'confirm') {
      countUpdates.confirmsCount = sql`${dataPoints.confirmsCount} + 1`
    } else if (validationType === 'dispute') {
      countUpdates.disputesCount = sql`${dataPoints.disputesCount} + 1`
    } else if (validationType === 'flag') {
      countUpdates.flagsCount = sql`${dataPoints.flagsCount} + 1`
    }

    // Calculate trust score (simple formula)
    const trustScore = sql<number>`
      CASE 
        WHEN (${dataPoints.confirmsCount} + ${dataPoints.disputesCount} + ${dataPoints.flagsCount}) = 0 THEN 50
        ELSE (${dataPoints.confirmsCount}::float / (${dataPoints.confirmsCount} + ${dataPoints.disputesCount} + ${dataPoints.flagsCount})::float) * 100
      END
    `

    await db
      .update(dataPoints)
      .set({
        ...countUpdates,
        trustScore,
        updatedAt: new Date()
      })
      .where(eq(dataPoints.id, dataPointId))

    // Update user stats
    await db
      .update(users)
      .set({
        validationsCount: sql`${users.validationsCount} + ${existing ? 0 : 1}`,
        updatedAt: new Date()
      })
      .where(eq(users.id, userId))

    // Fetch updated data point
    const [updatedDataPoint] = await db
      .select()
      .from(dataPoints)
      .where(eq(dataPoints.id, dataPointId))

    return NextResponse.json(
      {
        validation,
        dataPoint: {
          id: updatedDataPoint?.id,
          trustScore: updatedDataPoint?.trustScore,
          confirmsCount: updatedDataPoint?.confirmsCount,
          disputesCount: updatedDataPoint?.disputesCount,
          flagsCount: updatedDataPoint?.flagsCount
        },
        message: existing ? 'Validation updated' : 'Validation created'
      },
      { status: existing ? 200 : 201 }
    )
  } catch (error) {
    logger.error('Validation error', error)
    return NextResponse.json(
      { error: 'Failed to process validation' },
      { status: 500 }
    )
  }
}

// DELETE endpoint - Remove a validation
export async function DELETE(request: NextRequest) {
  try {
    // Get auth token
    const authHeader = request.headers.get('authorization')
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      )
    }

    const token = authHeader.substring(7)
    const decoded = await verifyToken(token)
    
    if (!decoded) {
      return NextResponse.json(
        { error: 'Invalid token' },
        { status: 401 }
      )
    }

    const userId = decoded.userId
    const { searchParams } = new URL(request.url)
    const dataPointId = searchParams.get('dataPointId')

    if (!dataPointId) {
      return NextResponse.json(
        { error: 'dataPointId is required' },
        { status: 400 }
      )
    }

    // Find validation
    const [validation] = await db
      .select()
      .from(validations)
      .where(
        and(
          eq(validations.userId, userId),
          eq(validations.dataPointId, dataPointId)
        )
      )
      .limit(1)

    if (!validation) {
      return NextResponse.json(
        { error: 'Validation not found' },
        { status: 404 }
      )
    }

    // Delete validation
    await db
      .delete(validations)
      .where(eq(validations.id, validation.id))

    // Update data point counts
    const validationType = validation.validationType
    const countUpdates: any = {}
    if (validationType === 'confirm') {
      countUpdates.confirmsCount = sql`GREATEST(${dataPoints.confirmsCount} - 1, 0)`
    } else if (validationType === 'dispute') {
      countUpdates.disputesCount = sql`GREATEST(${dataPoints.disputesCount} - 1, 0)`
    } else if (validationType === 'flag') {
      countUpdates.flagsCount = sql`GREATEST(${dataPoints.flagsCount} - 1, 0)`
    }

    // Recalculate trust score
    const trustScore = sql<number>`
      CASE 
        WHEN (${dataPoints.confirmsCount} + ${dataPoints.disputesCount} + ${dataPoints.flagsCount}) = 0 THEN 50
        ELSE (${dataPoints.confirmsCount}::float / (${dataPoints.confirmsCount} + ${dataPoints.disputesCount} + ${dataPoints.flagsCount})::float) * 100
      END
    `

    await db
      .update(dataPoints)
      .set({
        ...countUpdates,
        trustScore,
        updatedAt: new Date()
      })
      .where(eq(dataPoints.id, dataPointId))

    // Update user stats
    await db
      .update(users)
      .set({
        validationsCount: sql`GREATEST(${users.validationsCount} - 1, 0)`,
        updatedAt: new Date()
      })
      .where(eq(users.id, userId))

    return NextResponse.json({
      message: 'Validation removed successfully'
    })
  } catch (error) {
    logger.error('Delete validation error', error)
    return NextResponse.json(
      { error: 'Failed to delete validation' },
      { status: 500 }
    )
  }
}