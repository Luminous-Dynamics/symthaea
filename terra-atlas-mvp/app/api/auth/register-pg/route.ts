export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import jwt from 'jsonwebtoken'
import crypto from 'crypto'
import db from '@/lib/db'
import logger from '@/lib/logger'

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { email, username, password, fullName } = body

    // Validate input
    if (!email || !username || !password) {
      return NextResponse.json(
        { error: 'Email, username, and password are required' },
        { status: 400 }
      )
    }

    // Check if user already exists
    const existingEmail = await db.user.findByEmail(email)
    if (existingEmail) {
      return NextResponse.json(
        { error: 'Email already registered' },
        { status: 400 }
      )
    }

    const existingUsername = await db.user.findByUsername(username)
    if (existingUsername) {
      return NextResponse.json(
        { error: 'Username already taken' },
        { status: 400 }
      )
    }

    // Create user
    const user = await db.user.create({
      email,
      username,
      password,
      fullName
    })

    // Create session
    const refreshToken = crypto.randomBytes(32).toString('hex')
    const refreshTokenHash = crypto.createHash('sha256').update(refreshToken).digest('hex')
    
    await db.session.create({
      userId: user.id,
      refreshTokenHash,
      ipAddress: request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip'),
      userAgent: request.headers.get('user-agent')
    })

    // Generate JWT
    const token = jwt.sign(
      { 
        userId: user.id,
        email: user.email,
        username: user.username
      },
      JWT_SECRET,
      { expiresIn: '7d' }
    )

    // Log activity
    await db.activity.log({
      userId: user.id,
      action: 'user.register',
      resourceType: 'user',
      resourceId: user.id,
      ipAddress: request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip'),
      userAgent: request.headers.get('user-agent'),
      requestMethod: 'POST',
      requestPath: '/api/auth/register',
      statusCode: 201
    })

    return NextResponse.json({
      user: {
        id: user.id,
        email: user.email,
        username: user.username,
        fullName: user.full_name,
        reputationScore: user.reputation_score,
        trustLevel: user.trust_level
      },
      token,
      refreshToken,
      message: 'Registration successful!'
    }, { status: 201 })
  } catch (error: any) {
    logger.error('Registration error', error)
    return NextResponse.json(
      { error: 'Registration failed', details: error.message },
      { status: 500 }
    )
  }
}
