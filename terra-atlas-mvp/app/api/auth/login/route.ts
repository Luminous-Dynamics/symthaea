export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'
import { db } from '../../../../lib/drizzle/db'
import { users, sessions } from '../../../../lib/drizzle/schema'
import { eq, or, and } from 'drizzle-orm'
import crypto from 'crypto'
import logger from '@/lib/logger'

// JWT secret (in production, use environment variable)
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { emailOrUsername, password } = body

    // Validation
    if (!emailOrUsername || !password) {
      return NextResponse.json(
        { error: 'Email/username and password are required' },
        { status: 400 }
      )
    }

    // Find user by email or username
    const [user] = await db
      .select()
      .from(users)
      .where(
        and(
          or(
            eq(users.email, emailOrUsername.toLowerCase()),
            eq(users.username, emailOrUsername.toLowerCase())
          ),
          eq(users.isActive, true)
        )
      )
      .limit(1)

    if (!user) {
      // Don't reveal whether email/username exists
      return NextResponse.json(
        { error: 'Invalid credentials' },
        { status: 401 }
      )
    }

    // Verify password
    const isValidPassword = await bcrypt.compare(password, user.passwordHash)

    if (!isValidPassword) {
      // Log failed attempt (removed for now - activityLog table not in schema)
      return NextResponse.json(
        { error: 'Invalid credentials' },
        { status: 401 }
      )
    }

    // Update last login
    await db
      .update(users)
      .set({ lastLoginAt: new Date() })
      .where(eq(users.id, user.id))

    // Invalidate old sessions (optional - for single session)
    // await prisma.session.updateMany({
    //   where: { 
    //     userId: user.id,
    //     isActive: true
    //   },
    //   data: {
    //     isActive: false,
    //     revokedAt: new Date()
    //   }
    // })

    // Generate JWT token
    const token = jwt.sign(
      { 
        userId: user.id, 
        email: user.email,
        username: user.username,
        trustLevel: user.trustLevel,
        isAdmin: user.isAdmin,
        isModerator: user.isModerator
      },
      JWT_SECRET,
      { expiresIn: '7d' }
    )

    // Generate refresh token
    const refreshToken = crypto.randomBytes(32).toString('hex')
    const refreshTokenHash = await bcrypt.hash(refreshToken, 10)

    // Create new session
    await db
      .insert(sessions)
      .values({
        userId: user.id,
        refreshTokenHash,
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days
        ipAddress: request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || null,
        userAgent: request.headers.get('user-agent') || null
      })

    // Log successful login (removed for now - activityLog table not in schema)

    // Return user data and tokens
    return NextResponse.json({
      user: {
        id: user.id,
        email: user.email,
        username: user.username,
        fullName: user.fullName,
        avatarUrl: user.avatarUrl,
        bio: user.bio,
        trustLevel: user.trustLevel,
        reputationScore: user.reputationScore,
        validationsCount: user.validationsCount,
        validationAccuracy: user.validationAccuracy,
        isAdmin: user.isAdmin,
        isModerator: user.isModerator,
        emailVerified: user.emailVerified,
        createdAt: user.createdAt
      },
      token,
      refreshToken,
      expiresIn: 604800 // 7 days in seconds
    })
  } catch (error) {
    logger.error('Login error', error)
    
    // Log error (removed for now - activityLog table not in schema)

    return NextResponse.json(
      { error: 'Failed to login' },
      { status: 500 }
    )
  }
}
