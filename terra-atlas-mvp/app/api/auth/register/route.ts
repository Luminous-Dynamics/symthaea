// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'
import { db } from '../../../../lib/drizzle/db'
import { users, sessions } from '../../../../lib/drizzle/schema'
import { eq } from 'drizzle-orm'
import crypto from 'crypto'
import logger from '@/lib/logger'

// JWT secret (in production, use environment variable)
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production'

// Email regex for validation
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

// Username regex (alphanumeric, underscore, dash, 3-20 chars)
const USERNAME_REGEX = /^[a-zA-Z0-9_-]{3,20}$/

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { email, username, password, fullName } = body

    // Validation
    if (!email || !username || !password) {
      return NextResponse.json(
        { error: 'Email, username, and password are required' },
        { status: 400 }
      )
    }

    // Email validation
    if (!EMAIL_REGEX.test(email)) {
      return NextResponse.json(
        { error: 'Invalid email format' },
        { status: 400 }
      )
    }

    // Username validation
    if (!USERNAME_REGEX.test(username)) {
      return NextResponse.json(
        { error: 'Username must be 3-20 characters, alphanumeric, underscore, or dash only' },
        { status: 400 }
      )
    }

    // Password strength validation
    if (password.length < 8) {
      return NextResponse.json(
        { error: 'Password must be at least 8 characters long' },
        { status: 400 }
      )
    }

    // Check if email already exists
    const [existingEmail] = await db
      .select()
      .from(users)
      .where(eq(users.email, email.toLowerCase()))
      .limit(1)

    if (existingEmail) {
      return NextResponse.json(
        { error: 'Email already registered' },
        { status: 409 }
      )
    }

    // Check if username already exists
    const [existingUsername] = await db
      .select()
      .from(users)
      .where(eq(users.username, username.toLowerCase()))
      .limit(1)

    if (existingUsername) {
      return NextResponse.json(
        { error: 'Username already taken' },
        { status: 409 }
      )
    }

    // Hash password
    const salt = await bcrypt.genSalt(10)
    const passwordHash = await bcrypt.hash(password, salt)

    // Create user
    const [user] = await db
      .insert(users)
      .values({
        email: email.toLowerCase(),
        username: username.toLowerCase(),
        passwordHash,
        fullName,
        // Generate default avatar using initials or service like Gravatar
        avatarUrl: `https://ui-avatars.com/api/?name=${encodeURIComponent(
          fullName || username
        )}&background=random`,
        createdAt: new Date(),
        updatedAt: new Date()
      })
      .returning()

    // Generate JWT token
    const token = jwt.sign(
      { 
        userId: user.id, 
        email: user.email,
        username: user.username 
      },
      JWT_SECRET,
      { expiresIn: '7d' }
    )

    // Generate refresh token
    const refreshToken = crypto.randomBytes(32).toString('hex')
    const refreshTokenHash = await bcrypt.hash(refreshToken, 10)

    // Create session
    await db
      .insert(sessions)
      .values({
        userId: user.id,
        refreshTokenHash,
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days
        ipAddress: request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || null,
        userAgent: request.headers.get('user-agent') || null
      })

    // Log activity (removed for now - activityLog table not in schema)

    // Return user data and tokens
    return NextResponse.json(
      {
        user: {
          id: user.id,
          email: user.email,
          username: user.username,
          fullName: user.fullName,
          avatarUrl: user.avatarUrl,
          trustLevel: user.trustLevel,
          reputationScore: user.reputationScore
        },
        token,
        refreshToken,
        expiresIn: 604800 // 7 days in seconds
      },
      { status: 201 }
    )
  } catch (error) {
    logger.error('Registration error', error)
    
    // Log error (removed for now - activityLog table not in schema)

    return NextResponse.json(
      { error: 'Failed to register user' },
      { status: 500 }
    )
  }
}
