// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server'
import jwt from 'jsonwebtoken'
import crypto from 'crypto'
import bcrypt from 'bcryptjs'
import { db, users, sessions, dbHelpers } from '../../../../lib/drizzle/db'
import { eq } from 'drizzle-orm'
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

    // Check if user already exists (using helper functions)
    const existingEmail = await dbHelpers.findUserByEmail(email)
    if (existingEmail) {
      return NextResponse.json(
        { error: 'Email already registered' },
        { status: 400 }
      )
    }

    const existingUsername = await dbHelpers.findUserByUsername(username)
    if (existingUsername) {
      return NextResponse.json(
        { error: 'Username already taken' },
        { status: 400 }
      )
    }

    // Hash password
    const passwordHash = await bcrypt.hash(password, 10)

    // Create user using Drizzle (so much cleaner than raw SQL!)
    const [newUser] = await db.insert(users).values({
      email,
      username,
      passwordHash,
      fullName,
      reputationScore: 0,
      trustLevel: 'novice',
      emailVerified: false,
      isActive: true,
      preferences: {},
      notificationSettings: {
        email_validations: true,
        email_updates: true,
        push_alerts: false
      }
    }).returning()

    // Create session with Drizzle
    const refreshToken = crypto.randomBytes(32).toString('hex')
    const refreshTokenHash = crypto.createHash('sha256').update(refreshToken).digest('hex')
    
    const expiresAt = new Date()
    expiresAt.setDate(expiresAt.getDate() + 30)
    
    await db.insert(sessions).values({
      userId: newUser.id,
      refreshTokenHash,
      ipAddress: request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || '',
      userAgent: request.headers.get('user-agent') || '',
      expiresAt
    })

    // Generate JWT
    const token = jwt.sign(
      { 
        userId: newUser.id,
        email: newUser.email,
        username: newUser.username
      },
      JWT_SECRET,
      { expiresIn: '7d' }
    )

    return NextResponse.json({
      user: {
        id: newUser.id,
        email: newUser.email,
        username: newUser.username,
        fullName: newUser.fullName,
        reputationScore: newUser.reputationScore,
        trustLevel: newUser.trustLevel
      },
      token,
      refreshToken,
      message: 'Registration successful! 🌍 Welcome to Terra Atlas!'
    }, { status: 201 })
  } catch (error: any) {
    logger.error('Registration error', error)
    return NextResponse.json(
      { error: 'Registration failed', details: error.message },
      { status: 500 }
    )
  }
}
