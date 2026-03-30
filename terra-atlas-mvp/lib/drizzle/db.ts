// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Drizzle database connection for Terra Atlas
// Works perfectly on NixOS unlike Prisma!

import { drizzle } from 'drizzle-orm/node-postgres'
import { Pool } from 'pg'
import * as schema from './schema'
import logger from '@/lib/logger'
import { serverEnv } from '@/lib/env.server'

// Create connection pool
const pool = new Pool({
  connectionString: serverEnv.DATABASE_URL,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
})

// Create drizzle instance with schema
export const db = drizzle(pool, { schema })

// Export all schema tables for easy importing
export * from './schema'

// Test connection
pool.on('connect', () => {
  logger.info({ message: 'Drizzle connected to PostgreSQL' })
})

pool.on('error', (err) => {
  logger.error('Drizzle database error', err)
})

// Helper functions for common operations
export const dbHelpers = {
  // User operations
  async findUserByEmail(email: string) {
    return await db.query.users.findFirst({
      where: (users, { eq }) => eq(users.email, email),
    })
  },

  async findUserByUsername(username: string) {
    return await db.query.users.findFirst({
      where: (users, { eq }) => eq(users.username, username),
    })
  },

  async findUserById(id: string) {
    return await db.query.users.findFirst({
      where: (users, { eq }) => eq(users.id, id),
    })
  },

  // Validation operations
  async getUserValidations(userId: string) {
    return await db.query.validations.findMany({
      where: (validations, { eq }) => eq(validations.userId, userId),
      with: {
        dataPoint: true,
      },
      orderBy: (validations, { desc }) => [desc(validations.createdAt)],
    })
  },

  async getDataPointValidations(dataPointId: string) {
    return await db.query.validations.findMany({
      where: (validations, { eq }) => eq(validations.dataPointId, dataPointId),
      with: {
        user: {
          columns: {
            passwordHash: false, // Don't return password hash
          },
        },
      },
      orderBy: (validations, { desc }) => [desc(validations.createdAt)],
    })
  },

  // Data point operations
  async findDataPointsInArea(minLat: number, maxLat: number, minLng: number, maxLng: number) {
    return await db.query.dataPoints.findMany({
      where: (dataPoints, { and, between }) =>
        and(
          between(dataPoints.latitude, minLat.toString(), maxLat.toString()),
          between(dataPoints.longitude, minLng.toString(), maxLng.toString())
        ),
      orderBy: (dataPoints, { desc }) => [desc(dataPoints.trustScore)],
      limit: 1000,
    })
  },
}
