// Terra Atlas Database Schema using Drizzle ORM
// This replaces Prisma and works perfectly on NixOS

import { pgTable, uuid, varchar, text, integer, decimal, boolean, timestamp, jsonb, index, primaryKey, pgView } from 'drizzle-orm/pg-core'
import { relations } from 'drizzle-orm'

// =====================================================
// USER MANAGEMENT
// =====================================================

export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: varchar('email', { length: 255 }).unique().notNull(),
  username: varchar('username', { length: 100 }).unique().notNull(),
  passwordHash: varchar('password_hash', { length: 255 }).notNull(),
  fullName: varchar('full_name', { length: 255 }),
  avatarUrl: varchar('avatar_url', { length: 500 }),
  bio: text('bio'),
  
  // Reputation
  reputationScore: integer('reputation_score').default(0),
  validationsCount: integer('validations_count').default(0),
  accurateValidations: integer('accurate_validations').default(0),
  validationAccuracy: decimal('validation_accuracy', { precision: 5, scale: 2 }).default('0.00'),
  trustLevel: varchar('trust_level', { length: 50 }).default('novice'),
  
  // Status
  emailVerified: boolean('email_verified').default(false),
  isActive: boolean('is_active').default(true),
  isModerator: boolean('is_moderator').default(false),
  isAdmin: boolean('is_admin').default(false),
  
  // Timestamps
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
  lastLoginAt: timestamp('last_login_at'),
  
  // Preferences
  preferences: jsonb('preferences').default({}),
  notificationSettings: jsonb('notification_settings').default({
    email_validations: true,
    email_updates: true,
    push_alerts: false
  })
}, (table) => {
  return {
    emailIdx: index('idx_users_email').on(table.email),
    usernameIdx: index('idx_users_username').on(table.username),
    reputationIdx: index('idx_users_reputation').on(table.reputationScore),
    trustLevelIdx: index('idx_users_trust_level').on(table.trustLevel)
  }
})

// =====================================================
// DATA POINTS (Geographic without PostGIS)
// =====================================================

export const dataPoints = pgTable('data_points', {
  id: uuid('id').primaryKey().defaultRandom(),
  
  // Location
  latitude: decimal('latitude', { precision: 10, scale: 7 }).notNull(),
  longitude: decimal('longitude', { precision: 10, scale: 7 }).notNull(),
  
  // Data attributes
  dataType: varchar('data_type', { length: 50 }).notNull(),
  sourceId: varchar('source_id', { length: 100 }).notNull(),
  sourceName: varchar('source_name', { length: 255 }),
  
  // Details
  title: varchar('title', { length: 500 }),
  description: text('description'),
  properties: jsonb('properties').default({}),
  severity: varchar('severity', { length: 50 }),
  confidence: decimal('confidence', { precision: 5, scale: 2 }),
  
  // Trust metrics
  trustScore: decimal('trust_score', { precision: 5, scale: 2 }).default('50.00'),
  qualityScore: decimal('quality_score', { precision: 5, scale: 2 }).default('50.00'),
  confirmsCount: integer('confirms_count').default(0),
  disputesCount: integer('disputes_count').default(0),
  flagsCount: integer('flags_count').default(0),
  
  // Metadata
  observedAt: timestamp('observed_at'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
  expiresAt: timestamp('expires_at'),
  
  // Verification
  isVerified: boolean('is_verified').default(false),
  verifiedBy: uuid('verified_by').references(() => users.id),
  verifiedAt: timestamp('verified_at')
}, (table) => {
  return {
    locationIdx: index('idx_data_points_location').on(table.latitude, table.longitude),
    typeIdx: index('idx_data_points_type').on(table.dataType),
    sourceIdx: index('idx_data_points_source').on(table.sourceId),
    trustIdx: index('idx_data_points_trust').on(table.trustScore),
    createdIdx: index('idx_data_points_created').on(table.createdAt)
  }
})

// =====================================================
// SESSIONS (for authentication)
// =====================================================

export const sessions = pgTable('sessions', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id).notNull(),
  refreshTokenHash: varchar('refresh_token_hash', { length: 255 }).notNull(),
  
  ipAddress: varchar('ip_address', { length: 45 }),
  userAgent: text('user_agent'),
  
  expiresAt: timestamp('expires_at').notNull(),
  isActive: boolean('is_active').default(true),
  revokedAt: timestamp('revoked_at'),
  revokedReason: varchar('revoked_reason', { length: 255 }),
  
  createdAt: timestamp('created_at').defaultNow(),
  lastUsedAt: timestamp('last_used_at')
}, (table) => {
  return {
    userIdx: index('idx_sessions_user').on(table.userId),
    tokenIdx: index('idx_sessions_token').on(table.refreshTokenHash),
    expiresIdx: index('idx_sessions_expires').on(table.expiresAt)
  }
})

// =====================================================
// VALIDATIONS
// =====================================================

export const validations = pgTable('validations', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id),
  dataPointId: uuid('data_point_id').references(() => dataPoints.id),
  
  validationType: varchar('validation_type', { length: 20 }).notNull(),
  previousType: varchar('previous_type', { length: 20 }),
  comment: text('comment'),
  evidenceUrls: text('evidence_urls').array(),
  
  // Anonymous tracking
  anonymousId: varchar('anonymous_id', { length: 100 }),
  clientFingerprint: varchar('client_fingerprint', { length: 255 }),
  ipAddress: varchar('ip_address', { length: 45 }),
  userAgent: text('user_agent'),
  
  // Timestamps
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
}, (table) => {
  return {
    userIdx: index('idx_validations_user').on(table.userId),
    dataPointIdx: index('idx_validations_data_point').on(table.dataPointId),
    typeIdx: index('idx_validations_type').on(table.validationType),
    createdIdx: index('idx_validations_created').on(table.createdAt),
    uniqueUserValidation: index('unique_user_validation').on(table.userId, table.dataPointId)
  }
})

// =====================================================
// FUTURE TABLES (Ready to add as you scale)
// =====================================================

// Energy Projects
export const energyProjects = pgTable('energy_projects', {
  id: uuid('id').primaryKey().defaultRandom(),
  projectType: varchar('project_type', { length: 50 }).notNull(), // solar, wind, hydro, nuclear
  name: varchar('name', { length: 500 }).notNull(),
  capacity_mw: decimal('capacity_mw', { precision: 10, scale: 2 }),
  latitude: decimal('latitude', { precision: 10, scale: 7 }),
  longitude: decimal('longitude', { precision: 10, scale: 7 }),
  status: varchar('status', { length: 50 }),
  developer: varchar('developer', { length: 255 }),
  estimatedCost: decimal('estimated_cost', { precision: 15, scale: 2 }),
  completionDate: timestamp('completion_date'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
})

// Fire Events (example of environmental data table)
export const fireEvents = pgTable('fire_events', {
  id: uuid('id').primaryKey().defaultRandom(),
  latitude: decimal('latitude', { precision: 10, scale: 7 }).notNull(),
  longitude: decimal('longitude', { precision: 10, scale: 7 }).notNull(),
  detectedAt: timestamp('detected_at').notNull(),
  brightness: decimal('brightness', { precision: 8, scale: 2 }),
  confidence: decimal('confidence', { precision: 5, scale: 2 }),
  satelliteSource: varchar('satellite_source', { length: 100 }),
  affectedArea: decimal('affected_area_km2', { precision: 10, scale: 2 }),
  status: varchar('status', { length: 50 }), // active, contained, extinguished
  createdAt: timestamp('created_at').defaultNow()
})

// =====================================================
// RELATIONS (This makes queries beautiful)
// =====================================================

export const usersRelations = relations(users, ({ many }) => ({
  validations: many(validations),
  verifiedDataPoints: many(dataPoints)
}))

export const dataPointsRelations = relations(dataPoints, ({ one, many }) => ({
  verifier: one(users, {
    fields: [dataPoints.verifiedBy],
    references: [users.id]
  }),
  validations: many(validations)
}))

export const validationsRelations = relations(validations, ({ one }) => ({
  user: one(users, {
    fields: [validations.userId],
    references: [users.id]
  }),
  dataPoint: one(dataPoints, {
    fields: [validations.dataPointId],
    references: [dataPoints.id]
  })
}))