// Terra Atlas Energy Infrastructure Tables
// These are the core tables for tracking global energy projects

import { pgTable, uuid, varchar, text, integer, decimal, boolean, timestamp, jsonb, index, primaryKey } from 'drizzle-orm/pg-core'
import { relations } from 'drizzle-orm'
import { users } from './schema'

// =====================================================
// ENERGY PROJECTS - Core investment opportunities
// =====================================================

export const energyProjects = pgTable('energy_projects', {
  id: uuid('id').primaryKey().defaultRandom(),
  
  // Basic Info
  projectType: varchar('project_type', { length: 50 }).notNull(), // solar, wind, hydro, nuclear, storage, transmission
  subType: varchar('sub_type', { length: 100 }), // utility-scale, distributed, offshore, onshore, etc.
  name: varchar('name', { length: 500 }).notNull(),
  description: text('description'),
  
  // Location
  latitude: decimal('latitude', { precision: 10, scale: 7 }).notNull(),
  longitude: decimal('longitude', { precision: 10, scale: 7 }).notNull(),
  address: text('address'),
  city: varchar('city', { length: 255 }),
  state: varchar('state', { length: 100 }),
  country: varchar('country', { length: 100 }),
  zipCode: varchar('zip_code', { length: 20 }),
  
  // Capacity & Generation
  capacityMw: decimal('capacity_mw', { precision: 10, scale: 2 }),
  capacityMwh: decimal('capacity_mwh', { precision: 10, scale: 2 }), // For storage
  annualGenerationGwh: decimal('annual_generation_gwh', { precision: 12, scale: 2 }),
  capacityFactor: decimal('capacity_factor', { precision: 5, scale: 2 }),
  
  // Development Status
  status: varchar('status', { length: 50 }).notNull(), // planning, permitted, construction, operational, decommissioned
  developmentPhase: varchar('development_phase', { length: 100 }),
  codDate: timestamp('cod_date'), // Commercial Operation Date
  constructionStart: timestamp('construction_start'),
  permitDate: timestamp('permit_date'),
  ppaDate: timestamp('ppa_date'), // Power Purchase Agreement
  
  // Financial
  totalCostMillion: decimal('total_cost_million', { precision: 12, scale: 2 }),
  lcoePerMwh: decimal('lcoe_per_mwh', { precision: 8, scale: 2 }), // Levelized Cost of Energy
  ppaPrice: decimal('ppa_price', { precision: 8, scale: 2 }),
  expectedIrr: decimal('expected_irr', { precision: 5, scale: 2 }),
  
  // Ownership & Development
  developer: varchar('developer', { length: 500 }),
  owner: varchar('owner', { length: 500 }),
  operator: varchar('operator', { length: 500 }),
  epcContractor: varchar('epc_contractor', { length: 500 }),
  
  // Grid Connection
  interconnectionStatus: varchar('interconnection_status', { length: 100 }),
  interconnectionQueue: varchar('interconnection_queue', { length: 100 }),
  transmissionOwner: varchar('transmission_owner', { length: 255 }),
  substationName: varchar('substation_name', { length: 255 }),
  voltagekV: integer('voltage_kv'),
  
  // Environmental Impact
  co2AvoidedTonsYear: decimal('co2_avoided_tons_year', { precision: 12, scale: 2 }),
  landAreaAcres: decimal('land_area_acres', { precision: 10, scale: 2 }),
  waterUsageGallonsYear: decimal('water_usage_gallons_year', { precision: 15, scale: 0 }),
  
  // Employment
  constructionJobs: integer('construction_jobs'),
  permanentJobs: integer('permanent_jobs'),
  
  // Data Source
  dataSourceId: varchar('data_source_id', { length: 100 }),
  dataSourceName: varchar('data_source_name', { length: 255 }),
  externalId: varchar('external_id', { length: 255 }), // ID in source system
  
  // Metadata
  properties: jsonb('properties').default({}),
  tags: text('tags').array(),
  isPublic: boolean('is_public').default(true),
  isVerified: boolean('is_verified').default(false),
  verifiedBy: uuid('verified_by').references(() => users.id),
  verifiedAt: timestamp('verified_at'),
  
  // Trust & Quality
  trustScore: decimal('trust_score', { precision: 5, scale: 2 }).default('50.00'),
  dataQuality: decimal('data_quality', { precision: 5, scale: 2 }).default('50.00'),
  
  // Timestamps
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
  lastDataUpdate: timestamp('last_data_update')
}, (table) => {
  return {
    typeIdx: index('idx_energy_projects_type').on(table.projectType),
    statusIdx: index('idx_energy_projects_status').on(table.status),
    locationIdx: index('idx_energy_projects_location').on(table.latitude, table.longitude),
    capacityIdx: index('idx_energy_projects_capacity').on(table.capacityMw),
    developerIdx: index('idx_energy_projects_developer').on(table.developer),
    stateIdx: index('idx_energy_projects_state').on(table.state),
    countryIdx: index('idx_energy_projects_country').on(table.country)
  }
})

// =====================================================
// RENEWABLE ENERGY CERTIFICATES (RECs)
// =====================================================

export const renewableCertificates = pgTable('renewable_certificates', {
  id: uuid('id').primaryKey().defaultRandom(),
  projectId: uuid('project_id').references(() => energyProjects.id),
  
  certificateType: varchar('certificate_type', { length: 50 }).notNull(), // REC, SREC, I-REC, GO
  serialNumber: varchar('serial_number', { length: 255 }).unique(),
  vintage: varchar('vintage', { length: 20 }), // Year/Month of generation
  
  mwhGenerated: decimal('mwh_generated', { precision: 12, scale: 2 }),
  status: varchar('status', { length: 50 }), // active, retired, pending
  
  issuer: varchar('issuer', { length: 255 }),
  owner: varchar('owner', { length: 500 }),
  retiredBy: varchar('retired_by', { length: 500 }),
  retiredDate: timestamp('retired_date'),
  
  price: decimal('price', { precision: 10, scale: 2 }),
  currency: varchar('currency', { length: 10 }).default('USD'),
  
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
})

// =====================================================
// TRANSMISSION LINES
// =====================================================

export const transmissionLines = pgTable('transmission_lines', {
  id: uuid('id').primaryKey().defaultRandom(),
  
  name: varchar('name', { length: 500 }).notNull(),
  operator: varchar('operator', { length: 500 }),
  owner: varchar('owner', { length: 500 }),
  
  // Technical specs
  voltagekV: integer('voltage_kv').notNull(),
  lengthMiles: decimal('length_miles', { precision: 10, scale: 2 }),
  capacityMw: decimal('capacity_mw', { precision: 10, scale: 2 }),
  
  // Route (simplified - could be PostGIS linestring)
  startLat: decimal('start_lat', { precision: 10, scale: 7 }),
  startLng: decimal('start_lng', { precision: 10, scale: 7 }),
  endLat: decimal('end_lat', { precision: 10, scale: 7 }),
  endLng: decimal('end_lng', { precision: 10, scale: 7 }),
  waypoints: jsonb('waypoints').default([]), // Array of [lat, lng] points
  
  status: varchar('status', { length: 50 }),
  inServiceDate: timestamp('in_service_date'),
  
  estimatedCostMillion: decimal('estimated_cost_million', { precision: 12, scale: 2 }),
  
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
}, (table) => {
  return {
    voltageIdx: index('idx_transmission_voltage').on(table.voltagekV),
    operatorIdx: index('idx_transmission_operator').on(table.operator)
  }
})

// =====================================================
// POWER PURCHASE AGREEMENTS (PPAs)
// =====================================================

export const powerPurchaseAgreements = pgTable('power_purchase_agreements', {
  id: uuid('id').primaryKey().defaultRandom(),
  projectId: uuid('project_id').references(() => energyProjects.id),
  
  buyer: varchar('buyer', { length: 500 }).notNull(),
  seller: varchar('seller', { length: 500 }).notNull(),
  
  contractType: varchar('contract_type', { length: 50 }), // physical, virtual, synthetic
  
  capacityMw: decimal('capacity_mw', { precision: 10, scale: 2 }),
  annualGwh: decimal('annual_gwh', { precision: 12, scale: 2 }),
  
  pricePerMwh: decimal('price_per_mwh', { precision: 8, scale: 2 }),
  escalationRate: decimal('escalation_rate', { precision: 5, scale: 2 }),
  currency: varchar('currency', { length: 10 }).default('USD'),
  
  startDate: timestamp('start_date'),
  endDate: timestamp('end_date'),
  termYears: integer('term_years'),
  
  deliveryPoint: varchar('delivery_point', { length: 255 }),
  
  status: varchar('status', { length: 50 }), // negotiation, signed, active, expired
  
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
})

// =====================================================
// BATTERY STORAGE SYSTEMS
// =====================================================

export const batteryStorage = pgTable('battery_storage', {
  id: uuid('id').primaryKey().defaultRandom(),
  projectId: uuid('project_id').references(() => energyProjects.id),
  
  technology: varchar('technology', { length: 100 }), // lithium-ion, flow, lead-acid, etc.
  manufacturer: varchar('manufacturer', { length: 255 }),
  model: varchar('model', { length: 255 }),
  
  powerMw: decimal('power_mw', { precision: 10, scale: 2 }),
  energyMwh: decimal('energy_mwh', { precision: 10, scale: 2 }),
  duration: decimal('duration_hours', { precision: 5, scale: 2 }),
  
  roundTripEfficiency: decimal('round_trip_efficiency', { precision: 5, scale: 2 }),
  cyclesPerDay: decimal('cycles_per_day', { precision: 5, scale: 2 }),
  expectedLifetimeCycles: integer('expected_lifetime_cycles'),
  
  application: varchar('application', { length: 100 }), // frequency regulation, peak shaving, backup
  gridConnection: varchar('grid_connection', { length: 100 }), // front-of-meter, behind-meter
  
  warrantyYears: integer('warranty_years'),
  degradationRateYear: decimal('degradation_rate_year', { precision: 5, scale: 2 }),
  
  installationDate: timestamp('installation_date'),
  
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
})

// =====================================================
// RELATIONS
// =====================================================

export const energyProjectsRelations = relations(energyProjects, ({ one, many }) => ({
  verifier: one(users, {
    fields: [energyProjects.verifiedBy],
    references: [users.id]
  }),
  certificates: many(renewableCertificates),
  ppas: many(powerPurchaseAgreements),
  storage: many(batteryStorage)
}))

export const renewableCertificatesRelations = relations(renewableCertificates, ({ one }) => ({
  project: one(energyProjects, {
    fields: [renewableCertificates.projectId],
    references: [energyProjects.id]
  })
}))

export const powerPurchaseAgreementsRelations = relations(powerPurchaseAgreements, ({ one }) => ({
  project: one(energyProjects, {
    fields: [powerPurchaseAgreements.projectId],
    references: [energyProjects.id]
  })
}))

export const batteryStorageRelations = relations(batteryStorage, ({ one }) => ({
  project: one(energyProjects, {
    fields: [batteryStorage.projectId],
    references: [energyProjects.id]
  })
}))