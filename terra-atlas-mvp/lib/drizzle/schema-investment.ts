// Investment & Pledge System Schema
// Enables users to pledge investments in energy projects

import { pgTable, uuid, varchar, decimal, timestamp, integer, boolean, json, text, pgEnum } from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';
import { users } from './schema';
import { energyProjects } from './schema-energy';

// Investment status enum
export const investmentStatusEnum = pgEnum('investment_status', [
  'pledged',      // Initial pledge, not yet committed
  'pending',      // Awaiting payment processing
  'confirmed',    // Payment received, investment active
  'cancelled',    // Cancelled by user
  'refunded',     // Refunded to investor
  'matured'       // Investment reached maturity
]);

// Investment type enum
export const investmentTypeEnum = pgEnum('investment_type', [
  'equity',       // Ownership stake
  'debt',         // Fixed return loan
  'revenue_share', // Share of project revenues
  'crowdfunding', // Small investor participation
  'green_bond',   // Environmental impact bond
  'ppa'           // Power Purchase Agreement
]);

// User Portfolios - Track user's investment portfolio
export const portfolios = pgTable('portfolios', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id).notNull(),
  
  // Portfolio metrics
  totalInvested: decimal('total_invested', { precision: 15, scale: 2 }).default('0'),
  totalReturns: decimal('total_returns', { precision: 15, scale: 2 }).default('0'),
  totalPledged: decimal('total_pledged', { precision: 15, scale: 2 }).default('0'),
  activeInvestments: integer('active_investments').default(0),
  
  // Performance metrics
  portfolioIrr: decimal('portfolio_irr', { precision: 5, scale: 2 }), // Internal Rate of Return
  totalCo2Avoided: decimal('total_co2_avoided', { precision: 12, scale: 2 }).default('0'),
  totalMwhGenerated: decimal('total_mwh_generated', { precision: 12, scale: 2 }).default('0'),
  
  // Risk profile
  riskTolerance: varchar('risk_tolerance', { length: 20 }).default('moderate'), // low, moderate, high
  investmentGoals: json('investment_goals').$type<string[]>(),
  
  // Preferences
  preferredTechnologies: json('preferred_technologies').$type<string[]>(),
  preferredRegions: json('preferred_regions').$type<string[]>(),
  minInvestmentSize: decimal('min_investment_size', { precision: 10, scale: 2 }),
  maxInvestmentSize: decimal('max_investment_size', { precision: 10, scale: 2 }),
  
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Investment Pledges - Track investment commitments
export const investments = pgTable('investments', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id).notNull(),
  portfolioId: uuid('portfolio_id').references(() => portfolios.id),
  projectId: uuid('project_id').references(() => energyProjects.id).notNull(),
  
  // Investment details
  investmentType: investmentTypeEnum('investment_type').notNull(),
  amountUsd: decimal('amount_usd', { precision: 12, scale: 2 }).notNull(),
  status: investmentStatusEnum('status').notNull().default('pledged'),
  
  // Terms
  expectedReturn: decimal('expected_return', { precision: 5, scale: 2 }), // Percentage
  investmentPeriodYears: decimal('investment_period_years', { precision: 4, scale: 1 }),
  minimumHoldYears: decimal('minimum_hold_years', { precision: 4, scale: 1 }),
  
  // Dates
  pledgeDate: timestamp('pledge_date').defaultNow(),
  confirmationDate: timestamp('confirmation_date'),
  maturityDate: timestamp('maturity_date'),
  
  // Returns tracking
  totalReturns: decimal('total_returns', { precision: 12, scale: 2 }).default('0'),
  lastDistribution: timestamp('last_distribution'),
  distributionFrequency: varchar('distribution_frequency', { length: 20 }), // monthly, quarterly, annually
  
  // Impact metrics
  co2AvoidedTons: decimal('co2_avoided_tons', { precision: 10, scale: 2 }),
  mwhGenerated: decimal('mwh_generated', { precision: 10, scale: 2 }),
  homesEquivalent: integer('homes_equivalent'),
  
  // Documents
  agreementUrl: text('agreement_url'),
  prospectusUrl: text('prospectus_url'),
  
  // Additional data
  notes: text('notes'),
  metadata: json('metadata'),
  
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Investment Returns - Track actual returns/distributions
export const investmentReturns = pgTable('investment_returns', {
  id: uuid('id').primaryKey().defaultRandom(),
  investmentId: uuid('investment_id').references(() => investments.id).notNull(),
  
  // Return details
  returnType: varchar('return_type', { length: 50 }).notNull(), // dividend, interest, capital_gain, revenue_share
  amountUsd: decimal('amount_usd', { precision: 12, scale: 2 }).notNull(),
  returnDate: timestamp('return_date').notNull(),
  
  // Period covered
  periodStart: timestamp('period_start'),
  periodEnd: timestamp('period_end'),
  
  // Project performance for this period
  projectRevenue: decimal('project_revenue', { precision: 15, scale: 2 }),
  projectProfit: decimal('project_profit', { precision: 15, scale: 2 }),
  capacityFactor: decimal('capacity_factor', { precision: 5, scale: 2 }),
  
  // Tax information
  taxWithheld: decimal('tax_withheld', { precision: 10, scale: 2 }),
  taxRate: decimal('tax_rate', { precision: 5, scale: 2 }),
  
  // Documentation
  statementUrl: text('statement_url'),
  
  createdAt: timestamp('created_at').defaultNow()
});

// Investment Opportunities - Curated investment opportunities
export const investmentOpportunities = pgTable('investment_opportunities', {
  id: uuid('id').primaryKey().defaultRandom(),
  projectId: uuid('project_id').references(() => energyProjects.id).notNull(),
  
  // Opportunity details
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description').notNull(),
  opportunityType: investmentTypeEnum('opportunity_type').notNull(),
  
  // Investment parameters
  minimumInvestment: decimal('minimum_investment', { precision: 10, scale: 2 }).notNull(),
  maximumInvestment: decimal('maximum_investment', { precision: 10, scale: 2 }),
  targetRaise: decimal('target_raise', { precision: 12, scale: 2 }).notNull(),
  amountRaised: decimal('amount_raised', { precision: 12, scale: 2 }).default('0'),
  
  // Terms
  expectedIrr: decimal('expected_irr', { precision: 5, scale: 2 }).notNull(),
  investmentPeriod: varchar('investment_period', { length: 50 }).notNull(),
  distributionSchedule: varchar('distribution_schedule', { length: 100 }),
  
  // Status
  isActive: boolean('is_active').default(true),
  isFeatured: boolean('is_featured').default(false),
  openDate: timestamp('open_date').notNull(),
  closeDate: timestamp('close_date'),
  
  // Risk assessment
  riskLevel: varchar('risk_level', { length: 20 }).notNull(), // low, medium, high
  riskFactors: json('risk_factors').$type<string[]>(),
  
  // Benefits
  keyBenefits: json('key_benefits').$type<string[]>(),
  taxAdvantages: json('tax_advantages').$type<string[]>(),
  
  // Documentation
  prospectusUrl: text('prospectus_url'),
  termSheetUrl: text('term_sheet_url'),
  financialModelUrl: text('financial_model_url'),
  
  // Investor counts
  totalInvestors: integer('total_investors').default(0),
  accreditedRequired: boolean('accredited_required').default(false),
  
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Watchlist - Projects users are watching
export const watchlist = pgTable('watchlist', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id).notNull(),
  projectId: uuid('project_id').references(() => energyProjects.id).notNull(),
  
  // Notification preferences
  notifyOnOpportunity: boolean('notify_on_opportunity').default(true),
  notifyOnMilestone: boolean('notify_on_milestone').default(true),
  notifyOnPriceChange: boolean('notify_on_price_change').default(false),
  
  // User notes
  notes: text('notes'),
  
  addedAt: timestamp('added_at').defaultNow()
});

// Relations
export const portfolioRelations = relations(portfolios, ({ one, many }) => ({
  user: one(users, {
    fields: [portfolios.userId],
    references: [users.id]
  }),
  investments: many(investments)
}));

export const investmentRelations = relations(investments, ({ one, many }) => ({
  user: one(users, {
    fields: [investments.userId],
    references: [users.id]
  }),
  portfolio: one(portfolios, {
    fields: [investments.portfolioId],
    references: [portfolios.id]
  }),
  project: one(energyProjects, {
    fields: [investments.projectId],
    references: [energyProjects.id]
  }),
  returns: many(investmentReturns)
}));

export const investmentReturnRelations = relations(investmentReturns, ({ one }) => ({
  investment: one(investments, {
    fields: [investmentReturns.investmentId],
    references: [investments.id]
  })
}));

export const investmentOpportunityRelations = relations(investmentOpportunities, ({ one }) => ({
  project: one(energyProjects, {
    fields: [investmentOpportunities.projectId],
    references: [energyProjects.id]
  })
}));

export const watchlistRelations = relations(watchlist, ({ one }) => ({
  user: one(users, {
    fields: [watchlist.userId],
    references: [users.id]
  }),
  project: one(energyProjects, {
    fields: [watchlist.projectId],
    references: [energyProjects.id]
  })
}));