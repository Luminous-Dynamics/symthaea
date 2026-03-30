// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Investment/Pledge System Types
 * Community-driven investment in energy projects
 */

export interface Pledge {
  id: string
  projectId: string
  userId?: string
  email: string
  amount: number  // USD
  status: 'pending' | 'confirmed' | 'cancelled' | 'fulfilled'
  createdAt: Date
  confirmedAt?: Date
  anonymous: boolean
  message?: string
}

export interface PledgeInput {
  projectId: string
  email: string
  amount: number
  anonymous?: boolean
  message?: string
}

export interface ProjectPledgeSummary {
  projectId: string
  totalPledged: number
  pledgeCount: number
  targetAmount: number
  percentFunded: number
  topPledgers: Array<{
    name: string
    amount: number
    date: Date
  }>
  recentActivity: Array<{
    amount: number
    date: Date
    anonymous: boolean
  }>
}

export interface PledgeValidation {
  valid: boolean
  errors: string[]
}

// Minimum pledge amount
export const MIN_PLEDGE_AMOUNT = 10  // $10 minimum

// Maximum single pledge (for fraud prevention)
export const MAX_PLEDGE_AMOUNT = 100000  // $100K max

export function validatePledge(input: PledgeInput): PledgeValidation {
  const errors: string[] = []

  if (!input.projectId) {
    errors.push('Project ID is required')
  }

  if (!input.email || !input.email.includes('@')) {
    errors.push('Valid email is required')
  }

  if (input.amount < MIN_PLEDGE_AMOUNT) {
    errors.push(`Minimum pledge is $${MIN_PLEDGE_AMOUNT}`)
  }

  if (input.amount > MAX_PLEDGE_AMOUNT) {
    errors.push(`Maximum pledge is $${MAX_PLEDGE_AMOUNT.toLocaleString()}`)
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

export function formatCurrency(amount: number): string {
  if (amount >= 1000000) {
    return `$${(amount / 1000000).toFixed(1)}M`
  }
  if (amount >= 1000) {
    return `$${(amount / 1000).toFixed(1)}K`
  }
  return `$${amount.toFixed(0)}`
}

export function calculateProjectTarget(capacityMw: number): number {
  // Estimate: $2.5M per MW for typical renewable projects
  return capacityMw * 2500000
}
