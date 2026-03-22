// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Energy Project Pro Forma Generator
 *
 * Generates a 20-year financial projection for energy projects including:
 * - Capital costs + financing structure
 * - Revenue from energy sales (PPA or merchant)
 * - O&M costs with annual escalation
 * - IRA incentive impact
 * - LCOE calculation
 * - IRR, NPV, payback period
 */

export interface ProFormaInput {
  project_type: string
  capacity_mw: number
  capital_cost: number           // Total cost in USD
  annual_generation_mwh: number  // Expected annual generation
  ppa_price_per_mwh?: number     // PPA price (default: wholesale estimate)
  om_cost_per_mwh?: number       // O&M cost per MWh (default by type)
  debt_ratio?: number            // Debt/equity ratio (default: 0.70)
  interest_rate?: number         // Annual interest rate (default: 0.05)
  loan_term_years?: number       // Loan term (default: 15)
  ira_credit_value?: number      // From IRA calculator
  ira_annual_ptc?: number        // Annual PTC value
  escalation_rate?: number       // Annual revenue/cost escalation (default: 0.02)
  discount_rate?: number         // For NPV calculation (default: 0.08)
  project_life_years?: number    // default: 25
}

export interface ProFormaResult {
  /** Levelized Cost of Energy ($/MWh) */
  lcoe: number
  /** Internal Rate of Return (pre-tax) */
  irr: number
  /** Net Present Value at discount rate */
  npv: number
  /** Simple payback period in years */
  payback_years: number
  /** Year-by-year cash flows */
  annual_cashflows: AnnualCashflow[]
  /** Summary metrics */
  summary: {
    total_revenue: number
    total_om_cost: number
    total_debt_service: number
    total_incentives: number
    net_cash_flow_25yr: number
    capacity_factor: number
  }
  /** Sensitivity analysis */
  sensitivity: SensitivityResult[]
}

export interface AnnualCashflow {
  year: number
  revenue: number
  om_cost: number
  debt_service: number
  incentive: number
  net_cash_flow: number
  cumulative_cash_flow: number
}

export interface SensitivityResult {
  variable: string
  low_case: { value: number; irr: number }
  base_case: { value: number; irr: number }
  high_case: { value: number; irr: number }
}

// Default O&M costs by technology ($/MWh, 2024)
const OM_COSTS: Record<string, number> = {
  solar: 10, wind: 15, nuclear: 25, hydro: 12, geothermal: 18, storage: 8,
}

// Default wholesale prices by region ($/MWh, approximate)
const WHOLESALE_PRICE = 45 // US average

const CAPACITY_FACTORS: Record<string, number> = {
  solar: 0.25, wind: 0.35, nuclear: 0.92, hydro: 0.40, geothermal: 0.90, storage: 0.15,
}

export function generateProForma(input: ProFormaInput): ProFormaResult {
  const type = input.project_type.toLowerCase()
  const life = input.project_life_years || 25
  const escalation = input.escalation_rate || 0.02
  const discount = input.discount_rate || 0.08
  const debtRatio = input.debt_ratio || 0.70
  const interestRate = input.interest_rate || 0.05
  const loanTerm = input.loan_term_years || 15

  const ppaMwh = input.ppa_price_per_mwh || WHOLESALE_PRICE
  const omMwh = input.om_cost_per_mwh || OM_COSTS[type] || 15
  const capFactor = input.annual_generation_mwh / (input.capacity_mw * 8760)

  // Debt service (constant annual payment, amortizing)
  const debtAmount = input.capital_cost * debtRatio
  const equityAmount = input.capital_cost * (1 - debtRatio)
  const annualDebtService = loanTerm > 0
    ? debtAmount * (interestRate * Math.pow(1 + interestRate, loanTerm)) /
      (Math.pow(1 + interestRate, loanTerm) - 1)
    : 0

  // IRA incentive (ITC as year-0 equity offset, PTC as annual revenue)
  const itcValue = input.ira_credit_value || 0
  const annualPTC = input.ira_annual_ptc || 0

  // Generate annual cash flows
  const cashflows: AnnualCashflow[] = []
  let cumulative = -equityAmount + itcValue // Year 0: equity invested, ITC offset
  let totalRevenue = 0, totalOM = 0, totalDebt = 0, totalIncentive = itcValue

  for (let year = 1; year <= life; year++) {
    const escFactor = Math.pow(1 + escalation, year - 1)
    const revenue = input.annual_generation_mwh * ppaMwh * escFactor
    const omCost = input.annual_generation_mwh * omMwh * escFactor * 0.5 // O&M escalates slower
    const debtSvc = year <= loanTerm ? annualDebtService : 0
    const incentive = year <= 10 ? annualPTC : 0

    const netCash = revenue - omCost - debtSvc + incentive
    cumulative += netCash

    cashflows.push({
      year,
      revenue: Math.round(revenue),
      om_cost: Math.round(omCost),
      debt_service: Math.round(debtSvc),
      incentive: Math.round(incentive),
      net_cash_flow: Math.round(netCash),
      cumulative_cash_flow: Math.round(cumulative),
    })

    totalRevenue += revenue
    totalOM += omCost
    totalDebt += debtSvc
    totalIncentive += incentive
  }

  // Payback period
  let paybackYears = life
  for (const cf of cashflows) {
    if (cf.cumulative_cash_flow >= 0) {
      paybackYears = cf.year
      break
    }
  }

  // IRR (Newton-Raphson approximation)
  const irr = computeIRR(
    -equityAmount + itcValue,
    cashflows.map(c => c.net_cash_flow)
  )

  // NPV
  let npv = -equityAmount + itcValue
  for (let i = 0; i < cashflows.length; i++) {
    npv += cashflows[i].net_cash_flow / Math.pow(1 + discount, i + 1)
  }

  // LCOE = (Total lifecycle cost) / (Total generation)
  const totalGeneration = input.annual_generation_mwh * life
  const totalCost = input.capital_cost + totalOM - totalIncentive
  const lcoe = totalCost / totalGeneration

  // Sensitivity analysis
  const sensitivity: SensitivityResult[] = [
    computeSensitivity('PPA Price ($/MWh)', ppaMwh, 0.8, 1.2, input, lcoe, irr),
    computeSensitivity('Capital Cost ($M)', input.capital_cost / 1e6, 0.85, 1.15, input, lcoe, irr),
    computeSensitivity('Capacity Factor', capFactor, 0.85, 1.1, input, lcoe, irr),
  ]

  return {
    lcoe: Math.round(lcoe * 100) / 100,
    irr: Math.round(irr * 10000) / 10000,
    npv: Math.round(npv),
    payback_years: paybackYears,
    annual_cashflows: cashflows,
    summary: {
      total_revenue: Math.round(totalRevenue),
      total_om_cost: Math.round(totalOM),
      total_debt_service: Math.round(totalDebt),
      total_incentives: Math.round(totalIncentive),
      net_cash_flow_25yr: Math.round(cumulative),
      capacity_factor: Math.round(capFactor * 1000) / 1000,
    },
    sensitivity,
  }
}

function computeIRR(initialInvestment: number, cashFlows: number[]): number {
  let rate = 0.10 // initial guess
  for (let iter = 0; iter < 100; iter++) {
    let npv = initialInvestment
    let dnpv = 0
    for (let i = 0; i < cashFlows.length; i++) {
      const factor = Math.pow(1 + rate, i + 1)
      npv += cashFlows[i] / factor
      dnpv -= (i + 1) * cashFlows[i] / (factor * (1 + rate))
    }
    if (Math.abs(npv) < 1) break
    if (Math.abs(dnpv) < 0.001) break
    rate -= npv / dnpv
    rate = Math.max(-0.5, Math.min(2.0, rate)) // clamp
  }
  return rate
}

function computeSensitivity(
  name: string, baseValue: number, lowMult: number, highMult: number,
  input: ProFormaInput, baseLcoe: number, baseIrr: number
): SensitivityResult {
  // Simplified: approximate IRR change as proportional to the variable change
  const lowIrr = baseIrr * (name.includes('Cost') ? (2 - lowMult) : lowMult)
  const highIrr = baseIrr * (name.includes('Cost') ? (2 - highMult) : highMult)

  return {
    variable: name,
    low_case: { value: Math.round(baseValue * lowMult * 100) / 100, irr: Math.round(lowIrr * 10000) / 10000 },
    base_case: { value: Math.round(baseValue * 100) / 100, irr: Math.round(baseIrr * 10000) / 10000 },
    high_case: { value: Math.round(baseValue * highMult * 100) / 100, irr: Math.round(highIrr * 10000) / 10000 },
  }
}
