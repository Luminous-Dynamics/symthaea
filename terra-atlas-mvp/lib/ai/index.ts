// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Terra Atlas AI Intelligence Layer
 * The Sacred Seven analyzer system for intelligent site selection
 */

// Main service
export { IntelligenceService, getIntelligenceService } from './intelligence-service'

// Types
export type {
  ProjectType,
  ConfidenceLevel,
  AnalyzerResult,
  ScoringFactor,
  SiteAnalysis,
  InvestmentReadiness,
  Risk,
  Opportunity,
  SiteLocation,
  EnvironmentalFactors,
  GridData,
  ProjectInput,
  Analyzer,
  SearchOptions,
  PortfolioStats,
} from './types'

// Individual analyzers (for direct use if needed)
export {
  HeliosAnalyzer,
  AeolusAnalyzer,
  HydraAnalyzer,
  VulcanAnalyzer,
  PoseidonAnalyzer,
  ApolloAnalyzer,
  AtlasAnalyzer,
} from './analyzers'
