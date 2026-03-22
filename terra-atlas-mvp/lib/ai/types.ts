/**
 * Terra Atlas AI Intelligence Types
 * The Sacred Seven analyzer system for intelligent site selection
 */

// Project types supported by the platform
export type ProjectType =
  | 'solar'
  | 'wind'
  | 'hydro'
  | 'geothermal'
  | 'ocean'
  | 'nuclear'
  | 'storage'
  | 'hybrid'

// Analysis confidence levels
export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'unknown'

// Site analysis result from a single analyzer
export interface AnalyzerResult {
  score: number // 0-100
  confidence: ConfidenceLevel
  factors: ScoringFactor[]
  recommendations: string[]
  warnings: string[]
  metadata?: Record<string, unknown>
}

// Individual scoring factor
export interface ScoringFactor {
  name: string
  value: number // 0-100
  weight: number // 0-1
  description: string
  dataSource?: string
}

// Complete site analysis from all analyzers
export interface SiteAnalysis {
  projectId: string
  timestamp: Date
  overallScore: number
  viabilityRating: 'excellent' | 'good' | 'fair' | 'poor' | 'not_viable'
  analyses: {
    helios?: AnalyzerResult // Solar
    aeolus?: AnalyzerResult // Wind
    hydra?: AnalyzerResult // Hydro
    vulcan?: AnalyzerResult // Geothermal
    poseidon?: AnalyzerResult // Ocean
    apollo?: AnalyzerResult // Nuclear
    atlas?: AnalyzerResult // Grid/Storage
  }
  combinedRecommendations: string[]
  investmentReadiness: InvestmentReadiness
}

// Investment readiness assessment
export interface InvestmentReadiness {
  score: number // 0-100
  stage: 'early' | 'development' | 'permitted' | 'construction' | 'operational'
  estimatedTimeToRevenue: string // e.g., "18-24 months"
  keyRisks: Risk[]
  keyOpportunities: Opportunity[]
}

export interface Risk {
  category: 'technical' | 'regulatory' | 'financial' | 'environmental' | 'social'
  severity: 'high' | 'medium' | 'low'
  description: string
  mitigation?: string
}

export interface Opportunity {
  category: 'market' | 'policy' | 'technology' | 'location' | 'timing'
  impact: 'high' | 'medium' | 'low'
  description: string
}

// Geographic and environmental data
export interface SiteLocation {
  latitude: number
  longitude: number
  state: string
  county?: string
  nearestCity?: string
  elevation?: number // meters
  terrain?: 'flat' | 'hilly' | 'mountainous' | 'coastal' | 'offshore'
}

export interface EnvironmentalFactors {
  solarIrradiance?: number // kWh/m²/day
  windSpeed?: number // m/s average
  waterFlow?: number // m³/s
  geothermalGradient?: number // °C/km
  waveHeight?: number // meters
  tideRange?: number // meters
  seismicRisk?: 'high' | 'medium' | 'low'
  floodRisk?: 'high' | 'medium' | 'low'
}

// Grid and infrastructure data
export interface GridData {
  nearestSubstation?: {
    name: string
    distance: number // km
    capacity: number // MW
    availableCapacity?: number // MW
  }
  transmissionAccess?: 'direct' | 'nearby' | 'requires_extension'
  interconnectionQueue?: {
    position: number
    estimatedDate?: string
    status: string
  }
  curtailmentRisk?: 'high' | 'medium' | 'low'
}

// Project input for analysis
export interface ProjectInput {
  id: string
  name: string
  type: ProjectType
  location: SiteLocation
  capacityMw: number
  developer?: string
  status?: string
  environmental?: EnvironmentalFactors
  grid?: GridData
  metadata?: Record<string, unknown>
}

// Analyzer interface - all Sacred Seven analyzers implement this
export interface Analyzer {
  name: string
  code: string // HELIOS, AEOLUS, etc.
  description: string
  applicableTypes: ProjectType[]
  analyze(project: ProjectInput): Promise<AnalyzerResult>
}

// Search and filter options
export interface SearchOptions {
  query?: string
  types?: ProjectType[]
  states?: string[]
  minCapacity?: number
  maxCapacity?: number
  minScore?: number
  status?: string[]
  sortBy?: 'score' | 'capacity' | 'name' | 'date'
  sortOrder?: 'asc' | 'desc'
  limit?: number
  offset?: number
}

// Aggregated statistics
export interface PortfolioStats {
  totalProjects: number
  totalCapacityMw: number
  averageScore: number
  byType: Record<ProjectType, { count: number; capacityMw: number; avgScore: number }>
  byState: Record<string, { count: number; capacityMw: number }>
  byViability: Record<string, number>
}
