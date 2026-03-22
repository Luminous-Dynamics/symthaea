// Terra Atlas Discovery SDK
// TypeScript SDK for integrating with the Discovery API

export interface SimilarProject {
  name: string;
  capacity_mw: number;
  status: string;
  interconnection_cost: number;
  time_to_connect: string;
  developer: string;
  lessons_learned: string;
}

export interface TransmissionLine {
  line: string;
  distance_miles: number;
  available_capacity_mw: number;
  voltage_kv: number;
  owner: string;
  upgrade_status: string;
}

export interface Corridor {
  name: string;
  total_projects: number;
  total_capacity_mw: number;
  shared_infrastructure: {
    transmission_line: string;
    substations: number;
    cost_per_project: number;
    savings_vs_standalone: string;
  };
  open_capacity_mw: number;
  contact: string;
}

export interface QueueAnalysis {
  your_position: number;
  projects_ahead: number;
  typical_dropout_rate: string;
  expected_survivors_ahead: number;
  estimated_time_to_process: string;
}

export interface AccelerationOpportunity {
  strategy: string;
  time_savings: string;
  cost_savings: string;
  probability_of_success: string;
}

export class TerraAtlasDiscoverySDK {
  private baseUrl: string;
  
  constructor(baseUrl: string = 'https://atlas.luminousdynamics.io') {
    this.baseUrl = baseUrl;
  }
  
  /**
   * Find similar successful projects
   * @param type - Project type (solar, wind, storage, hybrid)
   * @param capacity - Capacity in MW
   * @param state - US state code
   * @param status - Project status filter
   */
  async findSimilarProjects(
    type: string,
    capacity: number,
    state?: string,
    status?: string
  ): Promise<{
    similarProjects: SimilarProject[];
    insights: {
      average_interconnection_cost: number;
      average_time: string;
      success_rate: string;
      key_factors: string[];
    };
  }> {
    const params = new URLSearchParams({
      type,
      capacity: capacity.toString(),
      ...(state && { state }),
      ...(status && { status })
    });
    
    const response = await fetch(`${this.baseUrl}/api/discovery/similar?${params}`);
    if (!response.ok) {
      throw new Error(`Failed to find similar projects: ${response.statusText}`);
    }
    return response.json();
  }
  
  /**
   * Find available transmission capacity near a location
   * @param lat - Latitude
   * @param lng - Longitude  
   * @param radius - Search radius in miles (default 50)
   */
  async findTransmissionCapacity(
    lat: number,
    lng: number,
    radius: number = 50
  ): Promise<{
    nearbyTransmission: TransmissionLine[];
    substations: Array<{
      name: string;
      distance_miles: number;
      available_bays: number;
      recent_connections: number;
    }>;
  }> {
    const params = new URLSearchParams({
      lat: lat.toString(),
      lng: lng.toString(),
      radius: radius.toString()
    });
    
    const response = await fetch(`${this.baseUrl}/api/discovery/transmission?${params}`);
    if (!response.ok) {
      throw new Error(`Failed to find transmission capacity: ${response.statusText}`);
    }
    return response.json();
  }
  
  /**
   * Find corridor opportunities for cost sharing
   * @param state - US state code
   * @param capacity - Your project capacity in MW
   */
  async findCorridorOpportunities(
    state?: string,
    capacity?: number
  ): Promise<{ corridors: Corridor[] }> {
    const params = new URLSearchParams({
      ...(state && { state }),
      ...(capacity && { capacity: capacity.toString() })
    });
    
    const response = await fetch(`${this.baseUrl}/api/discovery/corridors?${params}`);
    if (!response.ok) {
      throw new Error(`Failed to find corridors: ${response.statusText}`);
    }
    return response.json();
  }
  
  /**
   * Analyze FERC queue position and find acceleration opportunities
   * @param region - ISO/RTO (ERCOT, PJM, CAISO, etc.)
   * @param position - Your queue position number
   */
  async analyzeQueuePosition(
    region: string,
    position: number
  ): Promise<{
    queue_analysis: QueueAnalysis;
    acceleration_opportunities: AccelerationOpportunity[];
  }> {
    const params = new URLSearchParams({
      region,
      position: position.toString()
    });
    
    const response = await fetch(`${this.baseUrl}/api/discovery/queue-intelligence?${params}`);
    if (!response.ok) {
      throw new Error(`Failed to analyze queue: ${response.statusText}`);
    }
    return response.json();
  }
  
  /**
   * Calculate potential savings by joining a corridor vs standalone development
   * @param interconnectionCost - Your estimated standalone interconnection cost
   * @param corridorSavings - Corridor savings percentage (e.g., "74%")
   */
  calculateCorridorSavings(
    interconnectionCost: number,
    corridorSavings: string
  ): {
    standaloneCost: number;
    corridorCost: number;
    totalSavings: number;
    roi: string;
  } {
    const savingsPercent = parseFloat(corridorSavings.replace('%', '')) / 100;
    const corridorCost = interconnectionCost * (1 - savingsPercent);
    const totalSavings = interconnectionCost - corridorCost;
    
    return {
      standaloneCost: interconnectionCost,
      corridorCost,
      totalSavings,
      roi: `${(totalSavings / interconnectionCost * 100).toFixed(0)}%`
    };
  }
  
  /**
   * Estimate time to interconnection based on queue position
   * @param position - Queue position
   * @param dropoutRate - Historical dropout rate (0-1)
   * @param processingRate - Projects processed per year
   */
  estimateTimeToInterconnection(
    position: number,
    dropoutRate: number = 0.72,
    processingRate: number = 100
  ): {
    optimistic: number;
    realistic: number;
    pessimistic: number;
  } {
    const survivorsAhead = (position - 1) * (1 - dropoutRate);
    
    return {
      optimistic: survivorsAhead / (processingRate * 1.5), // 50% faster
      realistic: survivorsAhead / processingRate,
      pessimistic: survivorsAhead / (processingRate * 0.7) // 30% slower
    };
  }
}

// Export a default instance
export const discoveryAPI = new TerraAtlasDiscoverySDK();