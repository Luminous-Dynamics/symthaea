/**
 * Water ↔ Energy Integration Module
 *
 * Cross-domain bridge connecting water systems and energy grids for:
 * - Pumping station energy management
 * - Hydroelectric coordination
 * - Water treatment energy optimization
 * - Desalination plant energy scheduling
 * - Drought/flood energy grid impact management
 * - Renewable energy ↔ water storage correlation
 *
 * @module @mycelix/sdk/integrations/water-energy
 */

import type { ActionHash, AgentPubKey } from '@holochain/client';

// ============================================================================
// Water System Types
// ============================================================================

/**
 * Water source type
 */
export type WaterSourceType =
  | 'Reservoir'
  | 'Groundwater'
  | 'River'
  | 'Lake'
  | 'Desalination'
  | 'Recycled'
  | 'Rainwater'
  | 'Spring';

/**
 * Water quality classification
 */
export type WaterQuality =
  | 'Potable' // Safe for drinking
  | 'Agricultural' // Safe for irrigation
  | 'Industrial' // Safe for industrial use
  | 'Recreational' // Safe for swimming/recreation
  | 'NonPotable' // Not safe for consumption
  | 'Contaminated'; // Requires treatment

/**
 * Water system operational status
 */
export type WaterSystemStatus =
  | 'Normal'
  | 'LowCapacity'
  | 'HighDemand'
  | 'Maintenance'
  | 'Emergency'
  | 'Offline';

/**
 * Water facility types that consume energy
 */
export type WaterFacilityType =
  | 'PumpingStation'
  | 'TreatmentPlant'
  | 'DesalinationPlant'
  | 'DistributionCenter'
  | 'StorageFacility'
  | 'WastewaterTreatment'
  | 'HydroelectricDam'
  | 'Reservoir';

// ============================================================================
// Energy Grid Types
// ============================================================================

/**
 * Energy generation source
 */
export type EnergySourceType =
  | 'Solar'
  | 'Wind'
  | 'Hydro'
  | 'Nuclear'
  | 'NaturalGas'
  | 'Coal'
  | 'Geothermal'
  | 'Biomass'
  | 'Battery'
  | 'Grid';

/**
 * Grid load category
 */
export type GridLoadCategory =
  | 'CriticalInfrastructure'
  | 'Residential'
  | 'Commercial'
  | 'Industrial'
  | 'Agricultural'
  | 'Municipal';

/**
 * Time-of-use pricing period
 */
export type PricingPeriod = 'OffPeak' | 'MidPeak' | 'OnPeak' | 'Critical';

// ============================================================================
// Cross-Domain Types
// ============================================================================

/**
 * Water facility with energy profile
 */
export interface WaterEnergyFacility {
  facilityId: string;
  name: string;
  facilityType: WaterFacilityType;
  location: GeoLocation;
  /** Current operational status */
  status: WaterSystemStatus;
  /** Connected energy sources */
  energySources: EnergyConnection[];
  /** Average power consumption (kW) */
  avgPowerConsumption: number;
  /** Peak power consumption (kW) */
  peakPowerConsumption: number;
  /** Whether facility can generate power (e.g., hydroelectric) */
  canGeneratePower: boolean;
  /** Power generation capacity if applicable (kW) */
  generationCapacity?: number;
  /** Water throughput capacity (cubic meters/hour) */
  waterThroughput: number;
  /** Energy efficiency rating (0-1) */
  efficiencyRating: number;
  /** Whether facility supports demand response */
  demandResponseCapable: boolean;
  /** Minimum operational power threshold */
  minOperationalPower: number;
  /** On-chain reference */
  actionHash?: ActionHash;
}

export interface GeoLocation {
  latitude: number;
  longitude: number;
  elevation?: number;
}

export interface EnergyConnection {
  connectionId: string;
  sourceType: EnergySourceType;
  provider: string;
  capacityKw: number;
  currentLoadKw: number;
  gridPriority: GridLoadCategory;
  isBackup: boolean;
}

/**
 * Energy consumption record for water operations
 */
export interface WaterEnergyConsumption {
  recordId: string;
  facilityId: string;
  timestamp: number;
  durationMinutes: number;
  /** Energy consumed (kWh) */
  energyConsumed: number;
  /** Water processed (cubic meters) */
  waterProcessed: number;
  /** Energy efficiency (kWh per cubic meter) */
  efficiency: number;
  /** Time-of-use pricing period */
  pricingPeriod: PricingPeriod;
  /** Energy cost */
  cost: number;
  /** Carbon emissions (kg CO2) */
  carbonEmissions: number;
  /** Source breakdown */
  sourceBreakdown: EnergySourceBreakdown[];
}

export interface EnergySourceBreakdown {
  source: EnergySourceType;
  percentage: number;
  carbonIntensity: number; // kg CO2 per kWh
}

/**
 * Demand response event
 */
export interface DemandResponseEvent {
  eventId: string;
  eventType: DemandResponseType;
  triggerTime: number;
  duration: number;
  /** Requested load reduction (kW) */
  requestedReduction: number;
  /** Facilities participating */
  participatingFacilities: string[];
  /** Actual load reduction achieved (kW) */
  actualReduction?: number;
  /** Compensation rate */
  compensationRate?: number;
  status: DemandResponseStatus;
}

export type DemandResponseType =
  | 'VoluntaryReduction'
  | 'EmergencyCurtailment'
  | 'PeakShaving'
  | 'FrequencyRegulation'
  | 'LoadShifting';

export type DemandResponseStatus =
  | 'Scheduled'
  | 'Active'
  | 'Completed'
  | 'Cancelled'
  | 'Failed';

/**
 * Hydroelectric generation record
 */
export interface HydroGenerationRecord {
  recordId: string;
  facilityId: string;
  timestamp: number;
  durationMinutes: number;
  /** Power generated (kWh) */
  energyGenerated: number;
  /** Water flow (cubic meters) */
  waterFlow: number;
  /** Reservoir level before */
  reservoirLevelBefore: number;
  /** Reservoir level after */
  reservoirLevelAfter: number;
  /** Grid sell price */
  gridSellPrice: number;
  /** Revenue generated */
  revenue: number;
}

/**
 * Water-energy optimization schedule
 */
export interface OptimizationSchedule {
  scheduleId: string;
  facilityId: string;
  validFrom: number;
  validTo: number;
  /** Hourly power targets */
  hourlyTargets: HourlyTarget[];
  /** Expected savings */
  expectedSavings: number;
  /** Expected carbon reduction (kg CO2) */
  expectedCarbonReduction: number;
  status: ScheduleStatus;
}

export interface HourlyTarget {
  hour: number; // 0-23
  targetPowerKw: number;
  targetWaterThroughput: number;
  pricingPeriod: PricingPeriod;
  expectedCost: number;
}

export type ScheduleStatus = 'Draft' | 'Active' | 'Completed' | 'Suspended';

/**
 * Cross-domain alert
 */
export interface WaterEnergyAlert {
  alertId: string;
  alertType: AlertType;
  severity: AlertSeverity;
  facilityId: string;
  message: string;
  timestamp: number;
  /** Related metrics */
  metrics: AlertMetrics;
  /** Recommended actions */
  recommendations: string[];
  acknowledged: boolean;
  resolvedAt?: number;
}

export type AlertType =
  | 'PowerOutage'
  | 'HighEnergyConsumption'
  | 'LowWaterPressure'
  | 'EquipmentFailure'
  | 'GridInstability'
  | 'DroughtWarning'
  | 'FloodWarning'
  | 'MaintenanceRequired'
  | 'CostThresholdExceeded';

export type AlertSeverity = 'Info' | 'Warning' | 'Critical' | 'Emergency';

export interface AlertMetrics {
  currentValue: number;
  threshold: number;
  unit: string;
  trend: 'Rising' | 'Falling' | 'Stable';
}

// ============================================================================
// Reputation Types
// ============================================================================

/**
 * Cross-domain reputation for water-energy stewardship
 */
export interface WaterEnergyReputation {
  agentPubKey: AgentPubKey;
  waterDomainScore: number;
  energyDomainScore: number;
  combinedScore: number;
  efficiencyContributions: number;
  demandResponseParticipation: number;
  sustainabilityRating: SustainabilityRating;
  lastUpdated: number;
}

export type SustainabilityRating = 'Exemplary' | 'Good' | 'Average' | 'NeedsImprovement' | 'Poor';

// ============================================================================
// Configuration
// ============================================================================

export interface WaterEnergyBridgeConfig {
  /** Cache TTL in ms */
  cacheTtlMs?: number;
  /** Default optimization horizon (hours) */
  optimizationHorizon?: number;
  /** Carbon intensity defaults by source */
  carbonIntensityDefaults?: Partial<Record<EnergySourceType, number>>;
  /** Pricing period schedule */
  pricingSchedule?: PricingSchedule;
}

export interface PricingSchedule {
  timezone: string;
  offPeakHours: number[];
  midPeakHours: number[];
  onPeakHours: number[];
  criticalHours: number[];
}

const DEFAULT_CARBON_INTENSITY: Record<EnergySourceType, number> = {
  Solar: 0.02,
  Wind: 0.01,
  Hydro: 0.02,
  Nuclear: 0.01,
  NaturalGas: 0.45,
  Coal: 0.95,
  Geothermal: 0.04,
  Biomass: 0.23,
  Battery: 0.05,
  Grid: 0.40,
};

// ============================================================================
// Water-Energy Bridge Service
// ============================================================================

/**
 * Bridge service connecting water and energy systems
 */
export class WaterEnergyBridge {
  private config: Required<WaterEnergyBridgeConfig>;
  private facilityCache: Map<string, WaterEnergyFacility> = new Map();
  private consumptionHistory: Map<string, WaterEnergyConsumption[]> = new Map();
  private scheduleCache: Map<string, OptimizationSchedule> = new Map();
  private alertCache: Map<string, WaterEnergyAlert[]> = new Map();

  constructor(config: WaterEnergyBridgeConfig = {}) {
    this.config = {
      cacheTtlMs: config.cacheTtlMs ?? 5 * 60 * 1000,
      optimizationHorizon: config.optimizationHorizon ?? 24,
      carbonIntensityDefaults: { ...DEFAULT_CARBON_INTENSITY, ...config.carbonIntensityDefaults },
      pricingSchedule: config.pricingSchedule ?? {
        timezone: 'UTC',
        offPeakHours: [0, 1, 2, 3, 4, 5, 22, 23],
        midPeakHours: [6, 7, 8, 9, 19, 20, 21],
        onPeakHours: [10, 11, 12, 13, 14, 15, 16, 17, 18],
        criticalHours: [],
      },
    };
  }

  // --------------------------------------------------------------------------
  // Facility Management
  // --------------------------------------------------------------------------

  /**
   * Register a water-energy facility
   */
  async registerFacility(facility: Omit<WaterEnergyFacility, 'facilityId'>): Promise<WaterEnergyFacility> {
    const facilityId = `wef-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const registered: WaterEnergyFacility = {
      facilityId,
      ...facility,
    };
    this.facilityCache.set(facilityId, registered);
    this.consumptionHistory.set(facilityId, []);
    return registered;
  }

  /**
   * Get facility by ID
   */
  getFacility(facilityId: string): WaterEnergyFacility | undefined {
    return this.facilityCache.get(facilityId);
  }

  /**
   * List all facilities
   */
  listFacilities(filter?: {
    facilityType?: WaterFacilityType;
    status?: WaterSystemStatus;
    demandResponseCapable?: boolean;
  }): WaterEnergyFacility[] {
    let facilities = Array.from(this.facilityCache.values());

    if (filter?.facilityType) {
      facilities = facilities.filter((f) => f.facilityType === filter.facilityType);
    }
    if (filter?.status) {
      facilities = facilities.filter((f) => f.status === filter.status);
    }
    if (filter?.demandResponseCapable !== undefined) {
      facilities = facilities.filter((f) => f.demandResponseCapable === filter.demandResponseCapable);
    }

    return facilities;
  }

  /**
   * Update facility status
   */
  updateFacilityStatus(facilityId: string, status: WaterSystemStatus): WaterEnergyFacility | null {
    const facility = this.facilityCache.get(facilityId);
    if (!facility) return null;

    facility.status = status;
    return facility;
  }

  // --------------------------------------------------------------------------
  // Energy Consumption Tracking
  // --------------------------------------------------------------------------

  /**
   * Record energy consumption
   */
  recordConsumption(record: Omit<WaterEnergyConsumption, 'recordId' | 'efficiency' | 'pricingPeriod'>): WaterEnergyConsumption {
    const hour = new Date(record.timestamp).getUTCHours();
    const pricingPeriod = this.getPricingPeriod(hour);
    const efficiency = record.waterProcessed > 0 ? record.energyConsumed / record.waterProcessed : 0;

    const consumption: WaterEnergyConsumption = {
      recordId: `wec-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      ...record,
      efficiency,
      pricingPeriod,
    };

    const history = this.consumptionHistory.get(record.facilityId) ?? [];
    history.push(consumption);
    this.consumptionHistory.set(record.facilityId, history);

    return consumption;
  }

  /**
   * Get consumption history for a facility
   */
  getConsumptionHistory(facilityId: string, since?: number): WaterEnergyConsumption[] {
    const history = this.consumptionHistory.get(facilityId) ?? [];
    if (since) {
      return history.filter((c) => c.timestamp >= since);
    }
    return history;
  }

  /**
   * Calculate total consumption statistics
   */
  calculateConsumptionStats(facilityId: string, since: number): ConsumptionStats {
    const history = this.getConsumptionHistory(facilityId, since);

    if (history.length === 0) {
      return {
        totalEnergyKwh: 0,
        totalWaterCubicMeters: 0,
        avgEfficiency: 0,
        totalCost: 0,
        totalCarbonKg: 0,
        periodBreakdown: {
          offPeak: 0,
          midPeak: 0,
          onPeak: 0,
          critical: 0,
        },
      };
    }

    const stats = history.reduce(
      (acc, c) => ({
        totalEnergyKwh: acc.totalEnergyKwh + c.energyConsumed,
        totalWaterCubicMeters: acc.totalWaterCubicMeters + c.waterProcessed,
        totalCost: acc.totalCost + c.cost,
        totalCarbonKg: acc.totalCarbonKg + c.carbonEmissions,
        periodBreakdown: {
          offPeak: acc.periodBreakdown.offPeak + (c.pricingPeriod === 'OffPeak' ? c.energyConsumed : 0),
          midPeak: acc.periodBreakdown.midPeak + (c.pricingPeriod === 'MidPeak' ? c.energyConsumed : 0),
          onPeak: acc.periodBreakdown.onPeak + (c.pricingPeriod === 'OnPeak' ? c.energyConsumed : 0),
          critical: acc.periodBreakdown.critical + (c.pricingPeriod === 'Critical' ? c.energyConsumed : 0),
        },
      }),
      {
        totalEnergyKwh: 0,
        totalWaterCubicMeters: 0,
        totalCost: 0,
        totalCarbonKg: 0,
        periodBreakdown: { offPeak: 0, midPeak: 0, onPeak: 0, critical: 0 },
      }
    );

    return {
      ...stats,
      avgEfficiency: stats.totalWaterCubicMeters > 0
        ? stats.totalEnergyKwh / stats.totalWaterCubicMeters
        : 0,
    };
  }

  // --------------------------------------------------------------------------
  // Optimization
  // --------------------------------------------------------------------------

  /**
   * Generate optimization schedule for a facility
   */
  generateOptimizationSchedule(
    facilityId: string,
    options: {
      prioritizeCost?: boolean;
      prioritizeCarbon?: boolean;
      minWaterThroughput?: number;
    } = {}
  ): OptimizationSchedule {
    const facility = this.facilityCache.get(facilityId);
    if (!facility) {
      throw new Error(`Facility not found: ${facilityId}`);
    }

    const now = Date.now();
    const hourlyTargets: HourlyTarget[] = [];

    // Generate 24-hour schedule
    for (let hour = 0; hour < 24; hour++) {
      const pricingPeriod = this.getPricingPeriod(hour);
      const costMultiplier = this.getPricingMultiplier(pricingPeriod);
      const carbonMultiplier = this.getCarbonMultiplier(hour);

      // Calculate optimal power based on priorities
      let targetPower = facility.avgPowerConsumption;
      let targetThroughput = facility.waterThroughput;

      if (options.prioritizeCost) {
        // Reduce load during expensive periods
        if (pricingPeriod === 'OnPeak' || pricingPeriod === 'Critical') {
          targetPower *= 0.7;
          targetThroughput *= 0.7;
        } else if (pricingPeriod === 'OffPeak') {
          targetPower *= 1.2;
          targetThroughput *= 1.2;
        }
      }

      if (options.prioritizeCarbon) {
        // Adjust based on grid carbon intensity (lower during high renewable hours)
        targetPower *= carbonMultiplier;
        targetThroughput *= carbonMultiplier;
      }

      // Ensure minimum throughput
      if (options.minWaterThroughput && targetThroughput < options.minWaterThroughput) {
        const ratio = options.minWaterThroughput / targetThroughput;
        targetPower *= ratio;
        targetThroughput = options.minWaterThroughput;
      }

      // Cap at facility limits
      targetPower = Math.min(targetPower, facility.peakPowerConsumption);
      targetThroughput = Math.min(targetThroughput, facility.waterThroughput * 1.2);

      hourlyTargets.push({
        hour,
        targetPowerKw: targetPower,
        targetWaterThroughput: targetThroughput,
        pricingPeriod,
        expectedCost: targetPower * costMultiplier,
      });
    }

    const schedule: OptimizationSchedule = {
      scheduleId: `sch-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      facilityId,
      validFrom: now,
      validTo: now + this.config.optimizationHorizon * 60 * 60 * 1000,
      hourlyTargets,
      expectedSavings: this.calculateExpectedSavings(facility, hourlyTargets),
      expectedCarbonReduction: this.calculateExpectedCarbonReduction(facility, hourlyTargets),
      status: 'Draft',
    };

    this.scheduleCache.set(schedule.scheduleId, schedule);
    return schedule;
  }

  /**
   * Activate an optimization schedule
   */
  activateSchedule(scheduleId: string): OptimizationSchedule | null {
    const schedule = this.scheduleCache.get(scheduleId);
    if (!schedule || schedule.status !== 'Draft') return null;

    schedule.status = 'Active';
    return schedule;
  }

  // --------------------------------------------------------------------------
  // Demand Response
  // --------------------------------------------------------------------------

  /**
   * Create a demand response event
   */
  createDemandResponseEvent(
    eventType: DemandResponseType,
    requestedReduction: number,
    duration: number,
    facilityIds: string[]
  ): DemandResponseEvent {
    const event: DemandResponseEvent = {
      eventId: `dre-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      eventType,
      triggerTime: Date.now(),
      duration,
      requestedReduction,
      participatingFacilities: facilityIds,
      status: 'Scheduled',
    };

    return event;
  }

  /**
   * Calculate potential demand response capacity
   */
  calculateDemandResponseCapacity(): DemandResponseCapacity {
    const facilities = this.listFacilities({ demandResponseCapable: true, status: 'Normal' });

    let totalCapacity = 0;
    let immediateCapacity = 0;
    const byFacilityType: Record<string, number> = {};

    for (const facility of facilities) {
      const flexibility = facility.peakPowerConsumption - facility.minOperationalPower;
      totalCapacity += flexibility;
      immediateCapacity += flexibility * 0.5; // Assume 50% can respond immediately

      const type = facility.facilityType;
      byFacilityType[type] = (byFacilityType[type] ?? 0) + flexibility;
    }

    return {
      totalCapacityKw: totalCapacity,
      immediateCapacityKw: immediateCapacity,
      participatingFacilities: facilities.length,
      byFacilityType,
    };
  }

  // --------------------------------------------------------------------------
  // Alerts
  // --------------------------------------------------------------------------

  /**
   * Create an alert
   */
  createAlert(
    facilityId: string,
    alertType: AlertType,
    severity: AlertSeverity,
    message: string,
    metrics: AlertMetrics,
    recommendations: string[]
  ): WaterEnergyAlert {
    const alert: WaterEnergyAlert = {
      alertId: `wea-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      alertType,
      severity,
      facilityId,
      message,
      timestamp: Date.now(),
      metrics,
      recommendations,
      acknowledged: false,
    };

    const alerts = this.alertCache.get(facilityId) ?? [];
    alerts.push(alert);
    this.alertCache.set(facilityId, alerts);

    return alert;
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(facilityId?: string): WaterEnergyAlert[] {
    if (facilityId) {
      return (this.alertCache.get(facilityId) ?? []).filter((a) => !a.resolvedAt);
    }

    const allAlerts: WaterEnergyAlert[] = [];
    for (const alerts of this.alertCache.values()) {
      allAlerts.push(...alerts.filter((a) => !a.resolvedAt));
    }
    return allAlerts;
  }

  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId: string): boolean {
    for (const alerts of this.alertCache.values()) {
      const alert = alerts.find((a) => a.alertId === alertId);
      if (alert) {
        alert.acknowledged = true;
        return true;
      }
    }
    return false;
  }

  /**
   * Resolve an alert
   */
  resolveAlert(alertId: string): boolean {
    for (const alerts of this.alertCache.values()) {
      const alert = alerts.find((a) => a.alertId === alertId);
      if (alert) {
        alert.resolvedAt = Date.now();
        return true;
      }
    }
    return false;
  }

  // --------------------------------------------------------------------------
  // Helper Methods
  // --------------------------------------------------------------------------

  private getPricingPeriod(hour: number): PricingPeriod {
    const schedule = this.config.pricingSchedule;
    if (schedule.criticalHours.includes(hour)) return 'Critical';
    if (schedule.onPeakHours.includes(hour)) return 'OnPeak';
    if (schedule.midPeakHours.includes(hour)) return 'MidPeak';
    return 'OffPeak';
  }

  private getPricingMultiplier(period: PricingPeriod): number {
    switch (period) {
      case 'OffPeak': return 0.5;
      case 'MidPeak': return 0.8;
      case 'OnPeak': return 1.2;
      case 'Critical': return 2.0;
    }
  }

  private getCarbonMultiplier(hour: number): number {
    // Assume higher renewable availability during daylight hours
    if (hour >= 10 && hour <= 16) return 0.8;
    if (hour >= 6 && hour <= 20) return 1.0;
    return 1.2;
  }

  private calculateExpectedSavings(facility: WaterEnergyFacility, targets: HourlyTarget[]): number {
    const baselineCost = targets.reduce((sum, t) => sum + facility.avgPowerConsumption * this.getPricingMultiplier(t.pricingPeriod), 0);
    const optimizedCost = targets.reduce((sum, t) => sum + t.expectedCost, 0);
    return Math.max(0, baselineCost - optimizedCost);
  }

  private calculateExpectedCarbonReduction(facility: WaterEnergyFacility, targets: HourlyTarget[]): number {
    const baselineCarbon = 24 * facility.avgPowerConsumption * 0.4; // Avg grid intensity
    const optimizedCarbon = targets.reduce((sum, t) => sum + t.targetPowerKw * this.getCarbonMultiplier(t.hour) * 0.4, 0);
    return Math.max(0, baselineCarbon - optimizedCarbon);
  }
}

// ============================================================================
// Result Types
// ============================================================================

export interface ConsumptionStats {
  totalEnergyKwh: number;
  totalWaterCubicMeters: number;
  avgEfficiency: number;
  totalCost: number;
  totalCarbonKg: number;
  periodBreakdown: {
    offPeak: number;
    midPeak: number;
    onPeak: number;
    critical: number;
  };
}

export interface DemandResponseCapacity {
  totalCapacityKw: number;
  immediateCapacityKw: number;
  participatingFacilities: number;
  byFacilityType: Record<string, number>;
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new Water-Energy bridge instance
 */
export function createWaterEnergyBridge(config?: WaterEnergyBridgeConfig): WaterEnergyBridge {
  return new WaterEnergyBridge(config);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate energy required for water pumping
 * Based on hydraulic power formula: P = ρgQh / η
 */
export function calculatePumpingEnergy(
  flowRateCubicMetersPerHour: number,
  headMeters: number,
  pumpEfficiency: number = 0.75
): number {
  const waterDensity = 1000; // kg/m³
  const gravity = 9.81; // m/s²
  const flowRatePerSecond = flowRateCubicMetersPerHour / 3600;
  const powerWatts = (waterDensity * gravity * flowRatePerSecond * headMeters) / pumpEfficiency;
  return powerWatts / 1000; // kW
}

/**
 * Calculate hydroelectric generation potential
 * Based on: P = ρgQh × η
 */
export function calculateHydroGenerationPotential(
  flowRateCubicMetersPerSecond: number,
  headMeters: number,
  turbineEfficiency: number = 0.85
): number {
  const waterDensity = 1000;
  const gravity = 9.81;
  const powerWatts = waterDensity * gravity * flowRateCubicMetersPerSecond * headMeters * turbineEfficiency;
  return powerWatts / 1000; // kW
}

/**
 * Estimate desalination energy requirement
 * Modern reverse osmosis typically requires 3-4 kWh per cubic meter
 */
export function calculateDesalinationEnergy(
  cubicMeters: number,
  salinity: 'low' | 'medium' | 'high' = 'medium'
): number {
  const energyPerCubicMeter = {
    low: 2.5,
    medium: 3.5,
    high: 4.5,
  };
  return cubicMeters * energyPerCubicMeter[salinity];
}

/**
 * Check if facility should participate in demand response
 */
export function shouldParticipateInDemandResponse(
  facility: WaterEnergyFacility,
  _event: DemandResponseEvent
): { participate: boolean; reason: string; maxReduction: number } {
  if (!facility.demandResponseCapable) {
    return { participate: false, reason: 'Facility not demand response capable', maxReduction: 0 };
  }

  if (facility.status !== 'Normal') {
    return { participate: false, reason: `Facility in ${facility.status} status`, maxReduction: 0 };
  }

  const maxReduction = facility.avgPowerConsumption - facility.minOperationalPower;
  if (maxReduction <= 0) {
    return { participate: false, reason: 'No flexibility available', maxReduction: 0 };
  }

  return {
    participate: true,
    reason: 'Eligible for participation',
    maxReduction,
  };
}
