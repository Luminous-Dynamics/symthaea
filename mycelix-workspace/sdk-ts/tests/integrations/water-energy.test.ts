/**
 * Water-Energy Integration Tests
 *
 * Tests for WaterEnergyBridge -- the cross-domain bridge service connecting
 * water systems and energy grids. Covers facility management, energy
 * consumption tracking, optimization scheduling, demand response,
 * alerts, and utility functions.
 *
 * WaterEnergyBridge uses an in-memory data model, so tests verify
 * local state management and calculations.
 */

import { describe, it, expect, beforeEach } from 'vitest';

import {
  WaterEnergyBridge,
  createWaterEnergyBridge,
  calculatePumpingEnergy,
  calculateHydroGenerationPotential,
  calculateDesalinationEnergy,
  shouldParticipateInDemandResponse,
  type WaterEnergyFacility,
  type WaterSourceType,
  type WaterQuality,
  type WaterSystemStatus,
  type WaterFacilityType,
  type EnergySourceType,
  type GridLoadCategory,
  type PricingPeriod,
  type EnergyConnection,
  type GeoLocation,
  type WaterEnergyConsumption,
  type EnergySourceBreakdown,
  type DemandResponseEvent,
  type DemandResponseType,
  type DemandResponseStatus,
  type HydroGenerationRecord,
  type OptimizationSchedule,
  type HourlyTarget,
  type ScheduleStatus,
  type WaterEnergyAlert,
  type AlertType,
  type AlertSeverity,
  type AlertMetrics,
  type WaterEnergyReputation,
  type SustainabilityRating,
  type WaterEnergyBridgeConfig,
  type PricingSchedule,
  type ConsumptionStats,
  type DemandResponseCapacity,
} from '../../src/integrations/water-energy/index.js';

// ============================================================================
// Test helpers
// ============================================================================

function makeFacilityInput(overrides: Partial<Omit<WaterEnergyFacility, 'facilityId'>> = {}): Omit<WaterEnergyFacility, 'facilityId'> {
  return {
    name: overrides.name ?? 'Main Pumping Station',
    facilityType: overrides.facilityType ?? 'PumpingStation',
    location: overrides.location ?? { latitude: 32.95, longitude: -96.75 },
    status: overrides.status ?? 'Normal',
    energySources: overrides.energySources ?? [{
      connectionId: 'conn-001',
      sourceType: 'Grid',
      provider: 'TXU Energy',
      capacityKw: 500,
      currentLoadKw: 250,
      gridPriority: 'CriticalInfrastructure',
      isBackup: false,
    }],
    avgPowerConsumption: overrides.avgPowerConsumption ?? 200,
    peakPowerConsumption: overrides.peakPowerConsumption ?? 400,
    canGeneratePower: overrides.canGeneratePower ?? false,
    generationCapacity: overrides.generationCapacity,
    waterThroughput: overrides.waterThroughput ?? 1000,
    efficiencyRating: overrides.efficiencyRating ?? 0.85,
    demandResponseCapable: overrides.demandResponseCapable ?? true,
    minOperationalPower: overrides.minOperationalPower ?? 100,
  };
}

function makeConsumptionInput(facilityId: string, overrides: Partial<{
  timestamp: number;
  durationMinutes: number;
  energyConsumed: number;
  waterProcessed: number;
  cost: number;
  carbonEmissions: number;
  sourceBreakdown: EnergySourceBreakdown[];
}> = {}): Omit<WaterEnergyConsumption, 'recordId' | 'efficiency' | 'pricingPeriod'> {
  return {
    facilityId,
    timestamp: overrides.timestamp ?? Date.now(),
    durationMinutes: overrides.durationMinutes ?? 60,
    energyConsumed: overrides.energyConsumed ?? 200,
    waterProcessed: overrides.waterProcessed ?? 500,
    cost: overrides.cost ?? 25,
    carbonEmissions: overrides.carbonEmissions ?? 80,
    sourceBreakdown: overrides.sourceBreakdown ?? [
      { source: 'Grid', percentage: 0.8, carbonIntensity: 0.4 },
      { source: 'Solar', percentage: 0.2, carbonIntensity: 0.02 },
    ],
  };
}

function makeAlertMetrics(overrides: Partial<AlertMetrics> = {}): AlertMetrics {
  return {
    currentValue: overrides.currentValue ?? 450,
    threshold: overrides.threshold ?? 400,
    unit: overrides.unit ?? 'kW',
    trend: overrides.trend ?? 'Rising',
  };
}

// ============================================================================
// Type construction tests
// ============================================================================

describe('Water-Energy Types', () => {
  describe('WaterSourceType', () => {
    it('should accept all WaterSourceType variants', () => {
      const types: WaterSourceType[] = [
        'Reservoir', 'Groundwater', 'River', 'Lake',
        'Desalination', 'Recycled', 'Rainwater', 'Spring',
      ];
      expect(types).toHaveLength(8);
    });
  });

  describe('WaterQuality', () => {
    it('should accept all WaterQuality variants', () => {
      const qualities: WaterQuality[] = [
        'Potable', 'Agricultural', 'Industrial',
        'Recreational', 'NonPotable', 'Contaminated',
      ];
      expect(qualities).toHaveLength(6);
    });
  });

  describe('WaterSystemStatus', () => {
    it('should accept all WaterSystemStatus variants', () => {
      const statuses: WaterSystemStatus[] = [
        'Normal', 'LowCapacity', 'HighDemand',
        'Maintenance', 'Emergency', 'Offline',
      ];
      expect(statuses).toHaveLength(6);
    });
  });

  describe('WaterFacilityType', () => {
    it('should accept all WaterFacilityType variants', () => {
      const types: WaterFacilityType[] = [
        'PumpingStation', 'TreatmentPlant', 'DesalinationPlant',
        'DistributionCenter', 'StorageFacility', 'WastewaterTreatment',
        'HydroelectricDam', 'Reservoir',
      ];
      expect(types).toHaveLength(8);
    });
  });

  describe('EnergySourceType', () => {
    it('should accept all EnergySourceType variants', () => {
      const types: EnergySourceType[] = [
        'Solar', 'Wind', 'Hydro', 'Nuclear', 'NaturalGas',
        'Coal', 'Geothermal', 'Biomass', 'Battery', 'Grid',
      ];
      expect(types).toHaveLength(10);
    });
  });

  describe('GridLoadCategory', () => {
    it('should accept all GridLoadCategory variants', () => {
      const categories: GridLoadCategory[] = [
        'CriticalInfrastructure', 'Residential', 'Commercial',
        'Industrial', 'Agricultural', 'Municipal',
      ];
      expect(categories).toHaveLength(6);
    });
  });

  describe('PricingPeriod', () => {
    it('should accept all PricingPeriod variants', () => {
      const periods: PricingPeriod[] = ['OffPeak', 'MidPeak', 'OnPeak', 'Critical'];
      expect(periods).toHaveLength(4);
    });
  });

  describe('DemandResponseType', () => {
    it('should accept all DemandResponseType variants', () => {
      const types: DemandResponseType[] = [
        'VoluntaryReduction', 'EmergencyCurtailment',
        'PeakShaving', 'FrequencyRegulation', 'LoadShifting',
      ];
      expect(types).toHaveLength(5);
    });
  });

  describe('DemandResponseStatus', () => {
    it('should accept all DemandResponseStatus variants', () => {
      const statuses: DemandResponseStatus[] = [
        'Scheduled', 'Active', 'Completed', 'Cancelled', 'Failed',
      ];
      expect(statuses).toHaveLength(5);
    });
  });

  describe('AlertType', () => {
    it('should accept all AlertType variants', () => {
      const types: AlertType[] = [
        'PowerOutage', 'HighEnergyConsumption', 'LowWaterPressure',
        'EquipmentFailure', 'GridInstability', 'DroughtWarning',
        'FloodWarning', 'MaintenanceRequired', 'CostThresholdExceeded',
      ];
      expect(types).toHaveLength(9);
    });
  });

  describe('AlertSeverity', () => {
    it('should accept all AlertSeverity variants', () => {
      const severities: AlertSeverity[] = ['Info', 'Warning', 'Critical', 'Emergency'];
      expect(severities).toHaveLength(4);
    });
  });

  describe('ScheduleStatus', () => {
    it('should accept all ScheduleStatus variants', () => {
      const statuses: ScheduleStatus[] = ['Draft', 'Active', 'Completed', 'Suspended'];
      expect(statuses).toHaveLength(4);
    });
  });

  describe('SustainabilityRating', () => {
    it('should accept all SustainabilityRating variants', () => {
      const ratings: SustainabilityRating[] = [
        'Exemplary', 'Good', 'Average', 'NeedsImprovement', 'Poor',
      ];
      expect(ratings).toHaveLength(5);
    });
  });

  describe('AlertMetrics', () => {
    it('should construct with all fields', () => {
      const metrics = makeAlertMetrics();
      expect(metrics.currentValue).toBe(450);
      expect(metrics.threshold).toBe(400);
      expect(metrics.unit).toBe('kW');
      expect(metrics.trend).toBe('Rising');
    });

    it('should accept all trend variants', () => {
      const trends: AlertMetrics['trend'][] = ['Rising', 'Falling', 'Stable'];
      expect(trends).toHaveLength(3);
    });
  });

  describe('EnergyConnection', () => {
    it('should construct with all fields', () => {
      const conn: EnergyConnection = {
        connectionId: 'conn-001',
        sourceType: 'Solar',
        provider: 'SunPower',
        capacityKw: 100,
        currentLoadKw: 50,
        gridPriority: 'Residential',
        isBackup: false,
      };
      expect(conn.sourceType).toBe('Solar');
      expect(conn.isBackup).toBe(false);
    });
  });
});

// ============================================================================
// Factory function
// ============================================================================

describe('createWaterEnergyBridge', () => {
  it('should create a WaterEnergyBridge instance', () => {
    const bridge = createWaterEnergyBridge();
    expect(bridge).toBeInstanceOf(WaterEnergyBridge);
  });

  it('should accept custom config', () => {
    const bridge = createWaterEnergyBridge({
      cacheTtlMs: 10000,
      optimizationHorizon: 48,
    });
    expect(bridge).toBeInstanceOf(WaterEnergyBridge);
  });
});

// ============================================================================
// WaterEnergyBridge
// ============================================================================

describe('WaterEnergyBridge', () => {
  let bridge: WaterEnergyBridge;

  beforeEach(() => {
    bridge = new WaterEnergyBridge();
  });

  // ==========================================================================
  // Facility Management
  // ==========================================================================

  describe('Facility Management', () => {
    describe('registerFacility', () => {
      it('should register a facility with generated ID', async () => {
        const facility = await bridge.registerFacility(makeFacilityInput());

        expect(facility.facilityId).toBeTruthy();
        expect(facility.facilityId).toMatch(/^wef-/);
        expect(facility.name).toBe('Main Pumping Station');
        expect(facility.facilityType).toBe('PumpingStation');
        expect(facility.status).toBe('Normal');
        expect(facility.avgPowerConsumption).toBe(200);
        expect(facility.peakPowerConsumption).toBe(400);
        expect(facility.demandResponseCapable).toBe(true);
        expect(facility.minOperationalPower).toBe(100);
      });

      it('should accept all facility types', async () => {
        const types: WaterFacilityType[] = [
          'PumpingStation', 'TreatmentPlant', 'DesalinationPlant',
          'DistributionCenter', 'StorageFacility', 'WastewaterTreatment',
          'HydroelectricDam', 'Reservoir',
        ];
        for (const ft of types) {
          const facility = await bridge.registerFacility(makeFacilityInput({ facilityType: ft }));
          expect(facility.facilityType).toBe(ft);
        }
      });

      it('should generate unique IDs', async () => {
        const f1 = await bridge.registerFacility(makeFacilityInput({ name: 'Station A' }));
        const f2 = await bridge.registerFacility(makeFacilityInput({ name: 'Station B' }));
        expect(f1.facilityId).not.toBe(f2.facilityId);
      });

      it('should register hydro facility with generation capacity', async () => {
        const facility = await bridge.registerFacility(makeFacilityInput({
          facilityType: 'HydroelectricDam',
          canGeneratePower: true,
          generationCapacity: 5000,
        }));
        expect(facility.canGeneratePower).toBe(true);
        expect(facility.generationCapacity).toBe(5000);
      });
    });

    describe('getFacility', () => {
      it('should return facility by ID', async () => {
        const created = await bridge.registerFacility(makeFacilityInput());
        const fetched = bridge.getFacility(created.facilityId);
        expect(fetched).toBeDefined();
        expect(fetched!.name).toBe('Main Pumping Station');
      });

      it('should return undefined for non-existent facility', () => {
        expect(bridge.getFacility('non-existent')).toBeUndefined();
      });
    });

    describe('listFacilities', () => {
      it('should list all facilities without filter', async () => {
        await bridge.registerFacility(makeFacilityInput({ name: 'A' }));
        await bridge.registerFacility(makeFacilityInput({ name: 'B' }));

        const all = bridge.listFacilities();
        expect(all).toHaveLength(2);
      });

      it('should filter by facility type', async () => {
        await bridge.registerFacility(makeFacilityInput({ facilityType: 'PumpingStation' }));
        await bridge.registerFacility(makeFacilityInput({ facilityType: 'TreatmentPlant' }));
        await bridge.registerFacility(makeFacilityInput({ facilityType: 'PumpingStation' }));

        const pumps = bridge.listFacilities({ facilityType: 'PumpingStation' });
        expect(pumps).toHaveLength(2);
      });

      it('should filter by status', async () => {
        const f1 = await bridge.registerFacility(makeFacilityInput({ name: 'A' }));
        await bridge.registerFacility(makeFacilityInput({ name: 'B' }));
        bridge.updateFacilityStatus(f1.facilityId, 'Maintenance');

        const normal = bridge.listFacilities({ status: 'Normal' });
        expect(normal).toHaveLength(1);
      });

      it('should filter by demand response capability', async () => {
        await bridge.registerFacility(makeFacilityInput({ demandResponseCapable: true }));
        await bridge.registerFacility(makeFacilityInput({ demandResponseCapable: false }));

        const drCapable = bridge.listFacilities({ demandResponseCapable: true });
        expect(drCapable).toHaveLength(1);
      });

      it('should return empty array when no facilities match', async () => {
        await bridge.registerFacility(makeFacilityInput({ facilityType: 'PumpingStation' }));
        expect(bridge.listFacilities({ facilityType: 'DesalinationPlant' })).toEqual([]);
      });
    });

    describe('updateFacilityStatus', () => {
      it('should update facility status', async () => {
        const facility = await bridge.registerFacility(makeFacilityInput());
        const updated = bridge.updateFacilityStatus(facility.facilityId, 'Maintenance');

        expect(updated).not.toBeNull();
        expect(updated!.status).toBe('Maintenance');
      });

      it('should accept all status variants', async () => {
        const facility = await bridge.registerFacility(makeFacilityInput());
        const statuses: WaterSystemStatus[] = [
          'Normal', 'LowCapacity', 'HighDemand', 'Maintenance', 'Emergency', 'Offline',
        ];
        for (const s of statuses) {
          const result = bridge.updateFacilityStatus(facility.facilityId, s);
          expect(result!.status).toBe(s);
        }
      });

      it('should return null for non-existent facility', () => {
        expect(bridge.updateFacilityStatus('ghost', 'Offline')).toBeNull();
      });
    });
  });

  // ==========================================================================
  // Energy Consumption Tracking
  // ==========================================================================

  describe('Energy Consumption Tracking', () => {
    let facilityId: string;

    beforeEach(async () => {
      const facility = await bridge.registerFacility(makeFacilityInput());
      facilityId = facility.facilityId;
    });

    describe('recordConsumption', () => {
      it('should record consumption with computed fields', () => {
        const record = bridge.recordConsumption(makeConsumptionInput(facilityId));

        expect(record.recordId).toBeTruthy();
        expect(record.recordId).toMatch(/^wec-/);
        expect(record.facilityId).toBe(facilityId);
        expect(record.energyConsumed).toBe(200);
        expect(record.waterProcessed).toBe(500);
        expect(record.efficiency).toBeCloseTo(200 / 500);
        expect(['OffPeak', 'MidPeak', 'OnPeak', 'Critical']).toContain(record.pricingPeriod);
      });

      it('should compute zero efficiency when no water processed', () => {
        const record = bridge.recordConsumption(makeConsumptionInput(facilityId, {
          waterProcessed: 0,
        }));
        expect(record.efficiency).toBe(0);
      });

      it('should assign pricing period based on timestamp hour', () => {
        // Create a timestamp at 2:00 UTC (off-peak in default schedule)
        const offPeakTime = new Date();
        offPeakTime.setUTCHours(2, 0, 0, 0);

        const record = bridge.recordConsumption(makeConsumptionInput(facilityId, {
          timestamp: offPeakTime.getTime(),
        }));
        expect(record.pricingPeriod).toBe('OffPeak');
      });
    });

    describe('getConsumptionHistory', () => {
      it('should return all consumption records for a facility', () => {
        bridge.recordConsumption(makeConsumptionInput(facilityId));
        bridge.recordConsumption(makeConsumptionInput(facilityId));

        const history = bridge.getConsumptionHistory(facilityId);
        expect(history).toHaveLength(2);
      });

      it('should filter by since timestamp', () => {
        const now = Date.now();
        bridge.recordConsumption(makeConsumptionInput(facilityId, {
          timestamp: now - 3600_000, // 1 hour ago
        }));
        bridge.recordConsumption(makeConsumptionInput(facilityId, {
          timestamp: now,
        }));

        const recent = bridge.getConsumptionHistory(facilityId, now - 1800_000); // Last 30 min
        expect(recent).toHaveLength(1);
      });

      it('should return empty array for facility with no records', () => {
        expect(bridge.getConsumptionHistory(facilityId)).toEqual([]);
      });

      it('should return empty array for non-existent facility', () => {
        expect(bridge.getConsumptionHistory('ghost')).toEqual([]);
      });
    });

    describe('calculateConsumptionStats', () => {
      it('should calculate aggregate statistics', () => {
        const now = Date.now();
        bridge.recordConsumption(makeConsumptionInput(facilityId, {
          timestamp: now,
          energyConsumed: 200,
          waterProcessed: 500,
          cost: 25,
          carbonEmissions: 80,
        }));
        bridge.recordConsumption(makeConsumptionInput(facilityId, {
          timestamp: now,
          energyConsumed: 300,
          waterProcessed: 700,
          cost: 35,
          carbonEmissions: 120,
        }));

        const stats = bridge.calculateConsumptionStats(facilityId, now - 60000);

        expect(stats.totalEnergyKwh).toBe(500);
        expect(stats.totalWaterCubicMeters).toBe(1200);
        expect(stats.totalCost).toBe(60);
        expect(stats.totalCarbonKg).toBe(200);
        expect(stats.avgEfficiency).toBeCloseTo(500 / 1200);
      });

      it('should return zero stats when no records in range', () => {
        const stats = bridge.calculateConsumptionStats(facilityId, Date.now());

        expect(stats.totalEnergyKwh).toBe(0);
        expect(stats.totalWaterCubicMeters).toBe(0);
        expect(stats.avgEfficiency).toBe(0);
        expect(stats.totalCost).toBe(0);
        expect(stats.totalCarbonKg).toBe(0);
        expect(stats.periodBreakdown.offPeak).toBe(0);
        expect(stats.periodBreakdown.midPeak).toBe(0);
        expect(stats.periodBreakdown.onPeak).toBe(0);
        expect(stats.periodBreakdown.critical).toBe(0);
      });
    });
  });

  // ==========================================================================
  // Optimization
  // ==========================================================================

  describe('Optimization', () => {
    let facilityId: string;

    beforeEach(async () => {
      const facility = await bridge.registerFacility(makeFacilityInput());
      facilityId = facility.facilityId;
    });

    describe('generateOptimizationSchedule', () => {
      it('should generate a 24-hour schedule', () => {
        const schedule = bridge.generateOptimizationSchedule(facilityId);

        expect(schedule.scheduleId).toBeTruthy();
        expect(schedule.scheduleId).toMatch(/^sch-/);
        expect(schedule.facilityId).toBe(facilityId);
        expect(schedule.hourlyTargets).toHaveLength(24);
        expect(schedule.status).toBe('Draft');
        expect(schedule.validFrom).toBeGreaterThan(0);
        expect(schedule.validTo).toBeGreaterThan(schedule.validFrom);
      });

      it('should cover hours 0-23', () => {
        const schedule = bridge.generateOptimizationSchedule(facilityId);
        const hours = schedule.hourlyTargets.map((t) => t.hour);
        expect(hours).toEqual(Array.from({ length: 24 }, (_, i) => i));
      });

      it('should set pricing periods for each hour', () => {
        const schedule = bridge.generateOptimizationSchedule(facilityId);
        for (const target of schedule.hourlyTargets) {
          expect(['OffPeak', 'MidPeak', 'OnPeak', 'Critical']).toContain(target.pricingPeriod);
        }
      });

      it('should reduce power during peak when prioritizing cost', () => {
        const schedule = bridge.generateOptimizationSchedule(facilityId, {
          prioritizeCost: true,
        });

        // On-peak hours (10-18 in default schedule) should have reduced power
        const onPeakTargets = schedule.hourlyTargets.filter((t) => t.pricingPeriod === 'OnPeak');
        const offPeakTargets = schedule.hourlyTargets.filter((t) => t.pricingPeriod === 'OffPeak');

        if (onPeakTargets.length > 0 && offPeakTargets.length > 0) {
          const avgOnPeak = onPeakTargets.reduce((s, t) => s + t.targetPowerKw, 0) / onPeakTargets.length;
          const avgOffPeak = offPeakTargets.reduce((s, t) => s + t.targetPowerKw, 0) / offPeakTargets.length;
          expect(avgOnPeak).toBeLessThan(avgOffPeak);
        }
      });

      it('should respect minimum water throughput', () => {
        const schedule = bridge.generateOptimizationSchedule(facilityId, {
          prioritizeCost: true,
          minWaterThroughput: 800,
        });

        for (const target of schedule.hourlyTargets) {
          expect(target.targetWaterThroughput).toBeGreaterThanOrEqual(800);
        }
      });

      it('should throw for non-existent facility', () => {
        expect(() => {
          bridge.generateOptimizationSchedule('ghost');
        }).toThrow('Facility not found: ghost');
      });

      it('should calculate expected savings', () => {
        const schedule = bridge.generateOptimizationSchedule(facilityId, {
          prioritizeCost: true,
        });
        expect(schedule.expectedSavings).toBeGreaterThanOrEqual(0);
      });

      it('should calculate expected carbon reduction', () => {
        const schedule = bridge.generateOptimizationSchedule(facilityId, {
          prioritizeCarbon: true,
        });
        expect(schedule.expectedCarbonReduction).toBeGreaterThanOrEqual(0);
      });
    });

    describe('activateSchedule', () => {
      it('should activate a draft schedule', () => {
        const schedule = bridge.generateOptimizationSchedule(facilityId);
        const activated = bridge.activateSchedule(schedule.scheduleId);

        expect(activated).not.toBeNull();
        expect(activated!.status).toBe('Active');
      });

      it('should return null for non-existent schedule', () => {
        expect(bridge.activateSchedule('ghost')).toBeNull();
      });

      it('should return null for already activated schedule', () => {
        const schedule = bridge.generateOptimizationSchedule(facilityId);
        bridge.activateSchedule(schedule.scheduleId);
        const result = bridge.activateSchedule(schedule.scheduleId);
        expect(result).toBeNull();
      });
    });
  });

  // ==========================================================================
  // Demand Response
  // ==========================================================================

  describe('Demand Response', () => {
    describe('createDemandResponseEvent', () => {
      it('should create a demand response event', () => {
        const event = bridge.createDemandResponseEvent(
          'PeakShaving',
          100,
          3600,
          ['facility-001', 'facility-002'],
        );

        expect(event.eventId).toBeTruthy();
        expect(event.eventId).toMatch(/^dre-/);
        expect(event.eventType).toBe('PeakShaving');
        expect(event.requestedReduction).toBe(100);
        expect(event.duration).toBe(3600);
        expect(event.participatingFacilities).toHaveLength(2);
        expect(event.status).toBe('Scheduled');
      });

      it('should accept all demand response types', () => {
        const types: DemandResponseType[] = [
          'VoluntaryReduction', 'EmergencyCurtailment',
          'PeakShaving', 'FrequencyRegulation', 'LoadShifting',
        ];
        for (const t of types) {
          const event = bridge.createDemandResponseEvent(t, 50, 1800, []);
          expect(event.eventType).toBe(t);
        }
      });
    });

    describe('calculateDemandResponseCapacity', () => {
      it('should calculate capacity from demand-response-capable facilities', async () => {
        await bridge.registerFacility(makeFacilityInput({
          demandResponseCapable: true,
          peakPowerConsumption: 400,
          minOperationalPower: 100,
        }));
        await bridge.registerFacility(makeFacilityInput({
          demandResponseCapable: true,
          peakPowerConsumption: 600,
          minOperationalPower: 200,
        }));
        await bridge.registerFacility(makeFacilityInput({
          demandResponseCapable: false,
          peakPowerConsumption: 300,
          minOperationalPower: 50,
        }));

        const capacity = bridge.calculateDemandResponseCapacity();

        // (400-100) + (600-200) = 300 + 400 = 700
        expect(capacity.totalCapacityKw).toBe(700);
        expect(capacity.immediateCapacityKw).toBe(350); // 50% of total
        expect(capacity.participatingFacilities).toBe(2);
        expect(capacity.byFacilityType).toBeDefined();
      });

      it('should return zero capacity when no facilities exist', () => {
        const capacity = bridge.calculateDemandResponseCapacity();
        expect(capacity.totalCapacityKw).toBe(0);
        expect(capacity.immediateCapacityKw).toBe(0);
        expect(capacity.participatingFacilities).toBe(0);
      });

      it('should exclude facilities not in Normal status', async () => {
        const facility = await bridge.registerFacility(makeFacilityInput({
          demandResponseCapable: true,
        }));
        bridge.updateFacilityStatus(facility.facilityId, 'Maintenance');

        const capacity = bridge.calculateDemandResponseCapacity();
        expect(capacity.participatingFacilities).toBe(0);
      });
    });
  });

  // ==========================================================================
  // Alerts
  // ==========================================================================

  describe('Alerts', () => {
    let facilityId: string;

    beforeEach(async () => {
      const facility = await bridge.registerFacility(makeFacilityInput());
      facilityId = facility.facilityId;
    });

    describe('createAlert', () => {
      it('should create an alert with correct fields', () => {
        const alert = bridge.createAlert(
          facilityId,
          'HighEnergyConsumption',
          'Warning',
          'Energy consumption exceeds threshold',
          makeAlertMetrics(),
          ['Reduce non-essential loads', 'Check for leaks'],
        );

        expect(alert.alertId).toBeTruthy();
        expect(alert.alertId).toMatch(/^wea-/);
        expect(alert.alertType).toBe('HighEnergyConsumption');
        expect(alert.severity).toBe('Warning');
        expect(alert.facilityId).toBe(facilityId);
        expect(alert.message).toContain('exceeds threshold');
        expect(alert.recommendations).toHaveLength(2);
        expect(alert.acknowledged).toBe(false);
        expect(alert.resolvedAt).toBeUndefined();
      });

      it('should accept all alert types', () => {
        const types: AlertType[] = [
          'PowerOutage', 'HighEnergyConsumption', 'LowWaterPressure',
          'EquipmentFailure', 'GridInstability', 'DroughtWarning',
          'FloodWarning', 'MaintenanceRequired', 'CostThresholdExceeded',
        ];
        for (const t of types) {
          const alert = bridge.createAlert(facilityId, t, 'Info', 'msg', makeAlertMetrics(), []);
          expect(alert.alertType).toBe(t);
        }
      });
    });

    describe('getActiveAlerts', () => {
      it('should return active alerts for a specific facility', () => {
        bridge.createAlert(facilityId, 'PowerOutage', 'Critical', 'Outage', makeAlertMetrics(), []);
        bridge.createAlert(facilityId, 'LowWaterPressure', 'Warning', 'Low pressure', makeAlertMetrics(), []);

        const alerts = bridge.getActiveAlerts(facilityId);
        expect(alerts).toHaveLength(2);
      });

      it('should return all active alerts when no facility specified', async () => {
        const f2 = await bridge.registerFacility(makeFacilityInput({ name: 'Other' }));
        bridge.createAlert(facilityId, 'PowerOutage', 'Critical', 'A', makeAlertMetrics(), []);
        bridge.createAlert(f2.facilityId, 'GridInstability', 'Warning', 'B', makeAlertMetrics(), []);

        const alerts = bridge.getActiveAlerts();
        expect(alerts).toHaveLength(2);
      });

      it('should not return resolved alerts', () => {
        const alert = bridge.createAlert(facilityId, 'PowerOutage', 'Critical', 'msg', makeAlertMetrics(), []);
        bridge.resolveAlert(alert.alertId);

        const active = bridge.getActiveAlerts(facilityId);
        expect(active).toHaveLength(0);
      });

      it('should return empty array for facility with no alerts', () => {
        expect(bridge.getActiveAlerts(facilityId)).toEqual([]);
      });
    });

    describe('acknowledgeAlert', () => {
      it('should mark alert as acknowledged', () => {
        const alert = bridge.createAlert(facilityId, 'PowerOutage', 'Critical', 'msg', makeAlertMetrics(), []);

        const result = bridge.acknowledgeAlert(alert.alertId);
        expect(result).toBe(true);
        expect(alert.acknowledged).toBe(true);
      });

      it('should return false for non-existent alert', () => {
        expect(bridge.acknowledgeAlert('ghost')).toBe(false);
      });
    });

    describe('resolveAlert', () => {
      it('should mark alert as resolved with timestamp', () => {
        const alert = bridge.createAlert(facilityId, 'PowerOutage', 'Critical', 'msg', makeAlertMetrics(), []);

        const result = bridge.resolveAlert(alert.alertId);
        expect(result).toBe(true);
        expect(alert.resolvedAt).toBeGreaterThan(0);
      });

      it('should return false for non-existent alert', () => {
        expect(bridge.resolveAlert('ghost')).toBe(false);
      });
    });
  });

  // ==========================================================================
  // Configuration
  // ==========================================================================

  describe('Configuration', () => {
    it('should use default config values', () => {
      const defaultBridge = new WaterEnergyBridge();
      // Verify it works with defaults by registering a facility
      expect(defaultBridge).toBeInstanceOf(WaterEnergyBridge);
    });

    it('should accept custom pricing schedule', async () => {
      const customBridge = new WaterEnergyBridge({
        pricingSchedule: {
          timezone: 'America/Chicago',
          offPeakHours: [0, 1, 2, 3, 4, 5],
          midPeakHours: [6, 7, 8, 20, 21, 22, 23],
          onPeakHours: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
          criticalHours: [],
        },
      });

      const facility = await customBridge.registerFacility(makeFacilityInput());
      const schedule = customBridge.generateOptimizationSchedule(facility.facilityId);
      expect(schedule.hourlyTargets).toHaveLength(24);
    });

    it('should accept custom carbon intensity defaults', async () => {
      const customBridge = new WaterEnergyBridge({
        carbonIntensityDefaults: { Solar: 0.01, Grid: 0.50 },
      });
      expect(customBridge).toBeInstanceOf(WaterEnergyBridge);
    });
  });
});

// ============================================================================
// Utility functions
// ============================================================================

describe('Utility Functions', () => {
  describe('calculatePumpingEnergy', () => {
    it('should calculate pumping energy with default efficiency', () => {
      // P = rho * g * Q * h / eta / 1000
      // = 1000 * 9.81 * (100/3600) * 50 / 0.75 / 1000
      const result = calculatePumpingEnergy(100, 50);
      expect(result).toBeGreaterThan(0);
      // 1000 * 9.81 * (100/3600) * 50 / 0.75 / 1000 = ~18.17 kW
      expect(result).toBeCloseTo(18.167, 1);
    });

    it('should accept custom pump efficiency', () => {
      const highEff = calculatePumpingEnergy(100, 50, 0.90);
      const lowEff = calculatePumpingEnergy(100, 50, 0.60);
      expect(highEff).toBeLessThan(lowEff);
    });

    it('should return 0 for zero flow rate', () => {
      expect(calculatePumpingEnergy(0, 50)).toBe(0);
    });

    it('should return 0 for zero head', () => {
      expect(calculatePumpingEnergy(100, 0)).toBe(0);
    });
  });

  describe('calculateHydroGenerationPotential', () => {
    it('should calculate hydro generation with default efficiency', () => {
      // P = rho * g * Q * h * eta / 1000
      // = 1000 * 9.81 * 10 * 50 * 0.85 / 1000 = 4169.25 kW
      const result = calculateHydroGenerationPotential(10, 50);
      expect(result).toBeCloseTo(4169.25, 0);
    });

    it('should accept custom turbine efficiency', () => {
      const highEff = calculateHydroGenerationPotential(10, 50, 0.95);
      const lowEff = calculateHydroGenerationPotential(10, 50, 0.70);
      expect(highEff).toBeGreaterThan(lowEff);
    });

    it('should return 0 for zero flow', () => {
      expect(calculateHydroGenerationPotential(0, 50)).toBe(0);
    });
  });

  describe('calculateDesalinationEnergy', () => {
    it('should calculate for medium salinity by default', () => {
      const result = calculateDesalinationEnergy(100);
      expect(result).toBe(350); // 100 * 3.5
    });

    it('should calculate for low salinity', () => {
      expect(calculateDesalinationEnergy(100, 'low')).toBe(250);
    });

    it('should calculate for high salinity', () => {
      expect(calculateDesalinationEnergy(100, 'high')).toBe(450);
    });

    it('should scale linearly with volume', () => {
      const small = calculateDesalinationEnergy(10);
      const large = calculateDesalinationEnergy(100);
      expect(large).toBeCloseTo(small * 10);
    });

    it('should return 0 for zero volume', () => {
      expect(calculateDesalinationEnergy(0)).toBe(0);
    });
  });

  describe('shouldParticipateInDemandResponse', () => {
    const mockEvent: DemandResponseEvent = {
      eventId: 'test',
      eventType: 'PeakShaving',
      triggerTime: Date.now(),
      duration: 3600,
      requestedReduction: 100,
      participatingFacilities: [],
      status: 'Scheduled',
    };

    it('should allow participation for eligible facility', () => {
      const facility: WaterEnergyFacility = {
        facilityId: 'f1',
        name: 'Test',
        facilityType: 'PumpingStation',
        location: { latitude: 0, longitude: 0 },
        status: 'Normal',
        energySources: [],
        avgPowerConsumption: 200,
        peakPowerConsumption: 400,
        canGeneratePower: false,
        waterThroughput: 500,
        efficiencyRating: 0.85,
        demandResponseCapable: true,
        minOperationalPower: 100,
      };

      const result = shouldParticipateInDemandResponse(facility, mockEvent);
      expect(result.participate).toBe(true);
      expect(result.maxReduction).toBe(100); // 200 - 100
      expect(result.reason).toContain('Eligible');
    });

    it('should reject facility not demand response capable', () => {
      const facility: WaterEnergyFacility = {
        facilityId: 'f2',
        name: 'Non-DR',
        facilityType: 'TreatmentPlant',
        location: { latitude: 0, longitude: 0 },
        status: 'Normal',
        energySources: [],
        avgPowerConsumption: 200,
        peakPowerConsumption: 400,
        canGeneratePower: false,
        waterThroughput: 500,
        efficiencyRating: 0.85,
        demandResponseCapable: false,
        minOperationalPower: 100,
      };

      const result = shouldParticipateInDemandResponse(facility, mockEvent);
      expect(result.participate).toBe(false);
      expect(result.reason).toContain('not demand response capable');
    });

    it('should reject facility not in Normal status', () => {
      const facility: WaterEnergyFacility = {
        facilityId: 'f3',
        name: 'Maintenance',
        facilityType: 'PumpingStation',
        location: { latitude: 0, longitude: 0 },
        status: 'Maintenance',
        energySources: [],
        avgPowerConsumption: 200,
        peakPowerConsumption: 400,
        canGeneratePower: false,
        waterThroughput: 500,
        efficiencyRating: 0.85,
        demandResponseCapable: true,
        minOperationalPower: 100,
      };

      const result = shouldParticipateInDemandResponse(facility, mockEvent);
      expect(result.participate).toBe(false);
      expect(result.reason).toContain('Maintenance');
    });

    it('should reject facility with no flexibility', () => {
      const facility: WaterEnergyFacility = {
        facilityId: 'f4',
        name: 'NoFlex',
        facilityType: 'PumpingStation',
        location: { latitude: 0, longitude: 0 },
        status: 'Normal',
        energySources: [],
        avgPowerConsumption: 100,
        peakPowerConsumption: 200,
        canGeneratePower: false,
        waterThroughput: 500,
        efficiencyRating: 0.85,
        demandResponseCapable: true,
        minOperationalPower: 100, // avg == min, no flexibility
      };

      const result = shouldParticipateInDemandResponse(facility, mockEvent);
      expect(result.participate).toBe(false);
      expect(result.reason).toContain('No flexibility');
    });
  });
});

// ============================================================================
// Full Lifecycle
// ============================================================================

describe('Full Lifecycle', () => {
  it('should support register -> consume -> optimize -> alert -> resolve', async () => {
    const bridge = new WaterEnergyBridge();

    // Step 1: Register facility
    const facility = await bridge.registerFacility({
      name: 'Central Treatment Plant',
      facilityType: 'TreatmentPlant',
      location: { latitude: 32.95, longitude: -96.75 },
      status: 'Normal',
      energySources: [{
        connectionId: 'grid-001',
        sourceType: 'Grid',
        provider: 'Oncor',
        capacityKw: 1000,
        currentLoadKw: 500,
        gridPriority: 'CriticalInfrastructure',
        isBackup: false,
      }],
      avgPowerConsumption: 500,
      peakPowerConsumption: 800,
      canGeneratePower: false,
      waterThroughput: 2000,
      efficiencyRating: 0.82,
      demandResponseCapable: true,
      minOperationalPower: 250,
    });
    expect(facility.facilityId).toBeTruthy();

    // Step 2: Record consumption
    const consumption = bridge.recordConsumption({
      facilityId: facility.facilityId,
      timestamp: Date.now(),
      durationMinutes: 60,
      energyConsumed: 520,
      waterProcessed: 2100,
      cost: 65,
      carbonEmissions: 208,
      sourceBreakdown: [{ source: 'Grid', percentage: 1.0, carbonIntensity: 0.4 }],
    });
    expect(consumption.efficiency).toBeCloseTo(520 / 2100);

    // Step 3: Generate optimization schedule
    const schedule = bridge.generateOptimizationSchedule(facility.facilityId, {
      prioritizeCost: true,
      minWaterThroughput: 1500,
    });
    expect(schedule.hourlyTargets).toHaveLength(24);
    expect(schedule.status).toBe('Draft');

    // Step 4: Activate schedule
    const activated = bridge.activateSchedule(schedule.scheduleId);
    expect(activated!.status).toBe('Active');

    // Step 5: Create alert
    const alert = bridge.createAlert(
      facility.facilityId,
      'HighEnergyConsumption',
      'Warning',
      'Consumption 4% above threshold',
      { currentValue: 520, threshold: 500, unit: 'kWh', trend: 'Rising' },
      ['Check pump efficiency', 'Review treatment process'],
    );
    expect(alert.acknowledged).toBe(false);

    // Step 6: Acknowledge alert
    const acked = bridge.acknowledgeAlert(alert.alertId);
    expect(acked).toBe(true);

    // Step 7: Resolve alert
    const resolved = bridge.resolveAlert(alert.alertId);
    expect(resolved).toBe(true);

    // Step 8: Verify no active alerts remain
    const activeAlerts = bridge.getActiveAlerts(facility.facilityId);
    expect(activeAlerts).toHaveLength(0);

    // Step 9: Check consumption stats
    const stats = bridge.calculateConsumptionStats(facility.facilityId, Date.now() - 3600_000);
    expect(stats.totalEnergyKwh).toBe(520);
    expect(stats.totalWaterCubicMeters).toBe(2100);
  });
});
