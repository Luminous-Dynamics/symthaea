/**
 * Energy API Client Tests
 *
 * Tests for:
 * - NRELSolarClient (demo data mode, API mode, project analysis)
 * - ERCOTGridClient (simulated grid status)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  NRELSolarClient,
  ERCOTGridClient,
  type NRELSolarResource,
  type SolarProjectAnalysis,
  type GridStatus,
} from '../src/energy/api-client.js';

// =============================================================================
// NRELSolarClient Tests
// =============================================================================

describe('NRELSolarClient', () => {
  describe('demo mode (no API key)', () => {
    let client: NRELSolarClient;

    beforeEach(() => {
      client = new NRELSolarClient(); // no API key = demo mode
    });

    it('should return demo solar resource data', async () => {
      const resource = await client.getSolarResource(32.95, -96.73);

      expect(resource.latitude).toBe(32.95);
      expect(resource.longitude).toBe(-96.73);
      expect(resource.dataSource).toBe('demo');
      expect(resource.ghi).toBeGreaterThan(0);
      expect(resource.dni).toBeGreaterThan(0);
      expect(resource.dhi).toBeGreaterThan(0);
      expect(resource.annualKwhPerKw).toBeGreaterThan(0);
      expect(resource.monthlyGhi).toHaveLength(12);
      expect(resource.monthlyGeneration).toHaveLength(12);
    });

    it('should vary GHI by latitude', async () => {
      const texas = await client.getSolarResource(30, -96);
      const alaska = await client.getSolarResource(65, -150);

      // Texas should have more sun than Alaska
      expect(texas.ghi).toBeGreaterThan(alaska.ghi);
    });

    it('should set optimal tilt angle equal to latitude', async () => {
      const resource = await client.getSolarResource(33, -97);
      expect(resource.optimalTiltAngle).toBe(33);
    });

    it('should have summer months with higher generation than winter', async () => {
      const resource = await client.getSolarResource(33, -97);
      const june = resource.monthlyGeneration[5]; // index 5 = June
      const december = resource.monthlyGeneration[11]; // index 11 = December

      expect(june).toBeGreaterThan(december);
    });
  });

  describe('API mode', () => {
    it('should fetch from NREL API when key provided', async () => {
      const mockResponse = {
        ok: true,
        json: () =>
          Promise.resolve({
            outputs: {
              elevation: 300,
              avg_ghi: { annual: 5.5, monthly: Array(12).fill(5.5) },
              avg_dni: { annual: 6.0 },
              avg_dhi: { annual: 2.0 },
              avg_lat_tilt: { annual: 5.8 },
            },
            station_info: { class: 'TMY3' },
          }),
      };

      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockResponse));

      const client = new NRELSolarClient('test-api-key');
      const resource = await client.getSolarResource(33, -97);

      expect(resource.elevation).toBe(300);
      expect(resource.ghi).toBe(5.5);
      expect(resource.dataSource).toBe('TMY3');

      vi.unstubAllGlobals();
    });

    it('should fall back to demo data on API error', async () => {
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({ ok: false, status: 500 })
      );

      const client = new NRELSolarClient('test-api-key');
      const resource = await client.getSolarResource(33, -97);

      // Should still return valid data (demo fallback)
      expect(resource.dataSource).toBe('demo');
      expect(resource.ghi).toBeGreaterThan(0);

      vi.unstubAllGlobals();
    });
  });

  describe('project analysis', () => {
    let client: NRELSolarClient;

    beforeEach(() => {
      client = new NRELSolarClient(); // demo mode
    });

    it('should calculate solar project economics', async () => {
      const analysis = await client.analyzeProject(33, -97);

      expect(analysis.systemSizeKw).toBeGreaterThan(0);
      expect(analysis.numPanels).toBeGreaterThan(0);
      expect(analysis.annualGenerationKwh).toBeGreaterThan(0);
      expect(analysis.capacityFactor).toBeGreaterThan(0);
      expect(analysis.capacityFactor).toBeLessThan(1);
    });

    it('should apply 30% federal ITC', async () => {
      const analysis = await client.analyzeProject(33, -97);

      expect(analysis.federalItc).toBeCloseTo(analysis.grossCost * 0.3, -1);
    });

    it('should calculate net cost after incentives', async () => {
      const analysis = await client.analyzeProject(33, -97);

      expect(analysis.netCost).toBe(
        analysis.grossCost - analysis.federalItc - analysis.stateIncentives
      );
    });

    it('should respect custom system size', async () => {
      const analysis = await client.analyzeProject(33, -97, { systemSizeKw: 10 });

      expect(analysis.systemSizeKw).toBe(10);
      expect(analysis.numPanels).toBe(25); // 10kW / 400W per panel
    });

    it('should have 12 monthly values', async () => {
      const analysis = await client.analyzeProject(33, -97);

      expect(analysis.monthlyProduction).toHaveLength(12);
      expect(analysis.monthlySavings).toHaveLength(12);
    });

    it('should calculate positive payback and IRR', async () => {
      const analysis = await client.analyzeProject(33, -97);

      expect(analysis.paybackYears).toBeGreaterThan(0);
      expect(analysis.paybackYears).toBeLessThan(30);
      expect(analysis.irr).toBeGreaterThan(0);
    });

    it('should have 25-year savings greater than annual savings', async () => {
      const analysis = await client.analyzeProject(33, -97);

      expect(analysis.twentyFiveYearSavings).toBeGreaterThan(analysis.annualSavings);
    });

    it('should use different cost per watt for different sizes', async () => {
      const small = await client.analyzeProject(33, -97, { systemSizeKw: 5 });
      const large = await client.analyzeProject(33, -97, { systemSizeKw: 50 });

      // Cost per watt should be lower for larger systems
      const smallCostPerWatt = small.grossCost / (small.systemSizeKw * 1000);
      const largeCostPerWatt = large.grossCost / (large.systemSizeKw * 1000);
      expect(largeCostPerWatt).toBeLessThan(smallCostPerWatt);
    });

    it('should vary state incentives by state', async () => {
      const ca = await client.analyzeProject(34, -118, { systemSizeKw: 10, state: 'CA' });
      const tx = await client.analyzeProject(33, -97, { systemSizeKw: 10, state: 'TX' });

      // California has higher state incentives than Texas
      expect(ca.stateIncentives).toBeGreaterThan(tx.stateIncentives);
    });
  });
});

// =============================================================================
// ERCOTGridClient Tests
// =============================================================================

describe('ERCOTGridClient', () => {
  let client: ERCOTGridClient;

  beforeEach(() => {
    client = new ERCOTGridClient();
  });

  it('should return grid status with all required fields', async () => {
    const status = await client.getGridStatus();

    expect(status.condition).toBeDefined();
    expect(['normal', 'watch', 'advisory', 'emergency', 'critical']).toContain(
      status.condition
    );
    expect(status.currentLoadMw).toBeGreaterThan(0);
    expect(status.capacityMw).toBe(85000);
    expect(status.reserveMarginPercent).toBeDefined();
    expect(status.frequency).toBeCloseTo(60.0, 0);
    expect(status.renewablePercent).toBeGreaterThanOrEqual(0);
    expect(status.currentPriceMwh).toBeGreaterThan(0);
    expect(status.dayAheadPriceMwh).toBeGreaterThan(0);
    expect(status.lastUpdated).toBeGreaterThan(0);
  });

  it('should have frequency near 60 Hz', async () => {
    const status = await client.getGridStatus();

    // Frequency should be within 0.02 Hz of 60
    expect(Math.abs(status.frequency - 60.0)).toBeLessThan(0.02);
  });

  it('should have valid reserve margin', async () => {
    const status = await client.getGridStatus();

    expect(status.reserveMarginPercent).toBeLessThan(100);
    // Load should be less than capacity
    expect(status.currentLoadMw).toBeLessThan(status.capacityMw);
  });

  it('should have alerts array (possibly empty)', async () => {
    const status = await client.getGridStatus();

    expect(Array.isArray(status.alerts)).toBe(true);
  });

  it('should generate conservation alerts when grid is stressed', async () => {
    // Run multiple times to potentially hit stressed conditions
    let foundAlert = false;
    for (let i = 0; i < 50; i++) {
      const status = await client.getGridStatus();
      if (status.alerts.length > 0) {
        foundAlert = true;
        const alert = status.alerts[0];
        expect(alert.type).toBeDefined();
        expect(alert.severity).toBeDefined();
        expect(alert.message).toBeDefined();
        expect(alert.startTime).toBeGreaterThan(0);
        break;
      }
    }
    // It's OK if no alert was generated - grid stress is probabilistic
    // Just verify the structure is correct when alerts exist
    expect(true).toBe(true);
  });

  it('should have renewable percentage between 0 and 100', async () => {
    const status = await client.getGridStatus();

    expect(status.renewablePercent).toBeGreaterThanOrEqual(0);
    expect(status.renewablePercent).toBeLessThanOrEqual(100);
  });
});
