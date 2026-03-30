// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Health-Energy Integration Tests
 *
 * Tests for HealthEnergyBridge, MedicalDeviceClient, MedicalPriorityClient,
 * HealthcareFacilityClient, and utility functions (calculatePriorityScore,
 * estimateRestorationTime, requiresImmediateResponse).
 *
 * Uses a mock ZomeCallable for conductor interactions.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  HealthEnergyBridge,
  MedicalDeviceClient,
  MedicalPriorityClient,
  HealthcareFacilityClient,
  createHealthEnergyBridge,
  calculatePriorityScore,
  estimateRestorationTime,
  requiresImmediateResponse,
  type ZomeCallable,
  type MedicalDevice,
  type MedicalPriorityAlert,
  type HealthcareFacility,
  type PowerOutageEvent,
  type FacilityOutageResponse,
} from '../../src/integrations/health-energy/index.js';

function mockZomeClient(): ZomeCallable {
  return {
    callZome: vi.fn(),
  };
}

const fakeHash = new Uint8Array(39);

function makeDevice(overrides: Partial<MedicalDevice> = {}): MedicalDevice {
  return {
    deviceId: 'dev-001',
    patientHash: fakeHash,
    category: 'LifeSupport',
    powerPriority: 'Critical',
    powerRequirementWatts: 500,
    batteryBackupMinutes: 120,
    deviceName: 'Ventilator',
    criticalMedicalConditions: ['respiratory_failure'],
    registeredAt: Date.now(),
    updatedAt: Date.now(),
    active: true,
    ...overrides,
  };
}

describe('Health-Energy Integration', () => {
  describe('MedicalDeviceClient', () => {
    let client: ZomeCallable;
    let devices: MedicalDeviceClient;

    beforeEach(() => {
      client = mockZomeClient();
      devices = new MedicalDeviceClient(client);
    });

    it('should register a medical device', async () => {
      const device = makeDevice();
      vi.mocked(client.callZome).mockResolvedValue(device);

      const result = await devices.registerDevice({
        patientHash: fakeHash,
        category: 'LifeSupport',
        powerPriority: 'Critical',
        powerRequirementWatts: 500,
        deviceName: 'Ventilator',
        criticalMedicalConditions: ['respiratory_failure'],
      });

      expect(result.deviceId).toBe('dev-001');
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'health',
          zome_name: 'bridge',
          fn_name: 'register_medical_device',
        }),
      );
    });

    it('should get patient devices', async () => {
      vi.mocked(client.callZome).mockResolvedValue([makeDevice()]);

      const result = await devices.getPatientDevices(fakeHash);

      expect(result).toHaveLength(1);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ fn_name: 'get_patient_devices' }),
      );
    });

    it('should update device status', async () => {
      const updatedDevice = makeDevice({ active: false });
      vi.mocked(client.callZome).mockResolvedValue(updatedDevice);

      const result = await devices.updateDeviceStatus('dev-001', false);

      expect(result.active).toBe(false);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_device_status',
          payload: { device_id: 'dev-001', active: false },
        }),
      );
    });

    it('should get critical devices in an area', async () => {
      vi.mocked(client.callZome).mockResolvedValue([makeDevice()]);

      const result = await devices.getCriticalDevicesInArea(32.7, -96.8, 10);

      expect(result).toHaveLength(1);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_critical_devices_in_area',
          payload: { lat: 32.7, lng: -96.8, radius_km: 10 },
        }),
      );
    });
  });

  describe('MedicalPriorityClient', () => {
    let client: ZomeCallable;
    let priorities: MedicalPriorityClient;

    beforeEach(() => {
      client = mockZomeClient();
      priorities = new MedicalPriorityClient(client);
    });

    it('should create a priority alert', async () => {
      const alert: MedicalPriorityAlert = {
        alertId: 'alert-001',
        outageId: 'outage-001',
        patientDid: 'did:mycelix:patient-1',
        deviceId: 'dev-001',
        powerPriority: 'Critical',
        batteryRemainingMinutes: 60,
        medicalConditions: ['respiratory_failure'],
        requiresImmediateResponse: true,
        createdAt: Date.now(),
      };
      vi.mocked(client.callZome).mockResolvedValue(alert);

      const result = await priorities.createPriorityAlert('outage-001', 'dev-001', 60);

      expect(result.alertId).toBe('alert-001');
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'energy',
          zome_name: 'priorities',
          fn_name: 'create_medical_priority_alert',
        }),
      );
    });

    it('should get outage alerts', async () => {
      vi.mocked(client.callZome).mockResolvedValue([]);

      const result = await priorities.getOutageAlerts('outage-001');

      expect(result).toEqual([]);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ fn_name: 'get_outage_medical_alerts' }),
      );
    });

    it('should resolve an alert', async () => {
      vi.mocked(client.callZome).mockResolvedValue(undefined);

      await priorities.resolveAlert('alert-001');

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'resolve_medical_alert',
          payload: 'alert-001',
        }),
      );
    });

    it('should get restoration priority list', async () => {
      vi.mocked(client.callZome).mockResolvedValue([]);

      const result = await priorities.getRestorationPriorityList('outage-001');

      expect(result).toEqual([]);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ fn_name: 'get_restoration_priority_list' }),
      );
    });
  });

  describe('HealthcareFacilityClient', () => {
    let client: ZomeCallable;
    let facilities: HealthcareFacilityClient;

    beforeEach(() => {
      client = mockZomeClient();
      facilities = new HealthcareFacilityClient(client);
    });

    it('should register a healthcare facility', async () => {
      const facility: HealthcareFacility = {
        facilityId: 'fac-001',
        name: 'General Hospital',
        facilityType: 'Hospital',
        location: { lat: 32.7, lng: -96.8, address: '123 Main St' },
        totalPowerCapacityKw: 5000,
        criticalLoadKw: 2000,
        hasBackupPower: true,
        backupCapacityHours: 72,
        primaryContact: '+1-555-1234',
        emergencyContact: '+1-555-9999',
        registeredAt: Date.now(),
      };
      vi.mocked(client.callZome).mockResolvedValue(facility);

      const result = await facilities.registerFacility({
        name: 'General Hospital',
        facilityType: 'Hospital',
        location: { lat: 32.7, lng: -96.8, address: '123 Main St' },
        totalPowerCapacityKw: 5000,
        criticalLoadKw: 2000,
        hasBackupPower: true,
        backupCapacityHours: 72,
        primaryContact: '+1-555-1234',
        emergencyContact: '+1-555-9999',
      });

      expect(result.facilityId).toBe('fac-001');
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ fn_name: 'register_healthcare_facility' }),
      );
    });

    it('should get facilities in an area', async () => {
      vi.mocked(client.callZome).mockResolvedValue([]);

      const result = await facilities.getFacilitiesInArea(32.7, -96.8, 5);

      expect(result).toEqual([]);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_facilities_in_area',
          payload: { lat: 32.7, lng: -96.8, radius_km: 5 },
        }),
      );
    });

    it('should report outage response', async () => {
      const response: FacilityOutageResponse = {
        facilityId: 'fac-001',
        outageId: 'outage-001',
        backupActivated: true,
        backupRemainingHours: 48,
        criticalPatientsAffected: 15,
        evacuationRequired: false,
        status: 'BackupPower',
        lastUpdated: Date.now(),
      };
      vi.mocked(client.callZome).mockResolvedValue(response);

      const result = await facilities.reportOutageResponse({
        facilityId: 'fac-001',
        outageId: 'outage-001',
        backupActivated: true,
        backupRemainingHours: 48,
        criticalPatientsAffected: 15,
        evacuationRequired: false,
        status: 'BackupPower',
      });

      expect(result.status).toBe('BackupPower');
    });
  });

  describe('HealthEnergyBridge', () => {
    let client: ZomeCallable;
    let bridge: HealthEnergyBridge;

    beforeEach(() => {
      client = mockZomeClient();
      bridge = new HealthEnergyBridge(client);
    });

    it('should expose all sub-clients', () => {
      expect(bridge.devices).toBeInstanceOf(MedicalDeviceClient);
      expect(bridge.priorities).toBeInstanceOf(MedicalPriorityClient);
      expect(bridge.facilities).toBeInstanceOf(HealthcareFacilityClient);
    });

    it('should handle power outage event', async () => {
      vi.mocked(client.callZome)
        .mockResolvedValueOnce([makeDevice()]) // getCriticalDevicesInArea
        .mockResolvedValueOnce({
          // createPriorityAlert
          alertId: 'alert-1',
          outageId: 'out-1',
          patientDid: 'did:mycelix:p1',
          deviceId: 'dev-001',
          powerPriority: 'Critical',
          medicalConditions: ['respiratory_failure'],
          requiresImmediateResponse: true,
          createdAt: Date.now(),
        })
        .mockResolvedValueOnce([]); // getFacilitiesInArea

      const outage: PowerOutageEvent = {
        outageId: 'out-1',
        location: { lat: 32.7, lng: -96.8 },
        radiusKm: 5,
        startedAt: Date.now(),
        affectedArea: 'Downtown',
        severity: 'Major',
      };

      const result = await bridge.handlePowerOutage(outage);

      expect(result.affectedDevices).toHaveLength(1);
      expect(result.alerts).toHaveLength(1);
      expect(result.criticalCount).toBe(1);
    });

    it('should handle power restored event', async () => {
      vi.mocked(client.callZome)
        .mockResolvedValueOnce([
          // getOutageAlerts
          {
            alertId: 'alert-1',
            outageId: 'out-1',
            patientDid: 'did:mycelix:p1',
            deviceId: 'dev-001',
            powerPriority: 'Critical',
            medicalConditions: [],
            requiresImmediateResponse: true,
            createdAt: Date.now(),
          },
        ])
        .mockResolvedValueOnce(undefined) // resolveAlert
        .mockResolvedValueOnce([]); // getOutageFacilityResponses

      const result = await bridge.handlePowerRestored('out-1');

      expect(result.resolvedAlerts).toBe(1);
      expect(result.updatedFacilities).toBe(0);
    });
  });

  describe('calculatePriorityScore (utility)', () => {
    it('should assign highest score to life-support critical devices', () => {
      const device = makeDevice({
        powerPriority: 'Critical',
        category: 'LifeSupport',
        criticalMedicalConditions: ['respiratory_failure', 'cardiac'],
      });
      const score = calculatePriorityScore(device);

      expect(score).toBeGreaterThanOrEqual(1500);
    });

    it('should increase score for low battery', () => {
      const device = makeDevice();
      const scoreWithBattery = calculatePriorityScore(device, 120);
      const scoreLowBattery = calculatePriorityScore(device, 10);

      expect(scoreLowBattery).toBeGreaterThan(scoreWithBattery);
    });

    it('should increase score for no battery backup', () => {
      const noBattery = makeDevice({ batteryBackupMinutes: 0 });
      const withBattery = makeDevice({ batteryBackupMinutes: 120 });

      const scoreNoBattery = calculatePriorityScore(noBattery);
      const scoreWithBattery = calculatePriorityScore(withBattery);

      expect(scoreNoBattery).toBeGreaterThan(scoreWithBattery);
    });

    it('should give lower score to standard priority devices', () => {
      const standard = makeDevice({
        powerPriority: 'Standard',
        category: 'Other',
        criticalMedicalConditions: [],
      });
      const critical = makeDevice();

      expect(calculatePriorityScore(standard)).toBeLessThan(
        calculatePriorityScore(critical),
      );
    });

    it('should increase score per critical medical condition', () => {
      const one = makeDevice({ criticalMedicalConditions: ['a'] });
      const three = makeDevice({ criticalMedicalConditions: ['a', 'b', 'c'] });

      expect(calculatePriorityScore(three)).toBeGreaterThan(
        calculatePriorityScore(one),
      );
    });
  });

  describe('estimateRestorationTime (utility)', () => {
    it('should return shorter time for high priority', () => {
      const highPriority = estimateRestorationTime(1500, 10, 3);
      const lowPriority = estimateRestorationTime(100, 10, 3);

      expect(highPriority).toBeLessThan(lowPriority);
    });

    it('should increase time for more affected devices', () => {
      const fewDevices = estimateRestorationTime(500, 5, 3);
      const manyDevices = estimateRestorationTime(500, 100, 3);

      expect(manyDevices).toBeGreaterThan(fewDevices);
    });
  });

  describe('requiresImmediateResponse (utility)', () => {
    it('should return true for LifeSupport category', () => {
      const device = makeDevice({ category: 'LifeSupport' });
      expect(requiresImmediateResponse(device)).toBe(true);
    });

    it('should return true for Critical priority with low battery', () => {
      const device = makeDevice({
        category: 'Monitoring',
        powerPriority: 'Critical',
      });
      expect(requiresImmediateResponse(device, 15)).toBe(true);
    });

    it('should return true for OxygenEquipment with no battery', () => {
      const device = makeDevice({
        category: 'OxygenEquipment',
        powerPriority: 'Essential',
        batteryBackupMinutes: 0,
      });
      expect(requiresImmediateResponse(device)).toBe(true);
    });

    it('should return false for standard monitoring device', () => {
      const device = makeDevice({
        category: 'Monitoring',
        powerPriority: 'Standard',
        batteryBackupMinutes: 120,
      });
      expect(requiresImmediateResponse(device, 60)).toBe(false);
    });
  });

  describe('createHealthEnergyBridge (factory)', () => {
    it('should create a HealthEnergyBridge instance', () => {
      const bridge = createHealthEnergyBridge(mockZomeClient());
      expect(bridge).toBeInstanceOf(HealthEnergyBridge);
    });
  });
});
