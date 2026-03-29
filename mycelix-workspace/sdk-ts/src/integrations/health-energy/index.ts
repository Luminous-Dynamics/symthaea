// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Health ↔ Energy Integration Module
 *
 * Provides cross-module bridges between mycelix-health and mycelix-energy.
 * Key use cases:
 * - Medical equipment power priority during outages
 * - Healthcare facility energy management
 * - Patient medical device tracking
 *
 * @module @mycelix/sdk/integrations/health-energy
 */

import type { ActionHash } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

/**
 * Medical equipment power priority levels
 */
export type PowerPriority =
  | 'Critical' // Life support - immediate power restoration required
  | 'Essential' // Medical equipment - restore within minutes
  | 'Important' // Comfort/monitoring - restore within hours
  | 'Standard'; // Non-medical - normal restoration priority

/**
 * Medical device categories that may require power
 */
export type MedicalDeviceCategory =
  | 'LifeSupport' // Ventilators, dialysis, LVAD
  | 'OxygenEquipment' // Concentrators, CPAP/BiPAP
  | 'Refrigeration' // Medication storage (insulin, biologics)
  | 'Monitoring' // Home health monitors, CGM
  | 'Mobility' // Electric wheelchairs, lifts
  | 'Communication' // Medical alert systems
  | 'Other';

/**
 * Medical device registration for power priority
 */
export interface MedicalDevice {
  deviceId: string;
  patientHash: ActionHash;
  category: MedicalDeviceCategory;
  powerPriority: PowerPriority;
  powerRequirementWatts: number;
  batteryBackupMinutes?: number;
  deviceName: string;
  manufacturer?: string;
  serialNumber?: string;
  criticalMedicalConditions: string[];
  registeredAt: number;
  updatedAt: number;
  active: boolean;
}

/**
 * Input for registering a medical device
 */
export interface RegisterDeviceInput {
  patientHash: ActionHash;
  category: MedicalDeviceCategory;
  powerPriority: PowerPriority;
  powerRequirementWatts: number;
  batteryBackupMinutes?: number;
  deviceName: string;
  manufacturer?: string;
  serialNumber?: string;
  criticalMedicalConditions?: string[];
}

/**
 * Power outage event
 */
export interface PowerOutageEvent {
  outageId: string;
  location: { lat: number; lng: number };
  radiusKm: number;
  startedAt: number;
  estimatedRestoration?: number;
  affectedArea: string;
  severity: 'Minor' | 'Moderate' | 'Major' | 'Critical';
}

/**
 * Medical priority alert for energy grid
 */
export interface MedicalPriorityAlert {
  alertId: string;
  outageId: string;
  patientDid: string;
  deviceId: string;
  powerPriority: PowerPriority;
  batteryRemainingMinutes?: number;
  medicalConditions: string[];
  contactPhone?: string;
  requiresImmediateResponse: boolean;
  createdAt: number;
}

/**
 * Healthcare facility for energy management
 */
export interface HealthcareFacility {
  facilityId: string;
  name: string;
  facilityType: 'Hospital' | 'Clinic' | 'NursingHome' | 'EmergencyCenter' | 'Pharmacy' | 'Other';
  location: { lat: number; lng: number; address: string };
  totalPowerCapacityKw: number;
  criticalLoadKw: number;
  hasBackupPower: boolean;
  backupCapacityHours?: number;
  primaryContact: string;
  emergencyContact: string;
  registeredAt: number;
}

/**
 * Response plan for facility during outage
 */
export interface FacilityOutageResponse {
  facilityId: string;
  outageId: string;
  backupActivated: boolean;
  backupRemainingHours?: number;
  criticalPatientsAffected: number;
  evacuationRequired: boolean;
  status: 'Operating' | 'BackupPower' | 'ReducedCapacity' | 'Evacuating' | 'Closed';
  lastUpdated: number;
}

// ============================================================================
// Zome Callable Interface
// ============================================================================

export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Health-Energy Bridge Client
// ============================================================================

const HEALTH_ROLE = 'health';
const ENERGY_ROLE = 'energy';

/**
 * Client for medical device power priority management
 */
export class MedicalDeviceClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Register a medical device for power priority
   */
  async registerDevice(input: RegisterDeviceInput): Promise<MedicalDevice> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'bridge',
      fn_name: 'register_medical_device',
      payload: input,
    }) as Promise<MedicalDevice>;
  }

  /**
   * Get all registered devices for a patient
   */
  async getPatientDevices(patientHash: ActionHash): Promise<MedicalDevice[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'bridge',
      fn_name: 'get_patient_devices',
      payload: patientHash,
    }) as Promise<MedicalDevice[]>;
  }

  /**
   * Update device status (active/inactive)
   */
  async updateDeviceStatus(
    deviceId: string,
    active: boolean
  ): Promise<MedicalDevice> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'bridge',
      fn_name: 'update_device_status',
      payload: { device_id: deviceId, active },
    }) as Promise<MedicalDevice>;
  }

  /**
   * Get all critical devices in an area (for grid operators)
   */
  async getCriticalDevicesInArea(
    lat: number,
    lng: number,
    radiusKm: number
  ): Promise<MedicalDevice[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'bridge',
      fn_name: 'get_critical_devices_in_area',
      payload: { lat, lng, radius_km: radiusKm },
    }) as Promise<MedicalDevice[]>;
  }
}

/**
 * Client for power outage medical priority alerts
 */
export class MedicalPriorityClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Create a medical priority alert during outage
   */
  async createPriorityAlert(
    outageId: string,
    deviceId: string,
    batteryRemainingMinutes?: number
  ): Promise<MedicalPriorityAlert> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'priorities',
      fn_name: 'create_medical_priority_alert',
      payload: {
        outage_id: outageId,
        device_id: deviceId,
        battery_remaining_minutes: batteryRemainingMinutes,
      },
    }) as Promise<MedicalPriorityAlert>;
  }

  /**
   * Get all active alerts for an outage
   */
  async getOutageAlerts(outageId: string): Promise<MedicalPriorityAlert[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'priorities',
      fn_name: 'get_outage_medical_alerts',
      payload: outageId,
    }) as Promise<MedicalPriorityAlert[]>;
  }

  /**
   * Resolve an alert (power restored)
   */
  async resolveAlert(alertId: string): Promise<void> {
    await this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'priorities',
      fn_name: 'resolve_medical_alert',
      payload: alertId,
    });
  }

  /**
   * Get sorted priority list for power restoration
   * Returns devices in order of restoration priority
   */
  async getRestorationPriorityList(
    outageId: string
  ): Promise<
    Array<{
      device: MedicalDevice;
      alert?: MedicalPriorityAlert;
      priorityScore: number;
      estimatedTimeToRestore: number;
    }>
  > {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'priorities',
      fn_name: 'get_restoration_priority_list',
      payload: outageId,
    }) as Promise<
      Array<{
        device: MedicalDevice;
        alert?: MedicalPriorityAlert;
        priorityScore: number;
        estimatedTimeToRestore: number;
      }>
    >;
  }
}

/**
 * Client for healthcare facility energy management
 */
export class HealthcareFacilityClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Register a healthcare facility
   */
  async registerFacility(
    facility: Omit<HealthcareFacility, 'facilityId' | 'registeredAt'>
  ): Promise<HealthcareFacility> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'facilities',
      fn_name: 'register_healthcare_facility',
      payload: facility,
    }) as Promise<HealthcareFacility>;
  }

  /**
   * Get facilities in an area
   */
  async getFacilitiesInArea(
    lat: number,
    lng: number,
    radiusKm: number
  ): Promise<HealthcareFacility[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'facilities',
      fn_name: 'get_facilities_in_area',
      payload: { lat, lng, radius_km: radiusKm },
    }) as Promise<HealthcareFacility[]>;
  }

  /**
   * Report facility outage response status
   */
  async reportOutageResponse(
    response: Omit<FacilityOutageResponse, 'lastUpdated'>
  ): Promise<FacilityOutageResponse> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'facilities',
      fn_name: 'report_outage_response',
      payload: response,
    }) as Promise<FacilityOutageResponse>;
  }

  /**
   * Get all facility responses for an outage
   */
  async getOutageFacilityResponses(
    outageId: string
  ): Promise<FacilityOutageResponse[]> {
    return this.client.callZome({
      role_name: ENERGY_ROLE,
      zome_name: 'facilities',
      fn_name: 'get_outage_facility_responses',
      payload: outageId,
    }) as Promise<FacilityOutageResponse[]>;
  }
}

// ============================================================================
// Unified Health-Energy Bridge
// ============================================================================

/**
 * Unified interface for health-energy integration
 */
export class HealthEnergyBridge {
  readonly devices: MedicalDeviceClient;
  readonly priorities: MedicalPriorityClient;
  readonly facilities: HealthcareFacilityClient;

  constructor(client: ZomeCallable) {
    this.devices = new MedicalDeviceClient(client);
    this.priorities = new MedicalPriorityClient(client);
    this.facilities = new HealthcareFacilityClient(client);
  }

  /**
   * Handle power outage event - creates alerts for all affected medical devices
   *
   * This is the main entry point for grid operators when an outage occurs.
   * It automatically identifies affected medical devices and creates priority alerts.
   */
  async handlePowerOutage(
    outage: PowerOutageEvent
  ): Promise<{
    affectedDevices: MedicalDevice[];
    alerts: MedicalPriorityAlert[];
    criticalCount: number;
    essentialCount: number;
    facilitiesAffected: HealthcareFacility[];
  }> {
    // Get all critical/essential devices in the affected area
    const affectedDevices = await this.devices.getCriticalDevicesInArea(
      outage.location.lat,
      outage.location.lng,
      outage.radiusKm
    );

    // Create alerts for each device
    const alerts: MedicalPriorityAlert[] = [];
    for (const device of affectedDevices) {
      try {
        const alert = await this.priorities.createPriorityAlert(
          outage.outageId,
          device.deviceId,
          device.batteryBackupMinutes
        );
        alerts.push(alert);
      } catch (err) {
        console.error(`Failed to create alert for device ${device.deviceId}:`, err);
      }
    }

    // Get affected healthcare facilities
    const facilitiesAffected = await this.facilities.getFacilitiesInArea(
      outage.location.lat,
      outage.location.lng,
      outage.radiusKm
    );

    return {
      affectedDevices,
      alerts,
      criticalCount: affectedDevices.filter(d => d.powerPriority === 'Critical').length,
      essentialCount: affectedDevices.filter(d => d.powerPriority === 'Essential').length,
      facilitiesAffected,
    };
  }

  /**
   * Notify power restoration - resolves alerts and updates device status
   */
  async handlePowerRestored(outageId: string): Promise<{
    resolvedAlerts: number;
    updatedFacilities: number;
  }> {
    // Get all active alerts for this outage
    const alerts = await this.priorities.getOutageAlerts(outageId);

    let resolvedCount = 0;
    for (const alert of alerts) {
      try {
        await this.priorities.resolveAlert(alert.alertId);
        resolvedCount++;
      } catch (err) {
        console.error(`Failed to resolve alert ${alert.alertId}:`, err);
      }
    }

    // Get facility responses to update
    const facilityResponses = await this.facilities.getOutageFacilityResponses(outageId);
    let updatedFacilities = 0;

    for (const response of facilityResponses) {
      if (response.status !== 'Operating') {
        try {
          await this.facilities.reportOutageResponse({
            ...response,
            status: 'Operating',
            backupActivated: false,
          });
          updatedFacilities++;
        } catch (err) {
          console.error(`Failed to update facility ${response.facilityId}:`, err);
        }
      }
    }

    return {
      resolvedAlerts: resolvedCount,
      updatedFacilities,
    };
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a Health-Energy bridge instance
 */
export function createHealthEnergyBridge(client: ZomeCallable): HealthEnergyBridge {
  return new HealthEnergyBridge(client);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate priority score for power restoration
 *
 * Higher score = higher priority for restoration
 */
export function calculatePriorityScore(
  device: MedicalDevice,
  batteryRemainingMinutes?: number
): number {
  let score = 0;

  // Base priority by category
  switch (device.powerPriority) {
    case 'Critical':
      score += 1000;
      break;
    case 'Essential':
      score += 500;
      break;
    case 'Important':
      score += 200;
      break;
    case 'Standard':
      score += 50;
      break;
  }

  // Device category modifier
  switch (device.category) {
    case 'LifeSupport':
      score += 500;
      break;
    case 'OxygenEquipment':
      score += 300;
      break;
    case 'Refrigeration':
      score += 200; // Medication can spoil
      break;
    case 'Monitoring':
      score += 100;
      break;
    case 'Mobility':
      score += 50;
      break;
    case 'Communication':
      score += 150; // Medical alerts are important
      break;
    default:
      score += 25;
  }

  // Battery urgency modifier (less battery = higher priority)
  if (batteryRemainingMinutes !== undefined) {
    if (batteryRemainingMinutes <= 15) {
      score += 500; // Critical - immediate action needed
    } else if (batteryRemainingMinutes <= 30) {
      score += 300;
    } else if (batteryRemainingMinutes <= 60) {
      score += 150;
    } else if (batteryRemainingMinutes <= 120) {
      score += 50;
    }
  } else if (!device.batteryBackupMinutes || device.batteryBackupMinutes === 0) {
    // No battery backup - needs power immediately
    score += 200;
  }

  // Number of critical conditions modifier
  score += device.criticalMedicalConditions.length * 50;

  return score;
}

/**
 * Estimate time to power restoration based on priority and grid capacity
 */
export function estimateRestorationTime(
  priorityScore: number,
  totalAffectedDevices: number,
  gridCrewsAvailable: number = 3
): number {
  // Base restoration time in minutes
  const baseTime = 60;

  // Reduce time for higher priority
  const priorityMultiplier = priorityScore > 1000 ? 0.5 : priorityScore > 500 ? 0.75 : 1.0;

  // Increase time for more affected devices
  const loadMultiplier = Math.min(totalAffectedDevices / (gridCrewsAvailable * 10), 3);

  return Math.round(baseTime * priorityMultiplier * loadMultiplier);
}

/**
 * Check if device requires immediate response (life-threatening)
 */
export function requiresImmediateResponse(
  device: MedicalDevice,
  batteryRemainingMinutes?: number
): boolean {
  // Life support always requires immediate response
  if (device.category === 'LifeSupport') {
    return true;
  }

  // Critical priority with low battery
  if (
    device.powerPriority === 'Critical' &&
    batteryRemainingMinutes !== undefined &&
    batteryRemainingMinutes <= 30
  ) {
    return true;
  }

  // Oxygen equipment with no battery
  if (
    device.category === 'OxygenEquipment' &&
    (!device.batteryBackupMinutes || device.batteryBackupMinutes === 0)
  ) {
    return true;
  }

  return false;
}

// ============================================================================
// Exports
// ============================================================================

export default {
  HealthEnergyBridge,
  MedicalDeviceClient,
  MedicalPriorityClient,
  HealthcareFacilityClient,
  createHealthEnergyBridge,
  calculatePriorityScore,
  estimateRestorationTime,
  requiresImmediateResponse,
};
