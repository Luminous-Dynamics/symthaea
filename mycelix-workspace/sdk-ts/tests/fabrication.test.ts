// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk FabricationService Unit Tests
 *
 * Comprehensive tests for the Fabrication integration including:
 * - Design CRUD operations
 * - Printer registration and matching
 * - Print job lifecycle
 * - PoGF score calculation
 * - Anticipatory repair workflows
 * - Bridge event emission
 * - Reputation management
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  FabricationService,
  getFabricationService,
  // Types
  type Design,
  type Printer,
  type PrintJob,
  type PrinterMatch,
  type RepairPrediction,
  type RepairWorkflow,
  type QualityAssessment,
  type PrintSettings,
  type GroundingCertificate,
  type MaterialPassport,
  type DesignCategory,
  type SafetyClass,
  type PrinterType,
  type MaterialType,
  type AvailabilityStatus,
  type PrintResult,
} from '../src/integrations/fabrication/index.js';

describe('FabricationService', () => {
  let service: FabricationService;

  beforeEach(() => {
    // Create fresh instance for each test
    service = new FabricationService();
  });

  // ============================================================================
  // Design Operations
  // ============================================================================

  describe('Design Operations', () => {
    it('should create a design with minimal inputs', () => {
      const design = service.createDesign({
        title: 'Test Widget',
        category: 'Parts',
        safetyClass: 'Class1Functional',
      });

      expect(design).toBeDefined();
      expect(design.id).toMatch(/^design-\d+-[a-z0-9]+$/);
      expect(design.title).toBe('Test Widget');
      expect(design.category).toBe('Parts');
      expect(design.safetyClass).toBe('Class1Functional');
      expect(design.license).toBe('PublicDomain');
      expect(design.author).toBe('local-agent');
      expect(design.createdAt).toBeGreaterThan(0);
      expect(design.updatedAt).toBe(design.createdAt);
    });

    it('should create a design with full inputs', () => {
      const design = service.createDesign({
        title: 'Replacement Battery Clip',
        description: 'Universal replacement clip for power tool batteries',
        category: 'Repair',
        safetyClass: 'Class2LoadBearing',
        license: 'OpenHardware',
        files: [
          {
            filename: 'battery_clip.stl',
            format: 'STL',
            ipfsCid: 'QmTest123',
            sizeBytes: 1024,
            checksumSha256: 'abc123',
          },
        ],
        parametricSchema: {
          engine: 'OpenSCAD',
          sourceTemplate: 'QmTemplate456',
          parameters: [
            {
              name: 'clip_width',
              paramType: 'Length',
              defaultValue: 25,
              minValue: 20,
              maxValue: 35,
              unit: 'mm',
            },
          ],
          constraints: ['clip_width > 0'],
          autoGenerate: true,
        },
        repairManifest: {
          parentProductModel: 'DeWalt DCD771',
          partName: 'Battery Clip',
          failureModes: ['MechanicalWear', 'ImpactDamage'],
          replacementIntervalHours: 2000,
          repairDifficulty: 'Easy',
        },
      });

      expect(design.description).toBe('Universal replacement clip for power tool batteries');
      expect(design.license).toBe('OpenHardware');
      expect(design.files).toHaveLength(1);
      expect(design.files[0].format).toBe('STL');
      expect(design.parametricSchema?.engine).toBe('OpenSCAD');
      expect(design.parametricSchema?.parameters).toHaveLength(1);
      expect(design.repairManifest?.parentProductModel).toBe('DeWalt DCD771');
      expect(design.repairManifest?.failureModes).toContain('MechanicalWear');
    });

    it('should retrieve a design by ID', () => {
      const created = service.createDesign({
        title: 'Test Design',
        category: 'Tools',
        safetyClass: 'Class0Decorative',
      });

      const retrieved = service.getDesign(created.id);

      expect(retrieved).toBeDefined();
      expect(retrieved?.id).toBe(created.id);
      expect(retrieved?.title).toBe('Test Design');
    });

    it('should return undefined for non-existent design', () => {
      const result = service.getDesign('non-existent-id');
      expect(result).toBeUndefined();
    });

    it('should get designs by category', () => {
      service.createDesign({ title: 'Tool 1', category: 'Tools', safetyClass: 'Class0Decorative' });
      service.createDesign({ title: 'Tool 2', category: 'Tools', safetyClass: 'Class1Functional' });
      service.createDesign({ title: 'Part 1', category: 'Parts', safetyClass: 'Class1Functional' });
      service.createDesign({ title: 'Medical 1', category: 'Medical', safetyClass: 'Class4Medical' });

      const tools = service.getDesignsByCategory('Tools');
      const parts = service.getDesignsByCategory('Parts');
      const medical = service.getDesignsByCategory('Medical');
      const art = service.getDesignsByCategory('Art');

      expect(tools).toHaveLength(2);
      expect(parts).toHaveLength(1);
      expect(medical).toHaveLength(1);
      expect(art).toHaveLength(0);
    });

    it('should fork a design', () => {
      const parent = service.createDesign({
        title: 'Original Widget',
        description: 'Original description',
        category: 'Parts',
        safetyClass: 'Class1Functional',
        license: 'OpenHardware',
      });

      const forked = service.forkDesign(parent.id, {
        title: 'Modified Widget',
        description: 'Improved version',
      });

      expect(forked).toBeDefined();
      expect(forked?.id).not.toBe(parent.id);
      expect(forked?.title).toBe('Modified Widget');
      expect(forked?.description).toBe('Improved version');
      expect(forked?.category).toBe(parent.category);
      expect(forked?.safetyClass).toBe(parent.safetyClass);
      expect(forked?.license).toBe(parent.license);
    });

    it('should return undefined when forking non-existent design', () => {
      const result = service.forkDesign('non-existent', { title: 'Test' });
      expect(result).toBeUndefined();
    });

    it('should set default epistemic values', () => {
      const design = service.createDesign({
        title: 'Test',
        category: 'Parts',
        safetyClass: 'Class1Functional',
      });

      expect(design.epistemic.manufacturability).toBe(0.5);
      expect(design.epistemic.safety).toBe(0.5);
      expect(design.epistemic.usability).toBe(0.5);
    });
  });

  // ============================================================================
  // Printer Operations
  // ============================================================================

  describe('Printer Operations', () => {
    it('should register a printer with minimal inputs', () => {
      const printer = service.registerPrinter({
        name: 'Prusa MK3S+',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 250, y: 210, z: 210 },
          layerHeights: [0.1, 0.15, 0.2, 0.3],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: false,
          maxTempHotend: 285,
          maxTempBed: 100,
          features: ['auto-leveling', 'power-recovery'],
        },
        materialsAvailable: ['PLA', 'PETG', 'ABS'],
      });

      expect(printer).toBeDefined();
      expect(printer.id).toMatch(/^printer-\d+-[a-z0-9]+$/);
      expect(printer.name).toBe('Prusa MK3S+');
      expect(printer.printerType).toBe('FDM');
      expect(printer.availability).toBe('Available');
      expect(printer.capabilities.buildVolume.x).toBe(250);
      expect(printer.materialsAvailable).toContain('PETG');
    });

    it('should register a printer with location and rates', () => {
      const printer = service.registerPrinter({
        name: 'Community Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: true,
          maxTempHotend: 300,
          maxTempBed: 110,
          features: [],
        },
        materialsAvailable: ['PLA'],
        location: {
          geohash: '9q8yy',
          city: 'San Francisco',
          region: 'California',
          country: 'USA',
        },
        rates: {
          baseFee: 5,
          perHour: 2,
          perGram: 0.05,
        },
      });

      expect(printer.location?.city).toBe('San Francisco');
      expect(printer.rates?.baseFee).toBe(5);
      expect(printer.rates?.perHour).toBe(2);
    });

    it('should find printers by minimum build volume', () => {
      service.registerPrinter({
        name: 'Small Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 100, y: 100, z: 100 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: false,
          maxTempHotend: 250,
          maxTempBed: 80,
          features: [],
        },
        materialsAvailable: ['PLA'],
      });

      service.registerPrinter({
        name: 'Large Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 300, y: 300, z: 400 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: true,
          maxTempHotend: 300,
          maxTempBed: 110,
          features: [],
        },
        materialsAvailable: ['PLA', 'ABS'],
      });

      const largeOnly = service.findPrintersByCapability({
        minBuildVolume: { x: 200, y: 200, z: 200 },
      });

      expect(largeOnly).toHaveLength(1);
      expect(largeOnly[0].name).toBe('Large Printer');
    });

    it('should find printers by required materials', () => {
      service.registerPrinter({
        name: 'Basic Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: false,
          maxTempHotend: 250,
          maxTempBed: 80,
          features: [],
        },
        materialsAvailable: ['PLA'],
      });

      service.registerPrinter({
        name: 'Advanced Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: true,
          maxTempHotend: 300,
          maxTempBed: 110,
          features: [],
        },
        materialsAvailable: ['PLA', 'PETG', 'ABS', 'ASA'],
      });

      const absCapable = service.findPrintersByCapability({
        requiredMaterials: ['ABS'],
      });

      const asaCapable = service.findPrintersByCapability({
        requiredMaterials: ['ASA'],
      });

      expect(absCapable).toHaveLength(1);
      expect(absCapable[0].name).toBe('Advanced Printer');
      expect(asaCapable).toHaveLength(1);
    });

    it('should find printers by enclosure requirement', () => {
      service.registerPrinter({
        name: 'Open Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: false,
          maxTempHotend: 250,
          maxTempBed: 80,
          features: [],
        },
        materialsAvailable: ['PLA'],
      });

      service.registerPrinter({
        name: 'Enclosed Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: true,
          maxTempHotend: 300,
          maxTempBed: 110,
          features: [],
        },
        materialsAvailable: ['PLA', 'ABS'],
      });

      const enclosed = service.findPrintersByCapability({
        requireEnclosure: true,
      });

      expect(enclosed).toHaveLength(1);
      expect(enclosed[0].name).toBe('Enclosed Printer');
    });

    it('should match design to printers', () => {
      const design = service.createDesign({
        title: 'Test Part',
        category: 'Parts',
        safetyClass: 'Class1Functional',
      });

      service.registerPrinter({
        name: 'Printer 1',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: false,
          maxTempHotend: 250,
          maxTempBed: 80,
          features: [],
        },
        materialsAvailable: ['PLA'],
        rates: { baseFee: 5, perHour: 2, perGram: 0.05 },
      });

      service.registerPrinter({
        name: 'Printer 2',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 300, y: 300, z: 300 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: true,
          maxTempHotend: 300,
          maxTempBed: 110,
          features: [],
        },
        materialsAvailable: ['PLA', 'PETG'],
      });

      const matches = service.matchDesignToPrinters(design.id);

      expect(matches).toHaveLength(2);
      expect(matches[0].compatibilityScore).toBeGreaterThanOrEqual(0);
      expect(matches[0].compatibilityScore).toBeLessThanOrEqual(1);
      expect(matches[0].estimatedTime).toBeGreaterThan(0);
      // First printer has rates, so estimated cost should be defined
      expect(matches.some((m) => m.estimatedCost !== undefined)).toBe(true);
    });

    it('should return empty array when matching non-existent design', () => {
      const matches = service.matchDesignToPrinters('non-existent');
      expect(matches).toHaveLength(0);
    });

    it('should update printer availability', () => {
      const printer = service.registerPrinter({
        name: 'Test Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: false,
          maxTempHotend: 250,
          maxTempBed: 80,
          features: [],
        },
        materialsAvailable: ['PLA'],
      });

      expect(printer.availability).toBe('Available');

      service.updatePrinterAvailability(printer.id, 'Maintenance');

      // Get fresh reference - the map contains the updated printer
      const printers = service.findPrintersByCapability({});
      const updated = printers.find((p) => p.id === printer.id);
      expect(updated?.availability).toBe('Maintenance');
    });
  });

  // ============================================================================
  // Print Job Operations
  // ============================================================================

  describe('Print Job Operations', () => {
    let design: Design;
    let printer: Printer;

    beforeEach(() => {
      design = service.createDesign({
        title: 'Test Part',
        category: 'Parts',
        safetyClass: 'Class1Functional',
      });

      printer = service.registerPrinter({
        name: 'Test Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: false,
          maxTempHotend: 250,
          maxTempBed: 60,
          features: [],
        },
        materialsAvailable: ['PLA'],
      });
    });

    it('should create a print job', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      expect(job).toBeDefined();
      expect(job?.id).toMatch(/^job-\d+-[a-z0-9]+$/);
      expect(job?.designHash).toBe(design.id);
      expect(job?.printerHash).toBe(printer.id);
      expect(job?.status).toBe('Pending');
      expect(job?.settings.layerHeight).toBe(0.2);
      expect(job?.settings.temperatures.hotend).toBe(200); // Default
      expect(job?.settings.temperatures.bed).toBe(60); // Default
    });

    it('should create a print job with grounding request', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
        groundingRequest: {
          requireRenewable: true,
          requireRecycledMaterial: true,
          targetPogScore: 0.8,
        },
      });

      expect(job?.groundingRequest).toBeDefined();
      expect(job?.groundingRequest?.requireRenewable).toBe(true);
      expect(job?.groundingRequest?.targetPogScore).toBe(0.8);
    });

    it('should return undefined for invalid design/printer', () => {
      const job1 = service.createPrintJob({
        designId: 'invalid',
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      const job2 = service.createPrintJob({
        designId: design.id,
        printerId: 'invalid',
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      expect(job1).toBeUndefined();
      expect(job2).toBeUndefined();
    });

    it('should accept a print job', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      expect(job?.status).toBe('Pending');

      const accepted = service.acceptPrintJob(job!.id);
      expect(accepted).toBe(true);
    });

    it('should not accept already accepted job', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      service.acceptPrintJob(job!.id);
      const secondAccept = service.acceptPrintJob(job!.id);

      expect(secondAccept).toBe(false);
    });

    it('should start printing', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      service.acceptPrintJob(job!.id);
      const started = service.startPrint(job!.id);

      expect(started).toBe(true);
    });

    it('should update printer availability when printing starts', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      service.acceptPrintJob(job!.id);
      service.startPrint(job!.id);

      // Printer should now be busy
      const matches = service.matchDesignToPrinters(design.id);
      expect(matches).toHaveLength(0); // No available printers
    });

    it('should not start print from wrong status', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      // Try to start from Pending (should fail)
      const started = service.startPrint(job!.id);
      expect(started).toBe(false);
    });

    it('should complete a print job', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      service.acceptPrintJob(job!.id);
      service.startPrint(job!.id);

      const record = service.completePrint(
        job!.id,
        'Success',
        {
          dimensionalAccuracy: 0.95,
          surfaceQuality: 0.90,
          structuralIntegrity: 0.92,
        },
        45
      );

      expect(record).toBeDefined();
      expect(record?.result).toBe('Success');
      expect(record?.qualityScore).toBeCloseTo(0.923, 2);
      expect(record?.pogScore).toBeGreaterThan(0);
      expect(record?.myceliumEarned).toBeGreaterThan(0);
      expect(record?.recordedAt).toBeGreaterThan(0);
    });

    it('should not complete job that is not printing', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      const record = service.completePrint(
        job!.id,
        'Success',
        { dimensionalAccuracy: 0.9, surfaceQuality: 0.9, structuralIntegrity: 0.9 },
        45
      );

      expect(record).toBeUndefined();
    });

    it('should restore printer availability after completion', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      service.acceptPrintJob(job!.id);
      service.startPrint(job!.id);
      service.completePrint(
        job!.id,
        'Success',
        { dimensionalAccuracy: 0.9, surfaceQuality: 0.9, structuralIntegrity: 0.9 },
        45
      );

      // Printer should be available again
      const matches = service.matchDesignToPrinters(design.id);
      expect(matches).toHaveLength(1);
    });
  });

  // ============================================================================
  // PoGF Score Calculation
  // ============================================================================

  describe('PoGF Score Calculation', () => {
    let design: Design;
    let printer: Printer;

    beforeEach(() => {
      design = service.createDesign({
        title: 'Test Part',
        category: 'Parts',
        safetyClass: 'Class1Functional',
      });

      printer = service.registerPrinter({
        name: 'Test Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: false,
          maxTempHotend: 250,
          maxTempBed: 60,
          features: [],
        },
        materialsAvailable: ['PLA'],
      });
    });

    it('should calculate base PoGF score without grounding certificate', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      service.acceptPrintJob(job!.id);
      service.startPrint(job!.id);

      const record = service.completePrint(
        job!.id,
        'Success',
        { dimensionalAccuracy: 0.9, surfaceQuality: 0.9, structuralIntegrity: 0.9 },
        45
      );

      // Base score without grounding: ~0.32
      // energyScore: 0.3 (GridMix default) * 0.3 = 0.09
      // materialScore: 0.2 (default) * 0.3 = 0.06
      // qualityScore: 0.9 * 0.2 = 0.18
      // localScore: 0.2 (no HEARTH) * 0.2 = 0.04
      // Total: ~0.37
      expect(record?.pogScore).toBeGreaterThan(0.3);
      expect(record?.pogScore).toBeLessThan(0.5);
    });

    it('should award mycelium based on PoGF score', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      service.acceptPrintJob(job!.id);
      service.startPrint(job!.id);

      const record = service.completePrint(
        job!.id,
        'Success',
        { dimensionalAccuracy: 1.0, surfaceQuality: 1.0, structuralIntegrity: 1.0 },
        45
      );

      // Mycelium = PoGF * 100
      expect(record?.myceliumEarned).toBe(Math.round(record!.pogScore * 100));
    });
  });

  // ============================================================================
  // Reputation Operations
  // ============================================================================

  describe('Reputation Operations', () => {
    let design: Design;
    let printer: Printer;

    beforeEach(() => {
      design = service.createDesign({
        title: 'Test Part',
        category: 'Parts',
        safetyClass: 'Class1Functional',
      });

      printer = service.registerPrinter({
        name: 'Test Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: false,
          maxTempHotend: 250,
          maxTempBed: 60,
          features: [],
        },
        materialsAvailable: ['PLA'],
      });
    });

    it('should return default trust score for new printer', () => {
      const score = service.getPrinterTrustScore(printer.id);
      expect(score).toBe(0.5); // Default for unknown
    });

    it('should increase trust after successful print', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      service.acceptPrintJob(job!.id);
      service.startPrint(job!.id);
      service.completePrint(
        job!.id,
        'Success',
        { dimensionalAccuracy: 0.9, surfaceQuality: 0.9, structuralIntegrity: 0.9 },
        45
      );

      const score = service.getPrinterTrustScore(printer.id);
      expect(score).toBeGreaterThan(0.5);
    });

    it('should decrease trust after failed print', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });

      service.acceptPrintJob(job!.id);
      service.startPrint(job!.id);
      service.completePrint(
        job!.id,
        { Failed: 'Nozzle clog' },
        { dimensionalAccuracy: 0.1, surfaceQuality: 0.1, structuralIntegrity: 0.1 },
        15
      );

      const score = service.getPrinterTrustScore(printer.id);
      expect(score).toBeLessThan(0.5);
    });

    it('should check if printer is trustworthy', () => {
      // Initially not trustworthy (score is 0.5, default threshold 0.7)
      expect(service.isPrinterTrustworthy(printer.id)).toBe(false);

      // After multiple successful prints
      for (let i = 0; i < 5; i++) {
        const job = service.createPrintJob({
          designId: design.id,
          printerId: printer.id,
          settings: {
            layerHeight: 0.2,
            infillPercent: 20,
            material: 'PLA',
            supports: false,
            raft: false,
          },
        });
        service.acceptPrintJob(job!.id);
        service.startPrint(job!.id);
        service.completePrint(
          job!.id,
          'Success',
          { dimensionalAccuracy: 0.95, surfaceQuality: 0.95, structuralIntegrity: 0.95 },
          45
        );
      }

      // Should now be trustworthy
      expect(service.isPrinterTrustworthy(printer.id)).toBe(true);
    });

    it('should respect custom trustworthiness threshold', () => {
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 20,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });
      service.acceptPrintJob(job!.id);
      service.startPrint(job!.id);
      service.completePrint(
        job!.id,
        'Success',
        { dimensionalAccuracy: 0.9, surfaceQuality: 0.9, structuralIntegrity: 0.9 },
        45
      );

      const score = service.getPrinterTrustScore(printer.id);

      // With low threshold, should be trustworthy
      expect(service.isPrinterTrustworthy(printer.id, 0.5)).toBe(true);

      // With high threshold, may not be trustworthy
      expect(service.isPrinterTrustworthy(printer.id, 0.95)).toBe(false);
    });
  });

  // ============================================================================
  // Anticipatory Repair Operations
  // ============================================================================

  describe('Anticipatory Repair Operations', () => {
    it('should process repair prediction and find matching design', () => {
      // Create a repair design
      service.createDesign({
        title: 'Battery Clip Replacement',
        category: 'Repair',
        safetyClass: 'Class2LoadBearing',
        repairManifest: {
          parentProductModel: 'DeWalt DCD771',
          partName: 'Battery Clip',
          failureModes: ['MechanicalWear'],
          repairDifficulty: 'Easy',
        },
      });

      // Create a printer
      service.registerPrinter({
        name: 'Local Printer',
        printerType: 'FDM',
        capabilities: {
          buildVolume: { x: 200, y: 200, z: 200 },
          layerHeights: [0.2],
          nozzleDiameters: [0.4],
          heatedBed: true,
          enclosure: false,
          maxTempHotend: 250,
          maxTempBed: 60,
          features: [],
        },
        materialsAvailable: ['PLA', 'PETG'],
      });

      const prediction: RepairPrediction = {
        propertyAssetHash: 'asset-123',
        assetModel: 'DeWalt DCD771',
        predictedFailureComponent: 'Battery Clip',
        failureProbability: 0.85,
        estimatedFailureDate: Date.now() + 30 * 24 * 60 * 60 * 1000,
        confidenceIntervalDays: 14,
        sensorDataSummary: 'Vibration patterns indicate wear',
        recommendedAction: 'PrintReplacement',
        createdAt: Date.now(),
      };

      const workflow = service.processRepairPrediction(prediction);

      expect(workflow).toBeDefined();
      expect(workflow.status).toBe('PrinterMatched'); // Design found + printer matched
      expect(workflow.designHash).toBeDefined();
      expect(workflow.printerHash).toBeDefined();
    });

    it('should return Predicted status when no matching design', () => {
      const prediction: RepairPrediction = {
        propertyAssetHash: 'asset-456',
        assetModel: 'Unknown Brand Model X',
        predictedFailureComponent: 'Widget',
        failureProbability: 0.7,
        estimatedFailureDate: Date.now() + 60 * 24 * 60 * 60 * 1000,
        confidenceIntervalDays: 30,
        sensorDataSummary: 'General wear detected',
        recommendedAction: 'CreateDesign',
        createdAt: Date.now(),
      };

      const workflow = service.processRepairPrediction(prediction);

      expect(workflow).toBeDefined();
      expect(workflow.status).toBe('Predicted');
      expect(workflow.designHash).toBeUndefined();
      expect(workflow.printerHash).toBeUndefined();
    });

    it('should return DesignFound when design exists but no printer available', () => {
      // Create only a repair design, no printer
      service.createDesign({
        title: 'Rare Part',
        category: 'Repair',
        safetyClass: 'Class1Functional',
        repairManifest: {
          parentProductModel: 'Rare Machine',
          partName: 'Rare Part',
          failureModes: ['Fatigue'],
          repairDifficulty: 'Expert',
        },
      });

      const prediction: RepairPrediction = {
        propertyAssetHash: 'asset-789',
        assetModel: 'Rare Machine',
        predictedFailureComponent: 'Rare Part',
        failureProbability: 0.6,
        estimatedFailureDate: Date.now() + 90 * 24 * 60 * 60 * 1000,
        confidenceIntervalDays: 45,
        sensorDataSummary: 'Fatigue detected',
        recommendedAction: 'PrintReplacement',
        createdAt: Date.now(),
      };

      const workflow = service.processRepairPrediction(prediction);

      expect(workflow.status).toBe('DesignFound');
      expect(workflow.designHash).toBeDefined();
      expect(workflow.printerHash).toBeUndefined();
    });
  });

  // ============================================================================
  // Bridge Event Emission
  // ============================================================================

  describe('Bridge Event Emission', () => {
    it('should emit design published event', () => {
      const design = service.createDesign({
        title: 'New Design',
        category: 'Parts',
        safetyClass: 'Class1Functional',
      });

      // Should not throw
      expect(() => {
        service.emitEvent('DesignPublished', design.id, { title: design.title });
      }).not.toThrow();
    });

    it('should emit print completed event', () => {
      expect(() => {
        service.emitEvent('PrintCompleted', undefined, {
          jobId: 'job-123',
          result: 'Success',
          pogScore: 0.85,
        });
      }).not.toThrow();
    });

    it('should emit material shortage event', () => {
      expect(() => {
        service.emitEvent('MaterialShortage', undefined, {
          material: 'PETG',
          location: 'SF Bay Area',
        });
      }).not.toThrow();
    });
  });

  // ============================================================================
  // Singleton Pattern
  // ============================================================================

  describe('Singleton Pattern', () => {
    it('should return same instance from getFabricationService', () => {
      const service1 = getFabricationService();
      const service2 = getFabricationService();

      expect(service1).toBe(service2);
    });

    it('should preserve data across getInstance calls', () => {
      const service1 = getFabricationService();
      const design = service1.createDesign({
        title: 'Persistent Design',
        category: 'Art',
        safetyClass: 'Class0Decorative',
      });

      const service2 = getFabricationService();
      const retrieved = service2.getDesign(design.id);

      expect(retrieved).toBeDefined();
      expect(retrieved?.title).toBe('Persistent Design');
    });
  });

  // ============================================================================
  // FL Coordinator
  // ============================================================================

  describe('FL Coordinator', () => {
    it('should return FL coordinator instance', () => {
      const coordinator = service.getFLCoordinator();
      expect(coordinator).toBeDefined();
    });
  });

  // ============================================================================
  // Type Safety Tests
  // ============================================================================

  describe('Type Safety', () => {
    it('should accept all valid design categories', () => {
      const categories: DesignCategory[] = [
        'Tools',
        'Parts',
        'Housewares',
        'Medical',
        'Accessibility',
        'Art',
        'Education',
        'Repair',
        'Custom',
      ];

      categories.forEach((category) => {
        const design = service.createDesign({
          title: `${category} Design`,
          category,
          safetyClass: 'Class0Decorative',
        });
        expect(design.category).toBe(category);
      });
    });

    it('should accept all valid safety classes', () => {
      const safetyClasses: SafetyClass[] = [
        'Class0Decorative',
        'Class1Functional',
        'Class2LoadBearing',
        'Class3BodyContact',
        'Class4Medical',
        'Class5Critical',
      ];

      safetyClasses.forEach((safetyClass) => {
        const design = service.createDesign({
          title: `${safetyClass} Design`,
          category: 'Parts',
          safetyClass,
        });
        expect(design.safetyClass).toBe(safetyClass);
      });
    });

    it('should accept all valid printer types', () => {
      const printerTypes: PrinterType[] = ['FDM', 'SLA', 'SLS', 'DLP', 'MJF', { Other: 'Custom' }];

      printerTypes.forEach((printerType, i) => {
        const printer = service.registerPrinter({
          name: `Printer ${i}`,
          printerType,
          capabilities: {
            buildVolume: { x: 100, y: 100, z: 100 },
            layerHeights: [0.1],
            nozzleDiameters: [0.4],
            heatedBed: true,
            enclosure: false,
            maxTempHotend: 250,
            maxTempBed: 60,
            features: [],
          },
          materialsAvailable: ['PLA'],
        });
        expect(printer.printerType).toEqual(printerType);
      });
    });
  });
});
