/**
 * Fabrication Integration Tests
 *
 * Tests for FabricationService -- the domain-specific SDK service for
 * parametric design management, Proof of Grounded Fabrication (PoGF)
 * scoring, print job lifecycle, printer reputation, and anticipatory
 * repair workflow automation.
 *
 * FabricationService uses an in-memory data model with MATL reputation
 * scoring via LocalBridge, so tests verify local state management
 * rather than zome call dispatch.
 */

import { describe, it, expect, beforeEach } from 'vitest';

import {
  FabricationService,
  getFabricationService,
  type Design,
  type DesignCategory,
  type SafetyClass,
  type License,
  type FileFormat,
  type DesignFile,
  type ParametricEngine,
  type ParameterType,
  type DesignParameter,
  type ParametricSchema,
  type RepairManifest,
  type FailureMode,
  type DesignEpistemic,
  type Printer,
  type PrinterType,
  type AvailabilityStatus,
  type BuildVolume,
  type PrinterCapabilities,
  type GeoLocation,
  type PrinterRates,
  type PrinterMatch,
  type PrintJob,
  type PrintJobStatus,
  type PrintResult,
  type PrintIssue,
  type PrintSettings,
  type TemperatureSettings,
  type GroundingRequest,
  type QualityAssessment,
  type PrintRecord,
  type MaterialType,
  type MaterialOrigin,
  type EndOfLifeStrategy,
  type AnomalyType,
  type CincinnatiAction,
  type SensorSnapshot,
  type AnomalyEvent,
  type CincinnatiReport,
  type VerificationType,
  type VerificationResult,
  type RepairPrediction,
  type RepairAction,
  type RepairWorkflowStatus,
  type RepairWorkflow,
  type FabricationEventType,
  type FabricationEvent,
  type ListingType,
  type MarketplaceListing,
} from '../../src/integrations/fabrication/index.js';

// ============================================================================
// Test helpers
// ============================================================================

function makeDesignInput(overrides: Partial<{
  title: string;
  description: string;
  category: DesignCategory;
  safetyClass: SafetyClass;
  license: License;
  files: DesignFile[];
  parametricSchema: ParametricSchema;
  repairManifest: RepairManifest;
}> = {}) {
  return {
    title: overrides.title ?? 'Test Widget',
    description: overrides.description ?? 'A test design',
    category: overrides.category ?? 'Parts' as DesignCategory,
    safetyClass: overrides.safetyClass ?? 'Class1Functional' as SafetyClass,
    license: overrides.license,
    files: overrides.files,
    parametricSchema: overrides.parametricSchema,
    repairManifest: overrides.repairManifest,
  };
}

function makeDefaultCapabilities(): PrinterCapabilities {
  return {
    buildVolume: { x: 220, y: 220, z: 250 },
    layerHeights: [0.1, 0.2, 0.3],
    nozzleDiameters: [0.4],
    heatedBed: true,
    enclosure: false,
    maxTempHotend: 260,
    maxTempBed: 100,
    features: ['auto-bed-level'],
  };
}

function makePrinterInput(overrides: Partial<{
  name: string;
  printerType: PrinterType;
  capabilities: PrinterCapabilities;
  materialsAvailable: MaterialType[];
  location: GeoLocation;
  rates: PrinterRates;
}> = {}) {
  return {
    name: overrides.name ?? 'Prusa i3 MK3S+',
    printerType: overrides.printerType ?? ('FDM' as PrinterType),
    capabilities: overrides.capabilities ?? makeDefaultCapabilities(),
    materialsAvailable: overrides.materialsAvailable ?? ['PLA', 'PETG'] as MaterialType[],
    location: overrides.location,
    rates: overrides.rates,
  };
}

function makeQualityAssessment(overrides: Partial<QualityAssessment> = {}): QualityAssessment {
  return {
    dimensionalAccuracy: overrides.dimensionalAccuracy ?? 0.9,
    surfaceQuality: overrides.surfaceQuality ?? 0.85,
    structuralIntegrity: overrides.structuralIntegrity ?? 0.88,
  };
}

// ============================================================================
// Type construction tests
// ============================================================================

describe('Fabrication Types', () => {
  describe('DesignCategory', () => {
    it('should accept all DesignCategory variants', () => {
      const categories: DesignCategory[] = [
        'Tools', 'Parts', 'Housewares', 'Medical', 'Accessibility',
        'Art', 'Education', 'Repair', 'Custom',
      ];
      expect(categories).toHaveLength(9);
      categories.forEach((c) => expect(typeof c).toBe('string'));
    });
  });

  describe('SafetyClass', () => {
    it('should accept all SafetyClass variants', () => {
      const classes: SafetyClass[] = [
        'Class0Decorative', 'Class1Functional', 'Class2LoadBearing',
        'Class3BodyContact', 'Class4Medical', 'Class5Critical',
      ];
      expect(classes).toHaveLength(6);
    });
  });

  describe('License', () => {
    it('should accept string license variants', () => {
      const licenses: License[] = [
        'PublicDomain', 'CreativeCommons', 'OpenHardware', 'Proprietary',
      ];
      expect(licenses).toHaveLength(4);
    });

    it('should accept custom license variant', () => {
      const custom: License = { Custom: 'MIT License' };
      expect(custom).toEqual({ Custom: 'MIT License' });
    });
  });

  describe('FileFormat', () => {
    it('should accept all FileFormat variants', () => {
      const formats: FileFormat[] = ['STL', 'STEP', 'ThreeMF', 'SCAD', 'GCODE', 'OBJ', 'Other'];
      expect(formats).toHaveLength(7);
    });
  });

  describe('DesignFile', () => {
    it('should construct a valid DesignFile', () => {
      const file: DesignFile = {
        filename: 'widget.stl',
        format: 'STL',
        ipfsCid: 'Qm123abc',
        sizeBytes: 1024000,
        checksumSha256: 'abc123def456',
      };
      expect(file.filename).toBe('widget.stl');
      expect(file.format).toBe('STL');
      expect(file.sizeBytes).toBe(1024000);
    });
  });

  describe('ParametricEngine', () => {
    it('should accept all ParametricEngine variants', () => {
      const engines: ParametricEngine[] = ['OpenSCAD', 'CadQuery', 'FreeCAD', 'Fusion360', 'Other'];
      expect(engines).toHaveLength(5);
    });
  });

  describe('MaterialType', () => {
    it('should accept common FDM materials', () => {
      const materials: MaterialType[] = ['PLA', 'PETG', 'ABS', 'ASA', 'TPU', 'Nylon', 'PC', 'PEEK'];
      expect(materials).toHaveLength(8);
    });

    it('should accept custom material variant', () => {
      const custom: MaterialType = { Custom: 'CarbonFiberPLA' };
      expect(custom).toEqual({ Custom: 'CarbonFiberPLA' });
    });
  });

  describe('PrintJobStatus', () => {
    it('should accept all PrintJobStatus variants', () => {
      const statuses: PrintJobStatus[] = [
        'Pending', 'Accepted', 'Queued', 'Printing',
        'PostProcessing', 'Completed', 'Failed', 'Cancelled',
      ];
      expect(statuses).toHaveLength(8);
    });
  });

  describe('AvailabilityStatus', () => {
    it('should accept all AvailabilityStatus variants', () => {
      const statuses: AvailabilityStatus[] = [
        'Available', 'Busy', 'Maintenance', 'Offline', 'ByAppointment',
      ];
      expect(statuses).toHaveLength(5);
    });
  });

  describe('FailureMode', () => {
    it('should accept all FailureMode variants', () => {
      const modes: FailureMode[] = [
        'MechanicalWear', 'UvDegradation', 'ThermalCycling',
        'ChemicalExposure', 'ImpactDamage', 'Fatigue',
      ];
      expect(modes).toHaveLength(6);
    });
  });

  describe('AnomalyType', () => {
    it('should accept all AnomalyType variants', () => {
      const types: AnomalyType[] = [
        'ExtrusionInconsistency', 'TemperatureDeviation', 'VibrationAnomaly',
        'LayerAdhesionFailure', 'NozzleClog', 'BedLevelDrift',
        'PowerFluctuation', 'Unknown',
      ];
      expect(types).toHaveLength(8);
    });
  });

  describe('RepairWorkflowStatus', () => {
    it('should accept all RepairWorkflowStatus variants', () => {
      const statuses: RepairWorkflowStatus[] = [
        'Predicted', 'DesignFound', 'PrinterMatched', 'FundingApproved',
        'Printing', 'ReadyForInstall', 'Installed', 'Cancelled',
      ];
      expect(statuses).toHaveLength(8);
    });
  });

  describe('FabricationEventType', () => {
    it('should accept all FabricationEventType variants', () => {
      const types: FabricationEventType[] = [
        'DesignPublished', 'DesignVerified', 'PrintCompleted',
        'PrinterRegistered', 'MaterialShortage', 'VerificationRequested',
      ];
      expect(types).toHaveLength(6);
    });
  });

  describe('DesignEpistemic', () => {
    it('should construct a valid DesignEpistemic', () => {
      const epistemic: DesignEpistemic = {
        manufacturability: 0.8,
        safety: 0.95,
        usability: 0.7,
      };
      expect(epistemic.manufacturability).toBe(0.8);
      expect(epistemic.safety).toBe(0.95);
      expect(epistemic.usability).toBe(0.7);
    });
  });

  describe('QualityAssessment', () => {
    it('should construct a valid QualityAssessment', () => {
      const qa = makeQualityAssessment();
      expect(qa.dimensionalAccuracy).toBe(0.9);
      expect(qa.surfaceQuality).toBe(0.85);
      expect(qa.structuralIntegrity).toBe(0.88);
    });
  });

  describe('VerificationType', () => {
    it('should accept all VerificationType variants', () => {
      const types: VerificationType[] = [
        'StructuralAnalysis', 'MaterialCompatibility', 'PrintabilityTest',
        'SafetyReview', 'FoodSafeCertification', 'MedicalCertification',
        'CommunityReview',
      ];
      expect(types).toHaveLength(7);
    });
  });

  describe('VerificationResult', () => {
    it('should accept Passed variant', () => {
      const result: VerificationResult = { Passed: { confidence: 0.95, notes: 'Looks good' } };
      expect(result).toHaveProperty('Passed');
    });

    it('should accept Failed variant', () => {
      const result: VerificationResult = { Failed: { reasons: ['Wall too thin'] } };
      expect(result).toHaveProperty('Failed');
    });

    it('should accept ConditionalPass variant', () => {
      const result: VerificationResult = {
        ConditionalPass: { conditions: ['Reinforce top layer'], confidence: 0.75 },
      };
      expect(result).toHaveProperty('ConditionalPass');
    });

    it('should accept NeedsMoreEvidence variant', () => {
      const result: VerificationResult = 'NeedsMoreEvidence';
      expect(result).toBe('NeedsMoreEvidence');
    });
  });

  describe('ListingType', () => {
    it('should accept all ListingType variants', () => {
      const types: ListingType[] = ['DesignSale', 'PrintService', 'PrintedProduct'];
      expect(types).toHaveLength(3);
    });
  });
});

// ============================================================================
// Singleton
// ============================================================================

describe('FabricationService Singleton', () => {
  it('should return same instance on repeated calls to getFabricationService', () => {
    const a = getFabricationService();
    const b = getFabricationService();
    expect(a).toBe(b);
  });

  it('should return an instance of FabricationService', () => {
    const service = getFabricationService();
    expect(service).toBeInstanceOf(FabricationService);
  });
});

// ============================================================================
// FabricationService
// ============================================================================

describe('FabricationService', () => {
  let service: FabricationService;

  beforeEach(() => {
    service = new FabricationService();
  });

  // ==========================================================================
  // Design Operations
  // ==========================================================================

  describe('Design Operations', () => {
    describe('createDesign', () => {
      it('should create a design with correct fields', () => {
        const design = service.createDesign(makeDesignInput());

        expect(design.id).toBeTruthy();
        expect(design.title).toBe('Test Widget');
        expect(design.description).toBe('A test design');
        expect(design.category).toBe('Parts');
        expect(design.safetyClass).toBe('Class1Functional');
        expect(design.license).toBe('PublicDomain');
        expect(design.files).toEqual([]);
        expect(design.circularityScore).toBe(0.5);
        expect(design.embodiedEnergyKwh).toBe(0);
        expect(design.epistemic.manufacturability).toBe(0.5);
        expect(design.epistemic.safety).toBe(0.5);
        expect(design.epistemic.usability).toBe(0.5);
        expect(design.author).toBe('local-agent');
        expect(design.createdAt).toBeGreaterThan(0);
        expect(design.updatedAt).toBeGreaterThan(0);
      });

      it('should accept all DesignCategory variants', () => {
        const categories: DesignCategory[] = [
          'Tools', 'Parts', 'Housewares', 'Medical', 'Accessibility',
          'Art', 'Education', 'Repair', 'Custom',
        ];
        for (const cat of categories) {
          const design = service.createDesign(makeDesignInput({ category: cat }));
          expect(design.category).toBe(cat);
        }
      });

      it('should accept custom license', () => {
        const design = service.createDesign(makeDesignInput({ license: { Custom: 'GPL-3.0' } }));
        expect(design.license).toEqual({ Custom: 'GPL-3.0' });
      });

      it('should generate unique IDs for each design', () => {
        const d1 = service.createDesign(makeDesignInput({ title: 'Design A' }));
        const d2 = service.createDesign(makeDesignInput({ title: 'Design B' }));
        expect(d1.id).not.toBe(d2.id);
      });

      it('should accept optional parametricSchema', () => {
        const schema: ParametricSchema = {
          engine: 'OpenSCAD',
          sourceTemplate: 'template.scad',
          parameters: [{
            name: 'width',
            paramType: 'Length',
            defaultValue: 50,
            minValue: 10,
            maxValue: 200,
            unit: 'mm',
          }],
          constraints: ['width > 10'],
          autoGenerate: true,
        };
        const design = service.createDesign(makeDesignInput({ parametricSchema: schema }));
        expect(design.parametricSchema).toBeDefined();
        expect(design.parametricSchema!.engine).toBe('OpenSCAD');
        expect(design.parametricSchema!.parameters).toHaveLength(1);
      });

      it('should accept optional repairManifest', () => {
        const manifest: RepairManifest = {
          parentProductModel: 'DeWalt DCD771',
          partName: 'Battery Clip',
          failureModes: ['MechanicalWear', 'UvDegradation'],
          repairDifficulty: 'Easy',
        };
        const design = service.createDesign(makeDesignInput({
          category: 'Repair',
          repairManifest: manifest,
        }));
        expect(design.repairManifest).toBeDefined();
        expect(design.repairManifest!.partName).toBe('Battery Clip');
        expect(design.repairManifest!.failureModes).toHaveLength(2);
      });
    });

    describe('getDesign', () => {
      it('should return the design by ID', () => {
        const created = service.createDesign(makeDesignInput());
        const fetched = service.getDesign(created.id);
        expect(fetched).toBeDefined();
        expect(fetched!.title).toBe('Test Widget');
      });

      it('should return undefined for non-existent design', () => {
        const result = service.getDesign('non-existent');
        expect(result).toBeUndefined();
      });
    });

    describe('getDesignsByCategory', () => {
      it('should return designs filtered by category', () => {
        service.createDesign(makeDesignInput({ category: 'Tools' }));
        service.createDesign(makeDesignInput({ category: 'Parts' }));
        service.createDesign(makeDesignInput({ category: 'Tools' }));

        const tools = service.getDesignsByCategory('Tools');
        expect(tools).toHaveLength(2);
        expect(tools.every((d) => d.category === 'Tools')).toBe(true);
      });

      it('should return empty array for category with no designs', () => {
        const result = service.getDesignsByCategory('Medical');
        expect(result).toEqual([]);
      });
    });

    describe('forkDesign', () => {
      it('should fork a design with new title', () => {
        const original = service.createDesign(makeDesignInput({ title: 'Original Widget' }));
        const fork = service.forkDesign(original.id, { title: 'Forked Widget' });

        expect(fork).toBeDefined();
        expect(fork!.title).toBe('Forked Widget');
        expect(fork!.id).not.toBe(original.id);
        expect(fork!.category).toBe(original.category);
        expect(fork!.safetyClass).toBe(original.safetyClass);
      });

      it('should generate default fork title when not specified', () => {
        const original = service.createDesign(makeDesignInput({ title: 'Gadget' }));
        const fork = service.forkDesign(original.id, {});

        expect(fork).toBeDefined();
        expect(fork!.title).toBe('Gadget (Fork)');
      });

      it('should return undefined for non-existent parent', () => {
        const result = service.forkDesign('non-existent', { title: 'Fork' });
        expect(result).toBeUndefined();
      });
    });
  });

  // ==========================================================================
  // Printer Operations
  // ==========================================================================

  describe('Printer Operations', () => {
    describe('registerPrinter', () => {
      it('should register a printer with correct fields', () => {
        const printer = service.registerPrinter(makePrinterInput());

        expect(printer.id).toBeTruthy();
        expect(printer.name).toBe('Prusa i3 MK3S+');
        expect(printer.printerType).toBe('FDM');
        expect(printer.owner).toBe('local-agent');
        expect(printer.availability).toBe('Available');
        expect(printer.materialsAvailable).toContain('PLA');
        expect(printer.materialsAvailable).toContain('PETG');
        expect(printer.capabilities.heatedBed).toBe(true);
        expect(printer.createdAt).toBeGreaterThan(0);
      });

      it('should accept all PrinterType variants', () => {
        const types: PrinterType[] = ['FDM', 'SLA', 'SLS', 'DLP', 'MJF'];
        for (const pt of types) {
          const printer = service.registerPrinter(makePrinterInput({ printerType: pt }));
          expect(printer.printerType).toBe(pt);
        }
      });

      it('should accept custom printer type', () => {
        const printer = service.registerPrinter(makePrinterInput({
          printerType: { Other: 'BinderJet' },
        }));
        expect(printer.printerType).toEqual({ Other: 'BinderJet' });
      });

      it('should accept optional location and rates', () => {
        const printer = service.registerPrinter(makePrinterInput({
          location: { geohash: 'abc123', city: 'Richardson', country: 'US' },
          rates: { baseFee: 5, perHour: 2, perGram: 0.05 },
        }));
        expect(printer.location!.city).toBe('Richardson');
        expect(printer.rates!.baseFee).toBe(5);
      });

      it('should generate unique IDs', () => {
        const p1 = service.registerPrinter(makePrinterInput({ name: 'Printer A' }));
        const p2 = service.registerPrinter(makePrinterInput({ name: 'Printer B' }));
        expect(p1.id).not.toBe(p2.id);
      });
    });

    describe('findPrintersByCapability', () => {
      it('should filter by minimum build volume', () => {
        service.registerPrinter(makePrinterInput({
          name: 'Small',
          capabilities: { ...makeDefaultCapabilities(), buildVolume: { x: 100, y: 100, z: 100 } },
        }));
        service.registerPrinter(makePrinterInput({
          name: 'Large',
          capabilities: { ...makeDefaultCapabilities(), buildVolume: { x: 300, y: 300, z: 300 } },
        }));

        const results = service.findPrintersByCapability({
          minBuildVolume: { x: 200, y: 200, z: 200 },
        });
        expect(results).toHaveLength(1);
        expect(results[0].name).toBe('Large');
      });

      it('should filter by required materials', () => {
        service.registerPrinter(makePrinterInput({
          name: 'PLA only',
          materialsAvailable: ['PLA'],
        }));
        service.registerPrinter(makePrinterInput({
          name: 'Multi-material',
          materialsAvailable: ['PLA', 'PETG', 'ABS'],
        }));

        const results = service.findPrintersByCapability({
          requiredMaterials: ['PLA', 'ABS'],
        });
        expect(results).toHaveLength(1);
        expect(results[0].name).toBe('Multi-material');
      });

      it('should filter by enclosure requirement', () => {
        service.registerPrinter(makePrinterInput({
          name: 'Open',
          capabilities: { ...makeDefaultCapabilities(), enclosure: false },
        }));
        service.registerPrinter(makePrinterInput({
          name: 'Enclosed',
          capabilities: { ...makeDefaultCapabilities(), enclosure: true },
        }));

        const results = service.findPrintersByCapability({ requireEnclosure: true });
        expect(results).toHaveLength(1);
        expect(results[0].name).toBe('Enclosed');
      });

      it('should return all printers when no requirements specified', () => {
        service.registerPrinter(makePrinterInput({ name: 'A' }));
        service.registerPrinter(makePrinterInput({ name: 'B' }));

        const results = service.findPrintersByCapability({});
        expect(results).toHaveLength(2);
      });
    });

    describe('matchDesignToPrinters', () => {
      it('should match available printers to a design', () => {
        const design = service.createDesign(makeDesignInput());
        service.registerPrinter(makePrinterInput({ name: 'Printer A' }));
        service.registerPrinter(makePrinterInput({ name: 'Printer B' }));

        const matches = service.matchDesignToPrinters(design.id);
        expect(matches).toHaveLength(2);
        expect(matches[0].compatibilityScore).toBeGreaterThan(0);
        expect(matches[0].materialMatch).toBe(true);
        expect(matches[0].volumeMatch).toBe(true);
      });

      it('should not match printers that are not available', () => {
        const design = service.createDesign(makeDesignInput());
        const printer = service.registerPrinter(makePrinterInput());
        service.updatePrinterAvailability(printer.id, 'Maintenance');

        const matches = service.matchDesignToPrinters(design.id);
        expect(matches).toHaveLength(0);
      });

      it('should return empty array for non-existent design', () => {
        const matches = service.matchDesignToPrinters('non-existent');
        expect(matches).toEqual([]);
      });

      it('should sort by compatibility score descending', () => {
        const design = service.createDesign(makeDesignInput());
        service.registerPrinter(makePrinterInput({ name: 'A' }));
        service.registerPrinter(makePrinterInput({ name: 'B' }));

        const matches = service.matchDesignToPrinters(design.id);
        for (let i = 1; i < matches.length; i++) {
          expect(matches[i - 1].compatibilityScore).toBeGreaterThanOrEqual(matches[i].compatibilityScore);
        }
      });

      it('should calculate estimated cost when rates are set', () => {
        const design = service.createDesign(makeDesignInput());
        service.registerPrinter(makePrinterInput({
          rates: { baseFee: 10, perHour: 3, perGram: 0.05 },
        }));

        const matches = service.matchDesignToPrinters(design.id);
        expect(matches).toHaveLength(1);
        expect(matches[0].estimatedCost).toBeDefined();
        expect(matches[0].estimatedCost).toBeGreaterThan(0);
      });
    });

    describe('updatePrinterAvailability', () => {
      it('should update printer status', () => {
        const printer = service.registerPrinter(makePrinterInput());
        expect(printer.availability).toBe('Available');

        service.updatePrinterAvailability(printer.id, 'Maintenance');
        // The printer object is mutated in place
        expect(printer.availability).toBe('Maintenance');
      });

      it('should accept all availability statuses', () => {
        const printer = service.registerPrinter(makePrinterInput());
        const statuses: AvailabilityStatus[] = ['Available', 'Busy', 'Maintenance', 'Offline', 'ByAppointment'];
        for (const s of statuses) {
          service.updatePrinterAvailability(printer.id, s);
          expect(printer.availability).toBe(s);
        }
      });
    });
  });

  // ==========================================================================
  // Print Job Operations
  // ==========================================================================

  describe('Print Job Operations', () => {
    let designId: string;
    let printerId: string;

    beforeEach(() => {
      const design = service.createDesign(makeDesignInput());
      const printer = service.registerPrinter(makePrinterInput());
      designId = design.id;
      printerId = printer.id;
    });

    describe('createPrintJob', () => {
      it('should create a print job with correct fields', () => {
        const job = service.createPrintJob({
          designId,
          printerId,
          settings: {
            layerHeight: 0.2,
            infillPercent: 20,
            material: 'PLA',
            supports: false,
            raft: false,
          },
        });

        expect(job).toBeDefined();
        expect(job!.id).toBeTruthy();
        expect(job!.designHash).toBe(designId);
        expect(job!.printerHash).toBe(printerId);
        expect(job!.requester).toBe('local-agent');
        expect(job!.status).toBe('Pending');
        expect(job!.settings.layerHeight).toBe(0.2);
        expect(job!.settings.material).toBe('PLA');
        expect(job!.settings.temperatures.hotend).toBe(200);
        expect(job!.settings.temperatures.bed).toBe(60);
        expect(job!.createdAt).toBeGreaterThan(0);
      });

      it('should accept custom temperatures', () => {
        const job = service.createPrintJob({
          designId,
          printerId,
          settings: {
            layerHeight: 0.2,
            infillPercent: 20,
            material: 'PETG',
            supports: false,
            raft: false,
            temperatures: { hotend: 240, bed: 80 },
          },
        });

        expect(job!.settings.temperatures.hotend).toBe(240);
        expect(job!.settings.temperatures.bed).toBe(80);
      });

      it('should accept grounding request', () => {
        const job = service.createPrintJob({
          designId,
          printerId,
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

        expect(job!.groundingRequest).toBeDefined();
        expect(job!.groundingRequest!.requireRenewable).toBe(true);
        expect(job!.groundingRequest!.targetPogScore).toBe(0.8);
      });

      it('should return undefined for non-existent design', () => {
        const job = service.createPrintJob({
          designId: 'non-existent',
          printerId,
          settings: {
            layerHeight: 0.2,
            infillPercent: 20,
            material: 'PLA',
            supports: false,
            raft: false,
          },
        });
        expect(job).toBeUndefined();
      });

      it('should return undefined for non-existent printer', () => {
        const job = service.createPrintJob({
          designId,
          printerId: 'non-existent',
          settings: {
            layerHeight: 0.2,
            infillPercent: 20,
            material: 'PLA',
            supports: false,
            raft: false,
          },
        });
        expect(job).toBeUndefined();
      });
    });

    describe('acceptPrintJob', () => {
      it('should accept a pending job', () => {
        const job = service.createPrintJob({
          designId,
          printerId,
          settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
        })!;

        const result = service.acceptPrintJob(job.id);
        expect(result).toBe(true);
        expect(job.status).toBe('Accepted');
      });

      it('should reject accepting a non-pending job', () => {
        const job = service.createPrintJob({
          designId,
          printerId,
          settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
        })!;

        service.acceptPrintJob(job.id);
        const result = service.acceptPrintJob(job.id);
        expect(result).toBe(false);
      });

      it('should return false for non-existent job', () => {
        const result = service.acceptPrintJob('non-existent');
        expect(result).toBe(false);
      });
    });

    describe('startPrint', () => {
      it('should start an accepted job', () => {
        const job = service.createPrintJob({
          designId,
          printerId,
          settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
        })!;

        service.acceptPrintJob(job.id);
        const result = service.startPrint(job.id);
        expect(result).toBe(true);
        expect(job.status).toBe('Printing');
        expect(job.startedAt).toBeGreaterThan(0);
      });

      it('should set printer to Busy when print starts', () => {
        const printer = service.registerPrinter(makePrinterInput({ name: 'Another' }));
        const design = service.createDesign(makeDesignInput());
        const job = service.createPrintJob({
          designId: design.id,
          printerId: printer.id,
          settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
        })!;

        service.acceptPrintJob(job.id);
        service.startPrint(job.id);
        expect(printer.availability).toBe('Busy');
      });

      it('should not start a pending job', () => {
        const job = service.createPrintJob({
          designId,
          printerId,
          settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
        })!;

        const result = service.startPrint(job.id);
        expect(result).toBe(false);
      });
    });

    describe('completePrint', () => {
      it('should complete a printing job and return PrintRecord', () => {
        const job = service.createPrintJob({
          designId,
          printerId,
          settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
        })!;

        service.acceptPrintJob(job.id);
        service.startPrint(job.id);

        const record = service.completePrint(job.id, 'Success', makeQualityAssessment(), 25);

        expect(record).toBeDefined();
        expect(record!.jobHash).toBe(job.id);
        expect(record!.result).toBe('Success');
        expect(record!.pogScore).toBeGreaterThan(0);
        expect(record!.pogScore).toBeLessThanOrEqual(1);
        expect(record!.myceliumEarned).toBeGreaterThan(0);
        expect(record!.materialCircularity).toBe(0.5);
        expect(record!.qualityAssessment).toBeDefined();
        expect(record!.recordedAt).toBeGreaterThan(0);
        expect(job.status).toBe('Completed');
        expect(job.materialUsedGrams).toBe(25);
      });

      it('should set printer back to Available after completion', () => {
        const printer = service.registerPrinter(makePrinterInput({ name: 'Test' }));
        const design = service.createDesign(makeDesignInput());
        const job = service.createPrintJob({
          designId: design.id,
          printerId: printer.id,
          settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
        })!;

        service.acceptPrintJob(job.id);
        service.startPrint(job.id);
        expect(printer.availability).toBe('Busy');

        service.completePrint(job.id, 'Success', makeQualityAssessment(), 15);
        expect(printer.availability).toBe('Available');
      });

      it('should return undefined when job is not in Printing status', () => {
        const job = service.createPrintJob({
          designId,
          printerId,
          settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
        })!;

        const result = service.completePrint(job.id, 'Success', makeQualityAssessment(), 10);
        expect(result).toBeUndefined();
      });

      it('should calculate quality score from assessment', () => {
        const job = service.createPrintJob({
          designId,
          printerId,
          settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
        })!;

        service.acceptPrintJob(job.id);
        service.startPrint(job.id);

        const assessment: QualityAssessment = {
          dimensionalAccuracy: 0.9,
          surfaceQuality: 0.8,
          structuralIntegrity: 1.0,
        };
        const record = service.completePrint(job.id, 'Success', assessment, 20);

        expect(record!.qualityScore).toBeCloseTo((0.9 + 0.8 + 1.0) / 3);
      });
    });
  });

  // ==========================================================================
  // Reputation Operations
  // ==========================================================================

  describe('Reputation Operations', () => {
    it('should return default trust score for unknown printer', () => {
      const score = service.getPrinterTrustScore('unknown-printer');
      expect(score).toBe(0.5);
    });

    it('should consider unknown printer as not trustworthy', () => {
      const result = service.isPrinterTrustworthy('unknown-printer');
      expect(result).toBe(false);
    });

    it('should update reputation after successful print', () => {
      const design = service.createDesign(makeDesignInput());
      const printer = service.registerPrinter(makePrinterInput());
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
      })!;

      service.acceptPrintJob(job.id);
      service.startPrint(job.id);
      service.completePrint(job.id, 'Success', makeQualityAssessment(), 20);

      const score = service.getPrinterTrustScore(printer.id);
      expect(score).toBeGreaterThanOrEqual(0);
    });

    it('should update reputation after failed print', () => {
      const design = service.createDesign(makeDesignInput());
      const printer = service.registerPrinter(makePrinterInput());
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: { layerHeight: 0.2, infillPercent: 20, material: 'PLA', supports: false, raft: false },
      })!;

      service.acceptPrintJob(job.id);
      service.startPrint(job.id);
      service.completePrint(job.id, { Failed: 'Nozzle clog' }, makeQualityAssessment(), 5);

      const score = service.getPrinterTrustScore(printer.id);
      expect(score).toBeDefined();
    });
  });

  // ==========================================================================
  // Anticipatory Repair
  // ==========================================================================

  describe('Anticipatory Repair', () => {
    describe('processRepairPrediction', () => {
      it('should create a workflow with Predicted status when no matching design exists', () => {
        const prediction: RepairPrediction = {
          propertyAssetHash: 'asset-001',
          assetModel: 'Samsung RF28R7551SR',
          predictedFailureComponent: 'Ice Maker Clip',
          failureProbability: 0.85,
          estimatedFailureDate: Date.now() + 30 * 24 * 3600_000,
          confidenceIntervalDays: 15,
          sensorDataSummary: 'Vibration anomaly detected in ice maker motor',
          recommendedAction: 'PrintReplacement',
          createdAt: Date.now(),
        };

        const workflow = service.processRepairPrediction(prediction);
        expect(workflow.status).toBe('Predicted');
        expect(workflow.predictionHash).toBeTruthy();
        expect(workflow.createdAt).toBeGreaterThan(0);
        expect(workflow.designHash).toBeUndefined();
      });

      it('should advance to DesignFound when matching repair design exists', () => {
        service.createDesign(makeDesignInput({
          category: 'Repair',
          repairManifest: {
            parentProductModel: 'DeWalt DCD771',
            partName: 'Battery Clip',
            failureModes: ['MechanicalWear'],
            repairDifficulty: 'Easy',
          },
        }));

        const prediction: RepairPrediction = {
          propertyAssetHash: 'asset-002',
          assetModel: 'DeWalt DCD771',
          predictedFailureComponent: 'Battery Clip',
          failureProbability: 0.9,
          estimatedFailureDate: Date.now() + 14 * 24 * 3600_000,
          confidenceIntervalDays: 7,
          sensorDataSummary: 'Wear pattern detected',
          recommendedAction: 'PrintReplacement',
          createdAt: Date.now(),
        };

        const workflow = service.processRepairPrediction(prediction);
        expect(workflow.status).toBe('DesignFound');
        expect(workflow.designHash).toBeTruthy();
      });

      it('should advance to PrinterMatched when matching design and printer exist', () => {
        service.createDesign(makeDesignInput({
          category: 'Repair',
          repairManifest: {
            parentProductModel: 'Makita XPH12',
            partName: 'Trigger Guard',
            failureModes: ['ImpactDamage'],
            repairDifficulty: 'Moderate',
          },
        }));
        service.registerPrinter(makePrinterInput());

        const prediction: RepairPrediction = {
          propertyAssetHash: 'asset-003',
          assetModel: 'Makita XPH12',
          predictedFailureComponent: 'Trigger Guard',
          failureProbability: 0.75,
          estimatedFailureDate: Date.now() + 60 * 24 * 3600_000,
          confidenceIntervalDays: 20,
          sensorDataSummary: 'Impact damage accumulation',
          recommendedAction: 'PrintReplacement',
          createdAt: Date.now(),
        };

        const workflow = service.processRepairPrediction(prediction);
        expect(workflow.status).toBe('PrinterMatched');
        expect(workflow.designHash).toBeTruthy();
        expect(workflow.printerHash).toBeTruthy();
      });
    });
  });

  // ==========================================================================
  // Event Emission
  // ==========================================================================

  describe('Event Emission', () => {
    it('should emit events without throwing', () => {
      expect(() => {
        service.emitEvent('DesignPublished', 'design-123', { category: 'Tools' });
      }).not.toThrow();
    });

    it('should accept all event types', () => {
      const types: FabricationEventType[] = [
        'DesignPublished', 'DesignVerified', 'PrintCompleted',
        'PrinterRegistered', 'MaterialShortage', 'VerificationRequested',
      ];
      for (const t of types) {
        expect(() => {
          service.emitEvent(t, undefined, {});
        }).not.toThrow();
      }
    });
  });

  // ==========================================================================
  // FL Coordinator
  // ==========================================================================

  describe('FL Coordinator', () => {
    it('should return the FL coordinator instance', () => {
      const coordinator = service.getFLCoordinator();
      expect(coordinator).toBeDefined();
    });
  });

  // ==========================================================================
  // Full Lifecycle
  // ==========================================================================

  describe('Full Print Lifecycle', () => {
    it('should support the complete design -> print -> complete workflow', () => {
      // Step 1: Create design
      const design = service.createDesign(makeDesignInput({
        title: 'Cable Organizer',
        category: 'Housewares',
        safetyClass: 'Class0Decorative',
      }));
      expect(design.category).toBe('Housewares');

      // Step 2: Register printer
      const printer = service.registerPrinter(makePrinterInput({
        name: 'Ender 3 V2',
        rates: { baseFee: 3, perHour: 1, perGram: 0.03 },
      }));
      expect(printer.availability).toBe('Available');

      // Step 3: Match design to printers
      const matches = service.matchDesignToPrinters(design.id);
      expect(matches.length).toBeGreaterThan(0);

      // Step 4: Create print job
      const job = service.createPrintJob({
        designId: design.id,
        printerId: printer.id,
        settings: {
          layerHeight: 0.2,
          infillPercent: 15,
          material: 'PLA',
          supports: false,
          raft: false,
        },
      });
      expect(job).toBeDefined();
      expect(job!.status).toBe('Pending');

      // Step 5: Accept job
      const accepted = service.acceptPrintJob(job!.id);
      expect(accepted).toBe(true);
      expect(job!.status).toBe('Accepted');

      // Step 6: Start printing
      const started = service.startPrint(job!.id);
      expect(started).toBe(true);
      expect(job!.status).toBe('Printing');
      expect(printer.availability).toBe('Busy');

      // Step 7: Complete print
      const record = service.completePrint(
        job!.id,
        'Success',
        { dimensionalAccuracy: 0.92, surfaceQuality: 0.88, structuralIntegrity: 0.95 },
        18
      );
      expect(record).toBeDefined();
      expect(record!.result).toBe('Success');
      expect(record!.pogScore).toBeGreaterThan(0);
      expect(record!.myceliumEarned).toBeGreaterThan(0);
      expect(job!.status).toBe('Completed');
      expect(printer.availability).toBe('Available');

      // Step 8: Check reputation
      const score = service.getPrinterTrustScore(printer.id);
      expect(score).toBeDefined();
    });
  });
});
