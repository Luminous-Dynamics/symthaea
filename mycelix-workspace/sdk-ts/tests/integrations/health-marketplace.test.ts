/**
 * Health-Marketplace Integration Tests
 *
 * Tests for the health-marketplace integration module including:
 * - Prescription marketplace functionality
 * - Health product verification
 * - DME (Durable Medical Equipment) rentals
 * - Utility functions
 * - Cross-domain workflows
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  HealthMarketplaceBridge,
  PrescriptionMarketplaceClient,
  HealthProductClient,
  DMERentalClient,
  daysUntilRefillable,
  isPrescriptionExpiringSoon,
  calculatePotentialSavings,
  requiresPrescriptionVerification,
  formatDirections,
  type MarketplacePrescription,
  type PrescriptionPriceQuote,
  type Pharmacy,
  type HealthProduct,
  type HealthProductCategory,
  type PrescriptionOrder,
  type CoverageCheckResult,
  type FulfillmentStatus,
} from '../../src/integrations/health-marketplace/index.js';
import type { MycelixClient } from '../../src/client/index.js';

describe('Health-Marketplace Integration', () => {
  describe('daysUntilRefillable', () => {
    it('should return 0 if already refillable', () => {
      const prescription: MarketplacePrescription = {
        prescriptionId: 'rx-1',
        patientHash: new Uint8Array(39) as any,
        medicationName: 'Lisinopril',
        dosage: '10mg',
        quantity: 30,
        daysSupply: 30,
        refillsRemaining: 5,
        lastFillDate: Date.now() - 86400000 * 25, // 25 days ago
        expirationDate: Date.now() + 86400000 * 180,
        providerName: 'Dr. Smith',
        status: 'Active',
        createdAt: Date.now() - 86400000 * 60,
      };

      // Can refill at 75% of supply (day 22.5), we're at day 25
      const days = daysUntilRefillable(prescription);
      expect(days).toBe(0);
    });

    it('should return positive days if not yet refillable', () => {
      const prescription: MarketplacePrescription = {
        prescriptionId: 'rx-1',
        patientHash: new Uint8Array(39) as any,
        medicationName: 'Lisinopril',
        dosage: '10mg',
        quantity: 30,
        daysSupply: 30,
        refillsRemaining: 5,
        lastFillDate: Date.now() - 86400000 * 10, // 10 days ago
        expirationDate: Date.now() + 86400000 * 180,
        providerName: 'Dr. Smith',
        status: 'Active',
        createdAt: Date.now() - 86400000 * 60,
      };

      // Can refill at 75% of supply (day 22.5), we're at day 10
      const days = daysUntilRefillable(prescription);
      expect(days).toBeGreaterThan(0);
      expect(days).toBeLessThanOrEqual(13); // ~12.5 days
    });

    it('should return 0 if no previous fill', () => {
      const prescription: MarketplacePrescription = {
        prescriptionId: 'rx-1',
        patientHash: new Uint8Array(39) as any,
        medicationName: 'Lisinopril',
        dosage: '10mg',
        quantity: 30,
        daysSupply: 30,
        refillsRemaining: 5,
        expirationDate: Date.now() + 86400000 * 180,
        providerName: 'Dr. Smith',
        status: 'Active',
        createdAt: Date.now(),
      };

      const days = daysUntilRefillable(prescription);
      expect(days).toBe(0);
    });
  });

  describe('isPrescriptionExpiringSoon', () => {
    it('should return true for prescription expiring within 30 days', () => {
      const prescription: MarketplacePrescription = {
        prescriptionId: 'rx-1',
        patientHash: new Uint8Array(39) as any,
        medicationName: 'Metformin',
        dosage: '500mg',
        quantity: 90,
        daysSupply: 90,
        refillsRemaining: 0,
        expirationDate: Date.now() + 86400000 * 15, // 15 days from now
        providerName: 'Dr. Johnson',
        status: 'Active',
        createdAt: Date.now() - 86400000 * 180,
      };

      expect(isPrescriptionExpiringSoon(prescription)).toBe(true);
    });

    it('should return false for prescription expiring beyond threshold', () => {
      const prescription: MarketplacePrescription = {
        prescriptionId: 'rx-1',
        patientHash: new Uint8Array(39) as any,
        medicationName: 'Metformin',
        dosage: '500mg',
        quantity: 90,
        daysSupply: 90,
        refillsRemaining: 3,
        expirationDate: Date.now() + 86400000 * 180, // 180 days from now
        providerName: 'Dr. Johnson',
        status: 'Active',
        createdAt: Date.now() - 86400000 * 30,
      };

      expect(isPrescriptionExpiringSoon(prescription)).toBe(false);
    });

    it('should use custom threshold days', () => {
      const prescription: MarketplacePrescription = {
        prescriptionId: 'rx-1',
        patientHash: new Uint8Array(39) as any,
        medicationName: 'Metformin',
        dosage: '500mg',
        quantity: 90,
        daysSupply: 90,
        refillsRemaining: 1,
        expirationDate: Date.now() + 86400000 * 50, // 50 days from now
        providerName: 'Dr. Johnson',
        status: 'Active',
        createdAt: Date.now() - 86400000 * 30,
      };

      expect(isPrescriptionExpiringSoon(prescription, 30)).toBe(false);
      expect(isPrescriptionExpiringSoon(prescription, 60)).toBe(true);
    });
  });

  describe('calculatePotentialSavings', () => {
    it('should calculate savings between quotes', () => {
      const quotes: PrescriptionPriceQuote[] = [
        { pharmacyId: 'pharm-1', pharmacyName: 'CVS', price: 45.00, estimatedWaitMinutes: 30, distance: 2.5 },
        { pharmacyId: 'pharm-2', pharmacyName: 'Walgreens', price: 52.00, estimatedWaitMinutes: 15, distance: 1.0 },
        { pharmacyId: 'pharm-3', pharmacyName: 'Costco', price: 28.00, estimatedWaitMinutes: 45, distance: 8.0 },
      ];

      const savings = calculatePotentialSavings(quotes);
      expect(savings.lowestPrice).toBe(28.00);
      expect(savings.highestPrice).toBe(52.00);
      expect(savings.maxSavings).toBe(24.00);
      expect(savings.lowestPricePharmacy).toBe('Costco');
    });

    it('should return zero savings for single quote', () => {
      const quotes: PrescriptionPriceQuote[] = [
        { pharmacyId: 'pharm-1', pharmacyName: 'CVS', price: 45.00, estimatedWaitMinutes: 30, distance: 2.5 },
      ];

      const savings = calculatePotentialSavings(quotes);
      expect(savings.maxSavings).toBe(0);
    });

    it('should handle empty quotes array', () => {
      const savings = calculatePotentialSavings([]);
      expect(savings.lowestPrice).toBe(0);
      expect(savings.highestPrice).toBe(0);
      expect(savings.maxSavings).toBe(0);
    });
  });

  describe('requiresPrescriptionVerification', () => {
    it('should require verification for controlled substances', () => {
      const categories: HealthProductCategory[] = [
        'ControlledSubstance',
        'Opioid',
        'Benzodiazepine',
        'Stimulant',
      ];

      categories.forEach(category => {
        expect(requiresPrescriptionVerification(category)).toBe(true);
      });
    });

    it('should require verification for prescription medications', () => {
      expect(requiresPrescriptionVerification('PrescriptionMedication')).toBe(true);
    });

    it('should not require verification for OTC products', () => {
      const categories: HealthProductCategory[] = [
        'OTC',
        'Supplement',
        'MedicalSupply',
        'PersonalCare',
      ];

      categories.forEach(category => {
        expect(requiresPrescriptionVerification(category)).toBe(false);
      });
    });
  });

  describe('formatDirections', () => {
    it('should format standard directions', () => {
      const directions = formatDirections({
        frequency: 'twice daily',
        route: 'oral',
        timing: 'with food',
        duration: '30 days',
      });

      expect(directions).toContain('twice daily');
      expect(directions).toContain('oral');
      expect(directions).toContain('with food');
    });

    it('should handle missing optional fields', () => {
      const directions = formatDirections({
        frequency: 'once daily',
        route: 'oral',
      });

      expect(directions).toContain('once daily');
      expect(directions).toContain('oral');
    });

    it('should format PRN directions', () => {
      const directions = formatDirections({
        frequency: 'as needed',
        route: 'oral',
        maxDailyDose: '4 tablets',
        prnReason: 'pain',
      });

      expect(directions).toContain('as needed');
      expect(directions).toContain('pain');
      expect(directions).toContain('4 tablets');
    });
  });

  describe('PrescriptionMarketplaceClient', () => {
    let mockClient: MycelixClient;
    let prescriptionClient: PrescriptionMarketplaceClient;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      prescriptionClient = new PrescriptionMarketplaceClient(mockClient);
    });

    it('should fetch patient prescriptions', async () => {
      const patientHash = new Uint8Array(39);
      vi.mocked(mockClient.callZome).mockResolvedValue([
        { prescriptionId: 'rx-1', medicationName: 'Lisinopril' },
        { prescriptionId: 'rx-2', medicationName: 'Metformin' },
      ]);

      const prescriptions = await prescriptionClient.getPatientPrescriptions(patientHash);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'health',
        zome_name: 'prescriptions',
        fn_name: 'get_patient_prescriptions',
        payload: { patientHash },
      });
      expect(prescriptions).toHaveLength(2);
    });

    it('should get price quotes from multiple pharmacies', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue([
        { pharmacyId: 'p1', price: 45.00 },
        { pharmacyId: 'p2', price: 38.00 },
      ]);

      const quotes = await prescriptionClient.getPriceQuotes('rx-1', {
        zipCode: '90210',
        radius: 10,
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'marketplace',
        zome_name: 'pharmacy_pricing',
        fn_name: 'get_price_quotes',
        payload: {
          prescriptionId: 'rx-1',
          options: { zipCode: '90210', radius: 10 },
        },
      });
      expect(quotes).toHaveLength(2);
    });

    it('should submit prescription order', async () => {
      const order: Partial<PrescriptionOrder> = {
        prescriptionId: 'rx-1',
        pharmacyId: 'pharm-1',
        patientHash: new Uint8Array(39) as any,
        deliveryMethod: 'Pickup',
      };

      vi.mocked(mockClient.callZome).mockResolvedValue({
        orderId: 'order-1',
        status: 'Pending',
      });

      const result = await prescriptionClient.submitOrder(order as PrescriptionOrder);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'marketplace',
        zome_name: 'prescription_orders',
        fn_name: 'submit_order',
        payload: order,
      });
      expect(result.orderId).toBe('order-1');
    });

    it('should check insurance coverage', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue({
        covered: true,
        copay: 15.00,
        deductibleApplied: 0,
        priorAuthRequired: false,
        tier: 'Preferred',
      });

      const coverage = await prescriptionClient.checkCoverage({
        prescriptionId: 'rx-1',
        insuranceId: 'ins-1',
        memberId: 'member-123',
      });

      expect(coverage.covered).toBe(true);
      expect(coverage.copay).toBe(15.00);
    });

    it('should transfer prescription between pharmacies', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue({
        transferId: 'transfer-1',
        status: 'Initiated',
      });

      const result = await prescriptionClient.transferPrescription({
        prescriptionId: 'rx-1',
        fromPharmacyId: 'pharm-1',
        toPharmacyId: 'pharm-2',
        patientConsent: true,
      });

      expect(result.transferId).toBe('transfer-1');
    });
  });

  describe('HealthProductClient', () => {
    let mockClient: MycelixClient;
    let productClient: HealthProductClient;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      productClient = new HealthProductClient(mockClient);
    });

    it('should search products by category', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue([
        { productId: 'prod-1', name: 'Blood Pressure Monitor' },
        { productId: 'prod-2', name: 'Glucose Meter' },
      ]);

      const products = await productClient.searchProducts({
        category: 'MedicalDevice',
        query: 'monitor',
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'marketplace',
        zome_name: 'health_products',
        fn_name: 'search_products',
        payload: { category: 'MedicalDevice', query: 'monitor' },
      });
      expect(products).toHaveLength(2);
    });

    it('should verify product authenticity', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue({
        verified: true,
        manufacturer: 'Omron',
        fdaClearance: '510(k)',
        lotNumber: 'LOT-2024-001',
      });

      const verification = await productClient.verifyProduct('prod-1');

      expect(verification.verified).toBe(true);
      expect(verification.fdaClearance).toBe('510(k)');
    });

    it('should get product reviews with trust scores', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue([
        { reviewId: 'r1', rating: 5, trustScore: 0.92 },
        { reviewId: 'r2', rating: 4, trustScore: 0.85 },
      ]);

      const reviews = await productClient.getProductReviews('prod-1');

      expect(reviews).toHaveLength(2);
      expect(reviews[0].trustScore).toBeGreaterThan(0.8);
    });
  });

  describe('DMERentalClient', () => {
    let mockClient: MycelixClient;
    let dmeClient: DMERentalClient;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      dmeClient = new DMERentalClient(mockClient);
    });

    it('should search available DME for rental', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue([
        { equipmentId: 'dme-1', type: 'Wheelchair', dailyRate: 15.00 },
        { equipmentId: 'dme-2', type: 'Hospital Bed', dailyRate: 35.00 },
      ]);

      const equipment = await dmeClient.searchAvailable({
        equipmentType: 'Mobility',
        zipCode: '90210',
        startDate: Date.now(),
        endDate: Date.now() + 86400000 * 30,
      });

      expect(equipment).toHaveLength(2);
    });

    it('should create rental reservation', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue({
        reservationId: 'res-1',
        status: 'Confirmed',
        totalCost: 450.00,
      });

      const reservation = await dmeClient.createReservation({
        equipmentId: 'dme-1',
        patientHash: new Uint8Array(39) as any,
        startDate: Date.now(),
        endDate: Date.now() + 86400000 * 30,
        deliveryAddress: '123 Main St',
        prescriptionId: 'rx-dme-1',
      });

      expect(reservation.reservationId).toBe('res-1');
      expect(reservation.totalCost).toBe(450.00);
    });

    it('should check Medicare/Medicaid coverage for DME', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue({
        covered: true,
        coverageType: 'Medicare Part B',
        patientResponsibility: 0.20, // 20% copay
        priorAuthRequired: true,
      });

      const coverage = await dmeClient.checkDMECoverage({
        equipmentId: 'dme-1',
        memberId: 'medicare-123',
        diagnosis: 'M62.81', // Muscle weakness
      });

      expect(coverage.covered).toBe(true);
      expect(coverage.patientResponsibility).toBe(0.20);
    });
  });

  describe('HealthMarketplaceBridge', () => {
    let mockClient: MycelixClient;
    let bridge: HealthMarketplaceBridge;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      bridge = new HealthMarketplaceBridge(mockClient);
    });

    it('should provide access to all sub-clients', () => {
      expect(bridge.prescriptions).toBeInstanceOf(PrescriptionMarketplaceClient);
      expect(bridge.products).toBeInstanceOf(HealthProductClient);
      expect(bridge.dme).toBeInstanceOf(DMERentalClient);
    });

    it('should aggregate marketplace metrics', async () => {
      vi.mocked(mockClient.callZome)
        .mockResolvedValueOnce([{ prescriptionId: 'rx-1' }, { prescriptionId: 'rx-2' }]) // active prescriptions
        .mockResolvedValueOnce([{ orderId: 'o1' }]) // pending orders
        .mockResolvedValueOnce([{ reservationId: 'r1' }]); // active rentals

      const metrics = await bridge.getMarketplaceMetrics(new Uint8Array(39) as any);

      expect(metrics.activePrescriptions).toBe(2);
      expect(metrics.pendingOrders).toBe(1);
      expect(metrics.activeRentals).toBe(1);
    });
  });

  describe('Cross-Domain Workflows', () => {
    let mockClient: MycelixClient;
    let bridge: HealthMarketplaceBridge;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      bridge = new HealthMarketplaceBridge(mockClient);
    });

    it('should complete prescription fulfillment workflow', async () => {
      // Step 1: Get prescriptions
      vi.mocked(mockClient.callZome)
        .mockResolvedValueOnce([
          {
            prescriptionId: 'rx-1',
            medicationName: 'Lisinopril',
            refillsRemaining: 3,
            status: 'Active',
          },
        ])
        // Step 2: Get price quotes
        .mockResolvedValueOnce([
          { pharmacyId: 'p1', pharmacyName: 'CVS', price: 45.00 },
          { pharmacyId: 'p2', pharmacyName: 'Costco', price: 28.00 },
        ])
        // Step 3: Check coverage
        .mockResolvedValueOnce({
          covered: true,
          copay: 10.00,
        })
        // Step 4: Submit order
        .mockResolvedValueOnce({
          orderId: 'order-1',
          status: 'Confirmed',
        });

      const patientHash = new Uint8Array(39) as any;

      // Get prescriptions
      const prescriptions = await bridge.prescriptions.getPatientPrescriptions(patientHash);
      expect(prescriptions).toHaveLength(1);

      // Get quotes
      const quotes = await bridge.prescriptions.getPriceQuotes('rx-1', { zipCode: '90210' });
      expect(quotes).toHaveLength(2);

      // Find best price
      const savings = calculatePotentialSavings(quotes);
      expect(savings.lowestPricePharmacy).toBe('Costco');

      // Check coverage
      const coverage = await bridge.prescriptions.checkCoverage({
        prescriptionId: 'rx-1',
        insuranceId: 'ins-1',
        memberId: 'member-1',
      });
      expect(coverage.covered).toBe(true);

      // Submit order to lowest price pharmacy
      const order = await bridge.prescriptions.submitOrder({
        orderId: '',
        prescriptionId: 'rx-1',
        pharmacyId: 'p2', // Costco
        patientHash,
        deliveryMethod: 'Pickup',
        status: 'Pending',
        createdAt: Date.now(),
      });
      expect(order.status).toBe('Confirmed');
    });

    it('should coordinate DME with prescription verification', async () => {
      // DME often requires a prescription
      vi.mocked(mockClient.callZome)
        .mockResolvedValueOnce({
          prescriptionId: 'rx-dme-1',
          medicationName: 'Wheelchair (Standard)',
          providerName: 'Dr. Ortho',
          status: 'Active',
        })
        .mockResolvedValueOnce([
          { equipmentId: 'dme-1', type: 'Wheelchair', dailyRate: 12.00 },
        ])
        .mockResolvedValueOnce({
          reservationId: 'res-1',
          status: 'Confirmed',
        });

      // Verify prescription exists
      const prescription = await bridge.prescriptions.getPrescription('rx-dme-1');
      expect(prescription.status).toBe('Active');

      // Search for equipment
      const equipment = await bridge.dme.searchAvailable({
        equipmentType: 'Wheelchair',
        zipCode: '90210',
        startDate: Date.now(),
        endDate: Date.now() + 86400000 * 90,
      });
      expect(equipment).toHaveLength(1);

      // Create reservation with prescription reference
      const reservation = await bridge.dme.createReservation({
        equipmentId: 'dme-1',
        patientHash: new Uint8Array(39) as any,
        startDate: Date.now(),
        endDate: Date.now() + 86400000 * 90,
        prescriptionId: 'rx-dme-1',
      });
      expect(reservation.status).toBe('Confirmed');
    });
  });

  describe('Type Validation', () => {
    it('should validate HealthProductCategory values', () => {
      const validCategories: HealthProductCategory[] = [
        'PrescriptionMedication',
        'OTC',
        'ControlledSubstance',
        'Opioid',
        'Benzodiazepine',
        'Stimulant',
        'MedicalDevice',
        'DME',
        'Supplement',
        'MedicalSupply',
        'PersonalCare',
      ];

      validCategories.forEach(category => {
        const value: HealthProductCategory = category;
        expect(value).toBe(category);
      });
    });

    it('should validate FulfillmentStatus values', () => {
      const validStatuses: FulfillmentStatus[] = [
        'Pending',
        'Processing',
        'ReadyForPickup',
        'OutForDelivery',
        'Delivered',
        'Cancelled',
        'OnHold',
      ];

      validStatuses.forEach(status => {
        const value: FulfillmentStatus = status;
        expect(value).toBe(status);
      });
    });

    it('should validate CoverageCheckResult structure', () => {
      const coverage: CoverageCheckResult = {
        covered: true,
        copay: 25.00,
        deductibleApplied: 100.00,
        priorAuthRequired: false,
        tier: 'Preferred',
        quantity: 30,
        daysSupply: 30,
        message: 'Coverage approved',
      };

      expect(coverage.covered).toBe(true);
      expect(coverage.copay).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Edge Cases', () => {
    it('should handle prescription with no refills remaining', () => {
      const prescription: MarketplacePrescription = {
        prescriptionId: 'rx-1',
        patientHash: new Uint8Array(39) as any,
        medicationName: 'Amoxicillin',
        dosage: '500mg',
        quantity: 21,
        daysSupply: 7,
        refillsRemaining: 0,
        expirationDate: Date.now() + 86400000 * 30,
        providerName: 'Dr. Smith',
        status: 'Active',
        createdAt: Date.now(),
      };

      expect(prescription.refillsRemaining).toBe(0);
      // Should still be able to calculate days until refillable
      const days = daysUntilRefillable(prescription);
      expect(typeof days).toBe('number');
    });

    it('should handle expired prescription', () => {
      const prescription: MarketplacePrescription = {
        prescriptionId: 'rx-1',
        patientHash: new Uint8Array(39) as any,
        medicationName: 'Lisinopril',
        dosage: '10mg',
        quantity: 30,
        daysSupply: 30,
        refillsRemaining: 5,
        expirationDate: Date.now() - 86400000, // Yesterday
        providerName: 'Dr. Smith',
        status: 'Expired',
        createdAt: Date.now() - 86400000 * 365,
      };

      expect(isPrescriptionExpiringSoon(prescription)).toBe(true); // Already expired
    });

    it('should handle very expensive medications', () => {
      const quotes: PrescriptionPriceQuote[] = [
        { pharmacyId: 'p1', pharmacyName: 'Specialty Pharmacy', price: 15000.00, estimatedWaitMinutes: 120, distance: 50 },
        { pharmacyId: 'p2', pharmacyName: 'Mail Order', price: 12000.00, estimatedWaitMinutes: 1440, distance: 0 },
      ];

      const savings = calculatePotentialSavings(quotes);
      expect(savings.maxSavings).toBe(3000.00);
    });

    it('should handle zero price quotes', () => {
      const quotes: PrescriptionPriceQuote[] = [
        { pharmacyId: 'p1', pharmacyName: 'Charity Pharmacy', price: 0, estimatedWaitMinutes: 60, distance: 5 },
        { pharmacyId: 'p2', pharmacyName: 'Regular Pharmacy', price: 50.00, estimatedWaitMinutes: 30, distance: 2 },
      ];

      const savings = calculatePotentialSavings(quotes);
      expect(savings.lowestPrice).toBe(0);
      expect(savings.maxSavings).toBe(50.00);
    });
  });

  describe('Error Handling', () => {
    let mockClient: MycelixClient;
    let bridge: HealthMarketplaceBridge;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      bridge = new HealthMarketplaceBridge(mockClient);
    });

    it('should handle prescription not found', async () => {
      vi.mocked(mockClient.callZome).mockRejectedValue(
        new Error('Prescription not found')
      );

      await expect(
        bridge.prescriptions.getPrescription('invalid-rx')
      ).rejects.toThrow('Prescription not found');
    });

    it('should handle pharmacy API errors', async () => {
      vi.mocked(mockClient.callZome).mockRejectedValue(
        new Error('Pharmacy service unavailable')
      );

      await expect(
        bridge.prescriptions.getPriceQuotes('rx-1', { zipCode: '90210' })
      ).rejects.toThrow('Pharmacy service unavailable');
    });

    it('should handle insurance verification timeout', async () => {
      vi.mocked(mockClient.callZome).mockRejectedValue(
        new Error('Request timeout')
      );

      await expect(
        bridge.prescriptions.checkCoverage({
          prescriptionId: 'rx-1',
          insuranceId: 'ins-1',
          memberId: 'member-1',
        })
      ).rejects.toThrow('Request timeout');
    });
  });
});
