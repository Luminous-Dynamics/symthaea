/**
 * Health ↔ Marketplace Integration Module
 *
 * Provides cross-module bridges between mycelix-health and mycelix-marketplace.
 * Key use cases:
 * - Prescription ordering and pharmacy integration
 * - Medical equipment and supply purchases
 * - Health product verification
 * - Insurance/HSA payment processing
 * - Durable medical equipment (DME) rentals
 *
 * @module @mycelix/sdk/integrations/health-marketplace
 */

import type { ActionHash } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

/**
 * Product categories in health marketplace
 */
export type HealthProductCategory =
  | 'Prescription' // Rx medications
  | 'PrescriptionMedication' // Alias for prescription medications
  | 'ControlledSubstance' // DEA scheduled
  | 'Opioid' // Opioid medications
  | 'Benzodiazepine' // Benzodiazepine medications
  | 'Stimulant' // Stimulant medications
  | 'OTC' // Over-the-counter medications
  | 'Supplement' // Vitamins, supplements
  | 'MedicalDevice' // Home monitoring, CPAP, etc.
  | 'MedicalSupply' // Medical supplies
  | 'DME' // Durable medical equipment (wheelchairs, etc.)
  | 'Supplies' // Bandages, test strips, etc.
  | 'PersonalCare' // Hygiene, skincare
  | 'Fitness' // Exercise equipment, wearables
  | 'Nutrition' // Special dietary foods
  | 'Telehealth' // Virtual care services
  | 'LabTest' // At-home lab test kits
  | 'Other';

/**
 * Prescription status
 */
export type PrescriptionStatus =
  | 'Active'
  | 'Filled'
  | 'PartialFill'
  | 'Expired'
  | 'Cancelled'
  | 'OnHold'
  | 'TransferredOut';

/**
 * Order fulfillment status
 */
export type FulfillmentStatus =
  | 'Pending'
  | 'Processing'
  | 'ReadyForPickup'
  | 'Shipped'
  | 'OutForDelivery'
  | 'Delivered'
  | 'Cancelled'
  | 'Returned'
  | 'OnHold'
  | 'Confirmed';

/**
 * Payment method types
 */
export type PaymentMethod =
  | 'Insurance'
  | 'HSA'
  | 'FSA'
  | 'CreditCard'
  | 'DebitCard'
  | 'HEARTH' // Mycelix native token
  | 'Assistance' // Patient assistance program
  | 'Cash';

/**
 * Prescription record linked to marketplace
 */
export interface MarketplacePrescription {
  prescriptionId: string;
  prescriptionHash?: ActionHash; // Link to health module
  patientHash: ActionHash;
  prescriberId?: string;
  prescriberName?: string;
  providerName?: string; // Alias for prescriberName
  medicationName: string;
  medicationNdc?: string;
  strength?: string;
  dosage?: string; // Alias for strength
  form?: string; // tablet, capsule, liquid, etc.
  quantity: number;
  refillsRemaining: number;
  refillsTotal?: number;
  daySupply?: number;
  daysSupply?: number; // Alias for daySupply
  directions?: string;
  status: PrescriptionStatus;
  writtenDate?: number;
  expirationDate: number;
  lastFilledDate?: number;
  lastFillDate?: number; // Alias for lastFilledDate
  pharmacyId?: string;
  pharmacyName?: string;
  substitutionAllowed?: boolean;
  priorAuthRequired?: boolean;
  priorAuthStatus?: 'Pending' | 'Approved' | 'Denied';
  createdAt?: number;
}

/**
 * Pharmacy for prescription fulfillment
 */
export interface Pharmacy {
  pharmacyId: string;
  name: string;
  pharmacyType: 'Retail' | 'Mail' | 'Specialty' | 'Compounding' | 'Hospital';
  address: {
    street: string;
    city: string;
    state: string;
    zip: string;
    country: string;
  };
  phone: string;
  fax?: string;
  email?: string;
  hours: {
    monday: string;
    tuesday: string;
    wednesday: string;
    thursday: string;
    friday: string;
    saturday: string;
    sunday: string;
  };
  services: string[]; // Immunizations, compounding, delivery, etc.
  acceptsInsurance: string[];
  rating: number;
  reviewCount: number;
  verified: boolean;
  matlTrustScore: number;
}

/**
 * Price quote for a prescription
 */
export interface PrescriptionPriceQuote {
  quoteId?: string;
  prescriptionId?: string;
  pharmacyId: string;
  pharmacyName: string;
  medicationName?: string;
  quantity?: number;
  daySupply?: number;
  price: number; // Primary price field
  prices?: {
    retail: number;
    insurance?: number;
    copay?: number;
    discount?: number;
    manufacturer?: number; // Manufacturer coupon
    goodRx?: number;
    costPlus?: number;
  };
  estimatedTotal?: number;
  estimatedWaitMinutes?: number;
  readyTime?: string; // "2 hours", "Next day"
  distance?: number; // Distance in miles
  deliveryAvailable?: boolean;
  deliveryFee?: number;
  expiresAt?: number;
}

/**
 * Prescription order
 */
export interface PrescriptionOrder {
  orderId: string;
  prescriptionId: string;
  patientHash: ActionHash;
  pharmacyId: string;
  quantity?: number;
  priceQuoteId?: string;
  paymentMethod?: PaymentMethod;
  insuranceId?: string;
  copay?: number;
  total?: number;
  fulfillmentMethod?: 'Pickup' | 'Delivery' | 'Mail';
  deliveryMethod?: 'Pickup' | 'Delivery' | 'Mail'; // Alias
  deliveryAddress?: {
    street: string;
    city: string;
    state: string;
    zip: string;
  };
  status: FulfillmentStatus;
  estimatedReady?: number;
  actualReady?: number;
  pickedUpAt?: number;
  deliveredAt?: number;
  pharmacistNotes?: string;
  createdAt?: number;
  updatedAt?: number;
}

/**
 * Medical device/supply product
 */
export interface HealthProduct {
  productId: string;
  sellerId: string;
  name: string;
  description: string;
  category: HealthProductCategory;
  subcategory?: string;
  brand: string;
  manufacturer: string;
  sku: string;
  upc?: string;
  ndcCode?: string; // For medications/OTC
  fdaCleared: boolean;
  fdaClearanceNumber?: string;
  requiresPrescription: boolean;
  linkedPrescriptionTypes?: string[];
  specifications: Record<string, string>;
  images: string[];
  price: number;
  insuranceEligible: boolean;
  hsaFsaEligible: boolean;
  medicareCovered?: boolean;
  medicaidCovered?: boolean;
  inStock: boolean;
  stockQuantity: number;
  leadTimeDays?: number;
  rentalAvailable: boolean;
  rentalPricePerDay?: number;
  rating: number;
  reviewCount: number;
  matlTrustScore: number;
}

/**
 * Health product order
 */
export interface HealthProductOrder {
  orderId: string;
  patientHash: ActionHash;
  items: Array<{
    productId: string;
    quantity: number;
    pricePerUnit: number;
    isRental: boolean;
    rentalDays?: number;
    prescriptionId?: string; // If requires Rx
  }>;
  subtotal: number;
  tax: number;
  shipping: number;
  insuranceApplied?: number;
  hsaFsaApplied?: number;
  total: number;
  paymentMethod: PaymentMethod;
  paymentStatus: 'Pending' | 'Authorized' | 'Captured' | 'Failed' | 'Refunded';
  shippingAddress: {
    name: string;
    street: string;
    city: string;
    state: string;
    zip: string;
    phone: string;
  };
  shippingMethod: string;
  trackingNumber?: string;
  status: FulfillmentStatus;
  createdAt: number;
  updatedAt: number;
}

/**
 * Insurance coverage check result
 */
export interface CoverageCheckResult {
  productId?: string;
  prescriptionId?: string;
  insuranceId?: string;
  covered: boolean;
  coverageLevel?: 'Full' | 'Partial' | 'Tier1' | 'Tier2' | 'Tier3' | 'NotCovered';
  tier?: string; // Alias for coverageLevel
  copay?: number; // Direct copay amount
  estimatedCopay?: number;
  estimatedCoinsurance?: number;
  deductibleApplied?: number;
  deductibleApplies?: boolean;
  deductibleRemaining?: number;
  priorAuthRequired?: boolean;
  stepTherapyRequired?: boolean;
  quantity?: number;
  daysSupply?: number;
  quantityLimits?: { max: number; perDays: number };
  alternatives?: Array<{
    productName: string;
    tier: string;
    estimatedCopay: number;
  }>;
  notes?: string;
  message?: string;
}

/**
 * Product verification for safety/authenticity
 */
export interface ProductVerification {
  productId: string;
  verified: boolean;
  verificationDate: number;
  verifiedBy: string;
  fdaStatus: 'Cleared' | 'Approved' | 'Exempt' | 'Unknown' | 'Recalled';
  recallStatus?: {
    isRecalled: boolean;
    recallDate?: number;
    recallReason?: string;
    recallClass?: 'I' | 'II' | 'III';
  };
  authenticityScore: number; // 0-1
  supplierVerified: boolean;
  serialNumberValid?: boolean;
  expirationValid: boolean;
  storageCompliant: boolean;
  warnings: string[];
  recommendations: string[];
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
// Client Classes
// ============================================================================

const HEALTH_ROLE = 'health';
const MARKETPLACE_ROLE = 'marketplace';

/**
 * Client for prescription marketplace operations
 */
export class PrescriptionMarketplaceClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Get patient's active prescriptions for ordering
   */
  async getActivePrescritions(patientHash: ActionHash): Promise<MarketplacePrescription[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'prescription_bridge',
      fn_name: 'get_active_prescriptions',
      payload: patientHash,
    }) as Promise<MarketplacePrescription[]>;
  }

  /**
   * Get patient prescriptions (alias for getActivePrescritions)
   */
  async getPatientPrescriptions(patientHash: ActionHash): Promise<MarketplacePrescription[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'prescriptions',
      fn_name: 'get_patient_prescriptions',
      payload: { patientHash },
    }) as Promise<MarketplacePrescription[]>;
  }

  /**
   * Get a single prescription by ID
   */
  async getPrescription(prescriptionId: string): Promise<MarketplacePrescription> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'prescriptions',
      fn_name: 'get_prescription',
      payload: { prescriptionId },
    }) as Promise<MarketplacePrescription>;
  }

  /**
   * Find pharmacies near a location
   */
  async findPharmacies(
    lat: number,
    lng: number,
    radiusMiles: number,
    filters?: {
      pharmacyType?: Pharmacy['pharmacyType'];
      acceptsInsurance?: string;
      services?: string[];
      openNow?: boolean;
    }
  ): Promise<Pharmacy[]> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'pharmacy',
      fn_name: 'find_pharmacies',
      payload: { lat, lng, radius_miles: radiusMiles, filters },
    }) as Promise<Pharmacy[]>;
  }

  /**
   * Get price quotes for a prescription from multiple pharmacies
   */
  async getPriceQuotes(
    prescriptionId: string,
    options?: { zipCode?: string; radius?: number; pharmacyIds?: string[] }
  ): Promise<PrescriptionPriceQuote[]> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'pharmacy_pricing',
      fn_name: 'get_price_quotes',
      payload: { prescriptionId, options },
    }) as Promise<PrescriptionPriceQuote[]>;
  }

  /**
   * Check insurance coverage for a prescription
   */
  async checkCoverage(options: {
    prescriptionId: string;
    insuranceId: string;
    memberId?: string;
  }): Promise<CoverageCheckResult> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'insurance',
      fn_name: 'check_prescription_coverage',
      payload: options,
    }) as Promise<CoverageCheckResult>;
  }

  /**
   * Place a prescription order
   */
  async placeOrder(
    order: Omit<PrescriptionOrder, 'orderId' | 'status' | 'createdAt' | 'updatedAt'>
  ): Promise<PrescriptionOrder> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'pharmacy',
      fn_name: 'place_prescription_order',
      payload: order,
    }) as Promise<PrescriptionOrder>;
  }

  /**
   * Submit a prescription order
   */
  async submitOrder(order: PrescriptionOrder): Promise<PrescriptionOrder> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'prescription_orders',
      fn_name: 'submit_order',
      payload: order,
    }) as Promise<PrescriptionOrder>;
  }

  /**
   * Transfer prescription between pharmacies
   */
  async transferPrescription(params: {
    prescriptionId: string;
    fromPharmacyId: string;
    toPharmacyId: string;
    patientConsent: boolean;
  }): Promise<{ transferId: string; status: string }> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'prescription_transfers',
      fn_name: 'transfer_prescription',
      payload: params,
    }) as Promise<{ transferId: string; status: string }>;
  }

  /**
   * Get order status
   */
  async getOrderStatus(orderId: string): Promise<PrescriptionOrder> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'pharmacy',
      fn_name: 'get_order_status',
      payload: orderId,
    }) as Promise<PrescriptionOrder>;
  }

  /**
   * Get patient's order history
   */
  async getOrderHistory(patientHash: ActionHash, limit?: number): Promise<PrescriptionOrder[]> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'pharmacy',
      fn_name: 'get_order_history',
      payload: { patient_hash: patientHash, limit: limit || 20 },
    }) as Promise<PrescriptionOrder[]>;
  }

  /**
   * Request prescription transfer to a different pharmacy
   */
  async requestTransfer(
    prescriptionId: string,
    fromPharmacyId: string,
    toPharmacyId: string
  ): Promise<{ transferId: string; status: string; estimatedTime: string }> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'pharmacy',
      fn_name: 'request_transfer',
      payload: {
        prescription_id: prescriptionId,
        from_pharmacy_id: fromPharmacyId,
        to_pharmacy_id: toPharmacyId,
      },
    }) as Promise<{ transferId: string; status: string; estimatedTime: string }>;
  }

  /**
   * Set up auto-refill for a prescription
   */
  async setupAutoRefill(
    prescriptionId: string,
    pharmacyId: string,
    paymentMethod: PaymentMethod
  ): Promise<{ enrolled: boolean; nextRefillDate: number }> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'pharmacy',
      fn_name: 'setup_auto_refill',
      payload: {
        prescription_id: prescriptionId,
        pharmacy_id: pharmacyId,
        payment_method: paymentMethod,
      },
    }) as Promise<{ enrolled: boolean; nextRefillDate: number }>;
  }
}

/**
 * Client for health product marketplace operations
 */
export class HealthProductClient {
  constructor(readonly client: ZomeCallable) {}

  /**
   * Search for health products
   */
  async searchProducts(
    params: {
      query?: string;
      category?: HealthProductCategory;
      priceMin?: number;
      priceMax?: number;
      hsaFsaEligible?: boolean;
      inStockOnly?: boolean;
      minRating?: number;
    }
  ): Promise<HealthProduct[]> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'health_products',
      fn_name: 'search_products',
      payload: params,
    }) as Promise<HealthProduct[]>;
  }

  /**
   * Get product details
   */
  async getProduct(productId: string): Promise<HealthProduct | null> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'products',
      fn_name: 'get_product',
      payload: productId,
    }) as Promise<HealthProduct | null>;
  }

  /**
   * Verify product safety and authenticity
   */
  async verifyProduct(productId: string): Promise<ProductVerification> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'products',
      fn_name: 'verify_product',
      payload: productId,
    }) as Promise<ProductVerification>;
  }

  /**
   * Check if product is compatible with patient's conditions/medications
   */
  async checkCompatibility(
    productId: string,
    patientHash: ActionHash
  ): Promise<{
    compatible: boolean;
    drugInteractions: string[];
    conditionContraindications: string[];
    allergyWarnings: string[];
    recommendations: string[];
  }> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'products',
      fn_name: 'check_patient_compatibility',
      payload: { product_id: productId, patient_hash: patientHash },
    }) as Promise<{
      compatible: boolean;
      drugInteractions: string[];
      conditionContraindications: string[];
      allergyWarnings: string[];
      recommendations: string[];
    }>;
  }

  /**
   * Check insurance/HSA/FSA eligibility
   */
  async checkEligibility(
    productId: string,
    insuranceId?: string
  ): Promise<CoverageCheckResult> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'insurance',
      fn_name: 'check_product_coverage',
      payload: { product_id: productId, insurance_id: insuranceId },
    }) as Promise<CoverageCheckResult>;
  }

  /**
   * Place a product order
   */
  async placeOrder(
    order: Omit<HealthProductOrder, 'orderId' | 'status' | 'paymentStatus' | 'createdAt' | 'updatedAt'>
  ): Promise<HealthProductOrder> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'orders',
      fn_name: 'place_product_order',
      payload: order,
    }) as Promise<HealthProductOrder>;
  }

  /**
   * Get recommended products based on patient's health profile
   */
  async getRecommendations(
    patientHash: ActionHash,
    category?: HealthProductCategory
  ): Promise<HealthProduct[]> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'products',
      fn_name: 'get_recommendations',
      payload: { patient_hash: patientHash, category },
    }) as Promise<HealthProduct[]>;
  }

  /**
   * Get product reviews with trust scores
   */
  async getProductReviews(
    productId: string
  ): Promise<Array<{ reviewId: string; rating: number; trustScore: number; text?: string }>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'health_products',
      fn_name: 'get_product_reviews',
      payload: { productId },
    }) as Promise<Array<{ reviewId: string; rating: number; trustScore: number; text?: string }>>;
  }
}

/**
 * Client for DME rental operations
 */
export class DMERentalClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Get available DME for rental
   */
  async getAvailableRentals(category?: string): Promise<HealthProduct[]> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'dme',
      fn_name: 'get_available_rentals',
      payload: category || null,
    }) as Promise<HealthProduct[]>;
  }

  /**
   * Search available DME for rental
   */
  async searchAvailable(params: {
    equipmentType?: string;
    zipCode?: string;
    startDate?: number;
    endDate?: number;
  }): Promise<Array<{ equipmentId: string; type: string; dailyRate: number }>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'dme',
      fn_name: 'search_available',
      payload: params,
    }) as Promise<Array<{ equipmentId: string; type: string; dailyRate: number }>>;
  }

  /**
   * Create a rental reservation
   */
  async createReservation(params: {
    equipmentId: string;
    patientHash: ActionHash;
    startDate: number;
    endDate: number;
    deliveryAddress: string;
    prescriptionId?: string;
  }): Promise<{ reservationId: string; status: string; totalCost: number }> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'dme',
      fn_name: 'create_reservation',
      payload: params,
    }) as Promise<{ reservationId: string; status: string; totalCost: number }>;
  }

  /**
   * Check Medicare/Medicaid DME coverage
   */
  async checkDMECoverage(params: {
    equipmentId?: string;
    productId?: string;
    memberId?: string;
    insuranceId?: string;
    diagnosis?: string;
    prescriptionId?: string;
  }): Promise<{
    covered: boolean;
    coverageType?: string;
    rentalCovered?: boolean;
    purchaseCovered?: boolean;
    rentalPeriodMonths?: number;
    patientResponsibility: number;
    requiresSupplierEnrollment?: boolean;
    priorAuthRequired?: boolean;
  }> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'dme',
      fn_name: 'check_dme_coverage',
      payload: params,
    }) as Promise<{
      covered: boolean;
      coverageType?: string;
      rentalCovered?: boolean;
      purchaseCovered?: boolean;
      rentalPeriodMonths?: number;
      patientResponsibility: number;
      requiresSupplierEnrollment?: boolean;
      priorAuthRequired?: boolean;
    }>;
  }

  /**
   * Start a DME rental
   */
  async startRental(
    productId: string,
    patientHash: ActionHash,
    prescriptionId: string,
    deliveryAddress: HealthProductOrder['shippingAddress']
  ): Promise<{
    rentalId: string;
    startDate: number;
    monthlyRate: number;
    deliveryScheduled: number;
  }> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'dme',
      fn_name: 'start_rental',
      payload: {
        product_id: productId,
        patient_hash: patientHash,
        prescription_id: prescriptionId,
        delivery_address: deliveryAddress,
      },
    }) as Promise<{
      rentalId: string;
      startDate: number;
      monthlyRate: number;
      deliveryScheduled: number;
    }>;
  }

  /**
   * End a DME rental and schedule pickup
   */
  async endRental(rentalId: string): Promise<{
    endDate: number;
    pickupScheduled: number;
    finalCharges: number;
  }> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'dme',
      fn_name: 'end_rental',
      payload: rentalId,
    }) as Promise<{
      endDate: number;
      pickupScheduled: number;
      finalCharges: number;
    }>;
  }

  /**
   * Get active rentals for a patient
   */
  async getActiveRentals(patientHash: ActionHash): Promise<Array<{ reservationId: string; equipmentId: string; status: string }>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'dme',
      fn_name: 'get_active_rentals',
      payload: { patientHash },
    }) as Promise<Array<{ reservationId: string; equipmentId: string; status: string }>>;
  }
}

// ============================================================================
// Unified Health-Marketplace Bridge
// ============================================================================

/**
 * Unified interface for health-marketplace integration
 */
export class HealthMarketplaceBridge {
  readonly prescriptions: PrescriptionMarketplaceClient;
  readonly products: HealthProductClient;
  readonly dme: DMERentalClient;

  constructor(client: ZomeCallable) {
    this.prescriptions = new PrescriptionMarketplaceClient(client);
    this.products = new HealthProductClient(client);
    this.dme = new DMERentalClient(client);
  }

  /**
   * Get complete shopping context for a patient
   *
   * Includes active prescriptions ready to fill, recommended products,
   * and current orders
   */
  async getPatientShoppingContext(
    patientHash: ActionHash
  ): Promise<{
    prescriptionsToFill: MarketplacePrescription[];
    recommendedProducts: HealthProduct[];
    pendingOrders: Array<PrescriptionOrder | HealthProductOrder>;
    savedPharmacy?: Pharmacy;
    paymentMethods: PaymentMethod[];
  }> {
    const [prescriptions, recommendations, rxOrders, productOrders] = await Promise.all([
      this.prescriptions.getActivePrescritions(patientHash),
      this.products.getRecommendations(patientHash),
      this.prescriptions.getOrderHistory(patientHash, 10),
      this.products.client.callZome({
        role_name: MARKETPLACE_ROLE,
        zome_name: 'orders',
        fn_name: 'get_patient_orders',
        payload: { patient_hash: patientHash, limit: 10 },
      }) as Promise<HealthProductOrder[]>,
    ]);

    // Filter to pending orders only
    const pendingRxOrders = rxOrders.filter(
      (o) => !['Delivered', 'Cancelled', 'Returned'].includes(o.status)
    );
    const pendingProductOrders = productOrders.filter(
      (o) => !['Delivered', 'Cancelled', 'Returned'].includes(o.status)
    );

    // Get saved pharmacy (from most recent order)
    let savedPharmacy: Pharmacy | undefined;
    if (rxOrders.length > 0) {
      const recentOrder = rxOrders[0];
      const pharmacies = await this.prescriptions.findPharmacies(0, 0, 0); // Would need actual location
      savedPharmacy = pharmacies.find((p) => p.pharmacyId === recentOrder.pharmacyId);
    }

    return {
      prescriptionsToFill: prescriptions.filter(
        (p) => p.status === 'Active' && p.refillsRemaining > 0
      ),
      recommendedProducts: recommendations,
      pendingOrders: [...pendingRxOrders, ...pendingProductOrders],
      savedPharmacy,
      paymentMethods: ['Insurance', 'HSA', 'FSA', 'CreditCard', 'HEARTH'],
    };
  }

  /**
   * Smart prescription fill - finds best price across pharmacies
   */
  async findBestPrescriptionPrice(
    prescriptionId: string,
    _patientHash: ActionHash,
    insuranceId?: string,
    preferredPharmacyId?: string
  ): Promise<{
    bestOverall: PrescriptionPriceQuote;
    withInsurance?: PrescriptionPriceQuote;
    withCoupons?: PrescriptionPriceQuote;
    atPreferred?: PrescriptionPriceQuote;
    allOptions: PrescriptionPriceQuote[];
  }> {
    const quotes = await this.prescriptions.getPriceQuotes(prescriptionId);

    // Sort by estimated total
    const sorted = [...quotes].sort((a, b) => (a.estimatedTotal ?? 0) - (b.estimatedTotal ?? 0));

    let withInsurance: PrescriptionPriceQuote | undefined;
    if (insuranceId) {
      withInsurance = sorted.find((q) => q.prices?.insurance !== undefined);
    }

    const withCoupons = sorted.find(
      (q) => q.prices?.discount || q.prices?.goodRx || q.prices?.manufacturer
    );

    const atPreferred = preferredPharmacyId
      ? sorted.find((q) => q.pharmacyId === preferredPharmacyId)
      : undefined;

    return {
      bestOverall: sorted[0],
      withInsurance,
      withCoupons,
      atPreferred,
      allOptions: sorted,
    };
  }

  /**
   * Validate order for safety before placing
   */
  async validateOrderSafety(
    patientHash: ActionHash,
    items: Array<{ type: 'prescription' | 'product'; id: string }>
  ): Promise<{
    safe: boolean;
    drugInteractions: string[];
    duplicateTherapies: string[];
    allergyWarnings: string[];
    quantityWarnings: string[];
    recommendations: string[];
  }> {
    // Would call health module to check patient profile against order items
    return this.products.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'order_safety',
      fn_name: 'validate_order',
      payload: { patient_hash: patientHash, items },
    }) as Promise<{
      safe: boolean;
      drugInteractions: string[];
      duplicateTherapies: string[];
      allergyWarnings: string[];
      quantityWarnings: string[];
      recommendations: string[];
    }>;
  }

  /**
   * Get aggregated marketplace metrics for a patient
   */
  async getMarketplaceMetrics(patientHash: ActionHash): Promise<{
    activePrescriptions: number;
    pendingOrders: number;
    activeRentals: number;
  }> {
    const [prescriptions, orders, rentals] = await Promise.all([
      this.prescriptions.getPatientPrescriptions(patientHash),
      this.prescriptions.getOrderHistory(patientHash, 20),
      this.dme.getActiveRentals(patientHash),
    ]);

    const pendingOrders = orders.filter(
      (o) => !['Delivered', 'Cancelled', 'Returned'].includes(o.status)
    );

    return {
      activePrescriptions: prescriptions.length,
      pendingOrders: pendingOrders.length,
      activeRentals: rentals.length,
    };
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a Health-Marketplace bridge instance
 */
export function createHealthMarketplaceBridge(client: ZomeCallable): HealthMarketplaceBridge {
  return new HealthMarketplaceBridge(client);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate days until prescription can be refilled
 * (Most insurance requires waiting until 75-80% of supply used)
 */
export function daysUntilRefillable(
  prescription: MarketplacePrescription,
  refillThreshold = 0.75
): number {
  // Get the last fill date (support both property names)
  const lastFilledDate = prescription.lastFillDate ?? prescription.lastFilledDate;

  // If never filled, can refill immediately
  if (!lastFilledDate) {
    return 0;
  }

  // Get days supply (support both property names)
  const daySupply = prescription.daysSupply ?? prescription.daySupply ?? 30;

  const now = Date.now();
  const refillableDate = lastFilledDate + daySupply * refillThreshold * 24 * 60 * 60 * 1000;
  const daysRemaining = Math.ceil((refillableDate - now) / (24 * 60 * 60 * 1000));
  return Math.max(0, daysRemaining);
}

/**
 * Check if a prescription is about to expire (or already expired)
 */
export function isPrescriptionExpiringSoon(
  prescription: MarketplacePrescription,
  daysThreshold = 30
): boolean {
  const now = Date.now();
  const expirationMs = prescription.expirationDate;
  const daysUntilExpiration = (expirationMs - now) / (24 * 60 * 60 * 1000);
  // Return true if expiring within threshold OR already expired
  return daysUntilExpiration <= daysThreshold;
}

/**
 * Calculate potential savings from price quotes
 */
export function calculatePotentialSavings(
  quotes: PrescriptionPriceQuote[]
): {
  lowestPrice: number;
  highestPrice: number;
  maxSavings: number;
  lowestPricePharmacy?: string;
} {
  if (!quotes || quotes.length === 0) {
    return {
      lowestPrice: 0,
      highestPrice: 0,
      maxSavings: 0,
    };
  }

  const prices = quotes.map((q) => q.price);
  const lowestPrice = Math.min(...prices);
  const highestPrice = Math.max(...prices);
  const lowestQuote = quotes.find((q) => q.price === lowestPrice);

  return {
    lowestPrice,
    highestPrice,
    maxSavings: highestPrice - lowestPrice,
    lowestPricePharmacy: lowestQuote?.pharmacyName,
  };
}

/**
 * Determine if product category requires prescription verification
 */
export function requiresPrescriptionVerification(category: HealthProductCategory): boolean {
  const requiresVerification: HealthProductCategory[] = [
    'Prescription',
    'PrescriptionMedication',
    'ControlledSubstance',
    'Opioid',
    'Benzodiazepine',
    'Stimulant',
    'DME',
  ];
  return requiresVerification.includes(category);
}

/**
 * Format prescription directions for display
 */
export function formatDirections(params: {
  frequency: string;
  route: string;
  timing?: string;
  duration?: string;
  maxDailyDose?: string;
  prnReason?: string;
}): string {
  const parts: string[] = [];

  // Add frequency
  parts.push(`Take ${params.frequency}`);

  // Add route
  parts.push(`by ${params.route} route`);

  // Add timing if provided
  if (params.timing) {
    parts.push(params.timing);
  }

  // Add duration if provided
  if (params.duration) {
    parts.push(`for ${params.duration}`);
  }

  // Add PRN reason if provided
  if (params.prnReason) {
    parts.push(`for ${params.prnReason}`);
  }

  // Add max daily dose if provided
  if (params.maxDailyDose) {
    parts.push(`(max ${params.maxDailyDose} per day)`);
  }

  return parts.join(' ');
}

// ============================================================================
// Exports
// ============================================================================

export default {
  HealthMarketplaceBridge,
  PrescriptionMarketplaceClient,
  HealthProductClient,
  DMERentalClient,
  createHealthMarketplaceBridge,
  daysUntilRefillable,
  isPrescriptionExpiringSoon,
  calculatePotentialSavings,
  requiresPrescriptionVerification,
  formatDirections,
};
