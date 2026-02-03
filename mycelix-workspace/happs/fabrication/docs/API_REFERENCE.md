# Fabrication hApp API Reference

> Complete zome function reference

This document provides the complete API reference for all zome functions in the Mycelix Fabrication hApp.

## Zome Overview

| Zome | Purpose | Entry Types | Link Types |
|------|---------|-------------|------------|
| `designs` | Design CRUD & discovery | Design, DesignFile | AuthorToDesigns, CategoryToDesigns, ParentToForks |
| `printers` | Printer registry & matching | Printer | OwnerToPrinters, LocationToPrinters |
| `prints` | Print job lifecycle | PrintJob, PrintRecord | DesignToPrints, PrinterToPrints |
| `materials` | Material specifications | Material | TypeToMaterials, CertToMaterials |
| `verification` | Safety verification | Verification, SafetyClaim | DesignToVerifications |
| `bridge` | Cross-hApp integration | BridgeMessage, RepairWorkflow | - |
| `cincinnati` | Quality monitoring | CincinnatiSession, CincinnatiReport | JobToSession |

---

## Designs Zome

### `create_design`

Create a new design entry.

```rust
#[hdk_extern]
pub fn create_design(input: CreateDesignInput) -> ExternResult<Record>
```

**Input:**
```typescript
interface CreateDesignInput {
  title: string;
  description: string;
  category: DesignCategory;
  safetyClass: SafetyClass;
  license: License;
  files?: DesignFile[];
  parametricSchema?: ParametricSchema;
  repairManifest?: RepairManifest;
}
```

**Returns:** `Record` containing the created Design entry

**Example:**
```typescript
const result = await callZome({
  zome_name: 'designs',
  fn_name: 'create_design',
  payload: {
    title: 'Widget Mount',
    description: 'Universal mounting bracket',
    category: 'Parts',
    safetyClass: 'Class1Functional',
    license: 'OpenHardware',
  },
});
```

---

### `get_design`

Retrieve a design by its action hash.

```rust
#[hdk_extern]
pub fn get_design(hash: ActionHash) -> ExternResult<Option<Record>>
```

**Input:** `ActionHash` - The hash of the design creation action

**Returns:** `Option<Record>` - The design record if found

---

### `update_design`

Update an existing design.

```rust
#[hdk_extern]
pub fn update_design(input: UpdateDesignInput) -> ExternResult<Record>
```

**Input:**
```typescript
interface UpdateDesignInput {
  originalHash: ActionHash;
  title?: string;
  description?: string;
  files?: DesignFile[];
  parametricSchema?: ParametricSchema;
}
```

**Returns:** `Record` containing the updated Design entry

---

### `delete_design`

Delete a design (marks as deleted, doesn't remove from DHT).

```rust
#[hdk_extern]
pub fn delete_design(hash: ActionHash) -> ExternResult<ActionHash>
```

**Returns:** `ActionHash` of the deletion action

---

### `fork_design`

Create a derivative of an existing design.

```rust
#[hdk_extern]
pub fn fork_design(input: ForkDesignInput) -> ExternResult<Record>
```

**Input:**
```typescript
interface ForkDesignInput {
  parentHash: ActionHash;
  title: string;
  description?: string;
  modifications?: DesignModifications;
}
```

**Returns:** `Record` containing the forked Design entry

---

### `get_design_history`

Get all versions of a design.

```rust
#[hdk_extern]
pub fn get_design_history(hash: ActionHash) -> ExternResult<Vec<Record>>
```

**Returns:** `Vec<Record>` - All versions in chronological order

---

### `get_design_forks`

Get all forks of a design.

```rust
#[hdk_extern]
pub fn get_design_forks(hash: ActionHash) -> ExternResult<Vec<Record>>
```

**Returns:** `Vec<Record>` - All designs forked from this parent

---

### `get_designs_by_author`

Get all designs created by an agent.

```rust
#[hdk_extern]
pub fn get_designs_by_author(author: AgentPubKey) -> ExternResult<Vec<Record>>
```

**Returns:** `Vec<Record>` - All designs by the specified author

---

### `get_designs_by_category`

Get all designs in a category.

```rust
#[hdk_extern]
pub fn get_designs_by_category(category: DesignCategory) -> ExternResult<Vec<Record>>
```

**Returns:** `Vec<Record>` - All designs in the category

---

### `search_designs`

Search designs by query.

```rust
#[hdk_extern]
pub fn search_designs(query: DesignSearchQuery) -> ExternResult<Vec<Record>>
```

**Input:**
```typescript
interface DesignSearchQuery {
  text?: string;           // Full-text search
  category?: DesignCategory;
  safetyClass?: SafetyClass;
  minEpistemicScore?: number;
  hasRepairManifest?: boolean;
  parentProductModel?: string;
  limit?: number;
  offset?: number;
}
```

**Returns:** `Vec<Record>` - Matching designs

---

### `get_featured_designs`

Get top-rated/featured designs.

```rust
#[hdk_extern]
pub fn get_featured_designs(limit: u32) -> ExternResult<Vec<Record>>
```

**Returns:** `Vec<Record>` - Featured designs sorted by quality score

---

### `generate_variant`

Generate a parametric variant of a design.

```rust
#[hdk_extern]
pub fn generate_variant(input: GenerateVariantInput) -> ExternResult<DesignFile>
```

**Input:**
```typescript
interface GenerateVariantInput {
  designHash: ActionHash;
  parameters: Record<string, ParameterValue>;
  outputFormat: FileFormat;
}
```

**Returns:** `DesignFile` - Generated file with specified parameters

---

## Printers Zome

### `register_printer`

Register a new printer.

```rust
#[hdk_extern]
pub fn register_printer(input: RegisterPrinterInput) -> ExternResult<Record>
```

**Input:**
```typescript
interface RegisterPrinterInput {
  name: string;
  printerType: PrinterType;
  capabilities: PrinterCapabilities;
  materialsAvailable: MaterialType[];
  location?: GeoLocation;
  rates?: PrinterRates;
}
```

**Returns:** `Record` containing the registered Printer entry

---

### `get_printer`

Get a printer by hash.

```rust
#[hdk_extern]
pub fn get_printer(hash: ActionHash) -> ExternResult<Option<Record>>
```

---

### `update_printer`

Update printer information.

```rust
#[hdk_extern]
pub fn update_printer(input: UpdatePrinterInput) -> ExternResult<Record>
```

---

### `deactivate_printer`

Deactivate a printer (marks as inactive).

```rust
#[hdk_extern]
pub fn deactivate_printer(hash: ActionHash) -> ExternResult<ActionHash>
```

---

### `get_my_printers`

Get all printers owned by the calling agent.

```rust
#[hdk_extern]
pub fn get_my_printers() -> ExternResult<Vec<Record>>
```

---

### `find_printers_nearby`

Find printers near a location.

```rust
#[hdk_extern]
pub fn find_printers_nearby(input: NearbyQuery) -> ExternResult<Vec<Record>>
```

**Input:**
```typescript
interface NearbyQuery {
  location: GeoLocation;
  radiusKm: number;
}
```

---

### `find_printers_by_capability`

Find printers matching capability requirements.

```rust
#[hdk_extern]
pub fn find_printers_by_capability(requirements: PrinterRequirements) -> ExternResult<Vec<Record>>
```

**Input:**
```typescript
interface PrinterRequirements {
  minBuildVolume?: BuildVolume;
  requiredMaterials?: MaterialType[];
  requireEnclosure?: boolean;
  requireMultiMaterial?: boolean;
  minTempHotend?: number;
}
```

---

### `get_available_printers`

Get all printers with 'Available' status.

```rust
#[hdk_extern]
pub fn get_available_printers() -> ExternResult<Vec<Record>>
```

---

### `match_design_to_printers`

Match a design to compatible printers.

```rust
#[hdk_extern]
pub fn match_design_to_printers(design_hash: ActionHash) -> ExternResult<Vec<PrinterMatch>>
```

**Returns:**
```typescript
interface PrinterMatch {
  printer: Printer;
  compatibilityScore: number;  // 0.0-1.0
  estimatedTime: number;       // minutes
  estimatedCost?: number;
  materialMatch: boolean;
  volumeMatch: boolean;
}
```

---

### `check_printer_compatibility`

Check if a specific printer can print a design.

```rust
#[hdk_extern]
pub fn check_printer_compatibility(input: CompatibilityCheck) -> ExternResult<CompatibilityResult>
```

**Input:**
```typescript
interface CompatibilityCheck {
  printerHash: ActionHash;
  designHash: ActionHash;
}
```

---

### `update_availability`

Update printer availability status.

```rust
#[hdk_extern]
pub fn update_availability(input: UpdateAvailabilityInput) -> ExternResult<Record>
```

**Input:**
```typescript
interface UpdateAvailabilityInput {
  printerHash: ActionHash;
  status: AvailabilityStatus;
}
```

---

### `get_printer_queue`

Get print jobs queued for a printer.

```rust
#[hdk_extern]
pub fn get_printer_queue(hash: ActionHash) -> ExternResult<Vec<Record>>
```

---

## Prints Zome

### `create_print_job`

Create a new print job.

```rust
#[hdk_extern]
pub fn create_print_job(input: CreatePrintJobInput) -> ExternResult<Record>
```

**Input:**
```typescript
interface CreatePrintJobInput {
  designHash: ActionHash;
  printerHash: ActionHash;
  settings: PrintSettings;
  groundingRequest?: GroundingRequest;
}
```

---

### `accept_print_job`

Accept a pending print job (printer operator).

```rust
#[hdk_extern]
pub fn accept_print_job(hash: ActionHash) -> ExternResult<Record>
```

---

### `start_print`

Start printing a job.

```rust
#[hdk_extern]
pub fn start_print(hash: ActionHash) -> ExternResult<Record>
```

---

### `update_print_progress`

Update print progress percentage.

```rust
#[hdk_extern]
pub fn update_print_progress(input: UpdateProgressInput) -> ExternResult<Record>
```

**Input:**
```typescript
interface UpdateProgressInput {
  jobHash: ActionHash;
  progressPercent: number;  // 0-100
}
```

---

### `complete_print`

Complete a print job with results.

```rust
#[hdk_extern]
pub fn complete_print(input: CompletePrintInput) -> ExternResult<Record>
```

**Input:**
```typescript
interface CompletePrintInput {
  jobHash: ActionHash;
  result: PrintResult;
  qualityAssessment: QualityAssessment;
  materialUsedGrams: number;
  photos?: string[];
  notes?: string;
  issues?: PrintIssue[];
}
```

**Returns:** `Record` containing the PrintRecord entry

---

### `cancel_print`

Cancel a print job.

```rust
#[hdk_extern]
pub fn cancel_print(input: CancelPrintInput) -> ExternResult<Record>
```

**Input:**
```typescript
interface CancelPrintInput {
  jobHash: ActionHash;
  reason: string;
}
```

---

### `get_my_print_jobs`

Get all print jobs requested by the calling agent.

```rust
#[hdk_extern]
pub fn get_my_print_jobs() -> ExternResult<Vec<Record>>
```

---

### `get_printer_jobs`

Get all print jobs for a printer.

```rust
#[hdk_extern]
pub fn get_printer_jobs(printer_hash: ActionHash) -> ExternResult<Vec<Record>>
```

---

### `get_design_prints`

Get all print records for a design.

```rust
#[hdk_extern]
pub fn get_design_prints(design_hash: ActionHash) -> ExternResult<Vec<Record>>
```

---

### `get_print_statistics`

Get aggregated statistics for a design's prints.

```rust
#[hdk_extern]
pub fn get_print_statistics(design_hash: ActionHash) -> ExternResult<PrintStatistics>
```

**Returns:**
```typescript
interface PrintStatistics {
  totalPrints: number;
  successRate: number;
  averageQualityScore: number;
  averagePogScore: number;
  averageTimeMinutes: number;
  commonIssues: [PrintIssue, number][];
  recommendedSettings?: PrintSettings;
}
```

---

## Materials Zome

### `create_material`

Create a new material specification.

```rust
#[hdk_extern]
pub fn create_material(input: CreateMaterialInput) -> ExternResult<Record>
```

---

### `get_material`

Get a material by hash.

```rust
#[hdk_extern]
pub fn get_material(hash: ActionHash) -> ExternResult<Option<Record>>
```

---

### `get_materials_by_type`

Get all materials of a specific type.

```rust
#[hdk_extern]
pub fn get_materials_by_type(material_type: MaterialType) -> ExternResult<Vec<Record>>
```

---

### `find_compatible_materials`

Find materials compatible with a design.

```rust
#[hdk_extern]
pub fn find_compatible_materials(design_hash: ActionHash) -> ExternResult<Vec<Record>>
```

---

### `get_food_safe_materials`

Get all food-safe certified materials.

```rust
#[hdk_extern]
pub fn get_food_safe_materials() -> ExternResult<Vec<Record>>
```

---

### `get_materials_by_certification`

Get materials with a specific certification.

```rust
#[hdk_extern]
pub fn get_materials_by_certification(cert: CertificationType) -> ExternResult<Vec<Record>>
```

---

## Verification Zome

### `submit_verification`

Submit a verification for a design.

```rust
#[hdk_extern]
pub fn submit_verification(input: SubmitVerificationInput) -> ExternResult<Record>
```

**Input:**
```typescript
interface SubmitVerificationInput {
  designHash: ActionHash;
  verificationType: VerificationType;
  result: VerificationResult;
  evidence: string[];  // IPFS CIDs
  verifierCredentials: string[];
}
```

---

### `get_design_verifications`

Get all verifications for a design.

```rust
#[hdk_extern]
pub fn get_design_verifications(design_hash: ActionHash) -> ExternResult<Vec<Record>>
```

---

### `get_verification_summary`

Get verification summary for a design.

```rust
#[hdk_extern]
pub fn get_verification_summary(design_hash: ActionHash) -> ExternResult<VerificationSummary>
```

**Returns:**
```typescript
interface VerificationSummary {
  totalVerifications: number;
  passedVerifications: number;
  failedVerifications: number;
  pendingVerifications: number;
  verificationsByType: Record<VerificationType, VerificationResult[]>;
  overallConfidence: number;
}
```

---

### `submit_safety_claim`

Submit a safety claim (bridges to Knowledge hApp).

```rust
#[hdk_extern]
pub fn submit_safety_claim(input: SubmitSafetyClaimInput) -> ExternResult<Record>
```

---

### `create_verification_request`

Create a request for verification.

```rust
#[hdk_extern]
pub fn create_verification_request(input: CreateVerificationRequestInput) -> ExternResult<Record>
```

---

### `get_open_verification_requests`

Get all open verification requests.

```rust
#[hdk_extern]
pub fn get_open_verification_requests() -> ExternResult<Vec<Record>>
```

---

## Bridge Zome

### `query_fabrication`

Query fabrication data (for other hApps).

```rust
#[hdk_extern]
pub fn query_fabrication(input: FabricationQuery) -> ExternResult<FabQueryResult>
```

**Input:**
```typescript
interface FabricationQuery {
  queryType: FabQueryType;
  parameters: string;  // JSON
}

type FabQueryType =
  | 'GetDesign'
  | 'FindPrinters'
  | 'CheckVerification'
  | 'GetMaterialSuppliers'
  | 'GetPrintStatistics';
```

---

### `process_repair_prediction`

Process a repair prediction from Property hApp.

```rust
#[hdk_extern]
pub fn process_repair_prediction(prediction: RepairPrediction) -> ExternResult<RepairWorkflow>
```

**Input:**
```typescript
interface RepairPrediction {
  propertyAssetHash: ActionHash;
  assetModel: string;
  predictedFailureComponent: string;
  failureProbability: number;
  estimatedFailureDate: number;
  confidenceIntervalDays: number;
  sensorDataSummary: string;
  recommendedAction: RepairAction;
}
```

**Returns:**
```typescript
interface RepairWorkflow {
  predictionHash: ActionHash;
  status: RepairWorkflowStatus;
  designHash?: ActionHash;
  printerHash?: ActionHash;
  hearthFundingHash?: ActionHash;
  printJobHash?: ActionHash;
}
```

---

### `list_design_on_marketplace`

List a design on the marketplace.

```rust
#[hdk_extern]
pub fn list_design_on_marketplace(input: MarketplaceListingInput) -> ExternResult<Record>
```

---

### `link_material_to_supplier`

Link material to supply chain supplier.

```rust
#[hdk_extern]
pub fn link_material_to_supplier(input: SupplierLinkInput) -> ExternResult<Record>
```

---

## Cincinnati Zome

### `start_monitoring_session`

Start a Cincinnati monitoring session.

```rust
#[hdk_extern]
pub fn start_monitoring_session(input: StartSessionInput) -> ExternResult<CincinnatiSession>
```

---

### `record_sensor_sample`

Record a sensor sample.

```rust
#[hdk_extern]
pub fn record_sensor_sample(input: RecordSampleInput) -> ExternResult<()>
```

---

### `record_anomaly_event`

Record an anomaly event.

```rust
#[hdk_extern]
pub fn record_anomaly_event(input: RecordAnomalyInput) -> ExternResult<()>
```

---

### `complete_monitoring_session`

Complete a monitoring session and generate report.

```rust
#[hdk_extern]
pub fn complete_monitoring_session(session_id: String) -> ExternResult<CincinnatiReport>
```

---

### `get_session_report`

Get the report for a completed session.

```rust
#[hdk_extern]
pub fn get_session_report(session_id: String) -> ExternResult<Option<CincinnatiReport>>
```

---

## TypeScript SDK

The TypeScript SDK wraps all zome calls:

```typescript
import { getFabricationService } from '@mycelix/sdk/integrations/fabrication';

const fab = getFabricationService();

// Design operations
const design = fab.createDesign({ ... });
const designs = fab.getDesignsByCategory('Repair');

// Printer operations
const printer = fab.registerPrinter({ ... });
const matches = fab.matchDesignToPrinters(design.id);

// Print job operations
const job = fab.createPrintJob({ ... });
fab.acceptPrintJob(job.id);
fab.startPrint(job.id);
const record = fab.completePrint(job.id, 'Success', assessment, grams);

// Reputation
const score = fab.getPrinterTrustScore(printer.id);
const trustworthy = fab.isPrinterTrustworthy(printer.id, 0.7);

// Anticipatory repair
const workflow = fab.processRepairPrediction(prediction);

// Events
fab.emitEvent('DesignPublished', design.id, { title: design.title });
```

See the SDK-TS documentation for complete TypeScript API reference.

---

*Complete API access for decentralized manufacturing.*
