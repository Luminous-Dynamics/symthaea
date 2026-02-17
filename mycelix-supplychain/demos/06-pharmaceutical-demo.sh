#!/usr/bin/env bash
# Mycelix ERP Demo: MediCure Pharmaceuticals
# Scenario: FDA-Regulated Drug Manufacturing with Complete Compliance

set -e

API_BASE="http://localhost:8000/v1"
FIN_BASE="$API_BASE/fin"

echo "💊 Mycelix ERP Demo: MediCure Pharmaceuticals"
echo "==============================================="
echo ""
echo "Scenario: FDA-Regulated Drug Manufacturing"
echo "This demo shows:"
echo "  - Complete chain of custody for pharmaceutical ingredients"
echo "  - Batch traceability and lot tracking"
echo "  - Quality control at every stage (cGMP compliance)"
echo "  - Regulatory audit trail (21 CFR Part 11 ready)"
echo "  - Recall capability within 24 hours"
echo ""

if ! curl -s "$API_BASE/health" > /dev/null 2>&1; then
    echo "❌ Service not running"
    exit 1
fi

echo "✅ Service is running"
echo ""
echo "⚠️  COMPLIANCE NOTE: This demo shows blockchain-verified provenance"
echo "   for pharmaceutical supply chains. Every event is cryptographically"
echo "   signed and tamper-proof for FDA audits."
echo ""

# Step 1: API Supplier (Active Pharmaceutical Ingredient)
echo "🧪 Step 1: Receiving Active Pharmaceutical Ingredient (API)"
echo "----------------------------------------"

echo "Creating API supplier (Teva Pharmaceuticals)..."
API_SUPPLIER=$(curl -s -X POST "$FIN_BASE/vendors" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Teva API Manufacturing",
    "email": "supply@tevaapi.com",
    "type": "pharmaceutical_supplier",
    "payment_terms_days": 60
  }')
API_SUPPLIER_ID=$(echo "$API_SUPPLIER" | jq -r '.id')
echo "✅ API Supplier: Teva ($API_SUPPLIER_ID)"
echo ""

sleep 1

echo "Receiving API shipment with full chain of custody..."
API_RECEIPT=$(curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "API_RECEIPT",
    "product_id": "lisinopril-api-batch-2024-L001",
    "location": "MediCure - Receiving Dock (Cleanroom Ante)",
    "timestamp": "2024-02-01T08:00:00Z",
    "actor": "QA Inspector - Dr. Patricia Chen, PhD",
    "metadata": {
      "api_name": "Lisinopril Dihydrate USP",
      "lot_number": "TEVA-LIS-2024-L001",
      "quantity_kg": 100,
      "supplier": "Teva API Manufacturing",
      "coa_number": "COA-2024-001",
      "purity_percent": 99.87,
      "moisture_percent": 0.08,
      "heavy_metals_ppm": 8,
      "endotoxin_level": "<0.25 EU/mg",
      "temp_on_arrival_c": 22,
      "container_intact": true,
      "seal_intact": true,
      "dmd_file_number": "DMF-012345",
      "expiration": "2026-02-01",
      "quarantine_status": "pending_release"
    }
  }')
echo "✅ API received: 100 kg Lisinopril (Lot TEVA-LIS-2024-L001)"
echo "   Status: QUARANTINED (pending QC release)"
echo "   Purity: 99.87% (spec: ≥99.0%)"
echo "   Certificate of Analysis: COA-2024-001"
echo ""

sleep 1

# Step 2: QC Testing and Release
echo "🔬 Step 2: Quality Control Testing"
echo "----------------------------------------"

echo "Running QC tests (HPLC, Karl Fischer, ICP-MS)..."
QC_TEST=$(curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "QC_TESTING",
    "product_id": "lisinopril-api-batch-2024-L001",
    "location": "MediCure - QC Laboratory",
    "timestamp": "2024-02-02T10:00:00Z",
    "actor": "QC Analyst - Michael Park, MS",
    "metadata": {
      "lot_number": "TEVA-LIS-2024-L001",
      "tests_performed": [
        {
          "test": "HPLC Assay",
          "method": "USP-HPLC-001",
          "result": "99.85%",
          "specification": "98.0-102.0%",
          "status": "PASS"
        },
        {
          "test": "Karl Fischer (Moisture)",
          "method": "USP-KF-001",
          "result": "0.09%",
          "specification": "≤0.5%",
          "status": "PASS"
        },
        {
          "test": "ICP-MS (Heavy Metals)",
          "method": "USP-ICP-001",
          "result": "7 ppm",
          "specification": "≤20 ppm",
          "status": "PASS"
        },
        {
          "test": "Endotoxin (LAL)",
          "method": "USP-LAL-001",
          "result": "<0.25 EU/mg",
          "specification": "≤0.5 EU/mg",
          "status": "PASS"
        }
      ],
      "analyst": "Michael Park",
      "reviewer": "Dr. Patricia Chen",
      "release_status": "APPROVED",
      "release_date": "2024-02-02",
      "capa_required": false
    }
  }')
echo "✅ QC testing complete: ALL TESTS PASSED"
echo "   Status: RELEASED FOR MANUFACTURING"
echo "   Released by: Dr. Patricia Chen, QA Director"
echo ""

sleep 1

# Step 3: Manufacturing (Batch Production)
echo "⚙️  Step 3: Batch Manufacturing"
echo "----------------------------------------"

echo "Starting batch production (Lisinopril 10mg tablets)..."
MANUFACTURING=$(curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "BATCH_MANUFACTURING",
    "product_id": "lisinopril-10mg-batch-2024-MC001",
    "location": "MediCure - Manufacturing Suite 3 (Class 100K)",
    "timestamp": "2024-02-05T07:00:00Z",
    "actor": "Production Supervisor - James Wilson",
    "metadata": {
      "batch_number": "MC-LIS10-2024-001",
      "product": "Lisinopril Tablets 10mg",
      "batch_size": 500000,
      "api_lot_used": "TEVA-LIS-2024-L001",
      "api_qty_used_kg": 5,
      "excipients": {
        "microcrystalline_cellulose": "50 kg (Lot: MCC-2024-045)",
        "magnesium_stearate": "2 kg (Lot: MGS-2024-012)",
        "croscarmellose_sodium": "10 kg (Lot: CCS-2024-089)",
        "mannitol": "30 kg (Lot: MAN-2024-023)"
      },
      "manufacturing_steps": [
        {
          "step": "Weighing & Dispensing",
          "operator": "Sarah Martinez",
          "timestamp": "2024-02-05T07:00:00Z",
          "status": "complete"
        },
        {
          "step": "Blending",
          "operator": "David Kim",
          "equipment": "V-Blender #3",
          "blend_time_min": 20,
          "timestamp": "2024-02-05T09:00:00Z",
          "status": "complete"
        },
        {
          "step": "Compression",
          "operator": "Lisa Chen",
          "equipment": "Tablet Press #7",
          "timestamp": "2024-02-05T11:00:00Z",
          "tablets_per_hour": 125000,
          "status": "complete"
        },
        {
          "step": "Coating",
          "operator": "Robert Lee",
          "equipment": "Coating Pan #2",
          "coating_material": "HPMC (Lot: HPMC-2024-011)",
          "timestamp": "2024-02-05T15:00:00Z",
          "status": "complete"
        }
      ],
      "environmental_monitoring": {
        "room_temp_c": 22,
        "humidity_percent": 45,
        "particulate_count": 95000,
        "viable_count_cfu": 2
      },
      "yield_percent": 98.5,
      "status": "awaiting_qc"
    }
  }')
echo "✅ Batch manufacturing complete: MC-LIS10-2024-001"
echo "   Product: Lisinopril 10mg Tablets"
echo "   Quantity: 500,000 tablets (98.5% yield)"
echo "   Status: Awaiting final QC release"
echo ""

sleep 1

# Step 4: Final QC Release
echo "✅ Step 4: Final Product QC Release"
echo "----------------------------------------"

echo "Running finished product testing..."
FINAL_QC=$(curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "FINAL_QC_RELEASE",
    "product_id": "lisinopril-10mg-batch-2024-MC001",
    "location": "MediCure - QC Laboratory",
    "timestamp": "2024-02-06T14:00:00Z",
    "actor": "QC Manager - Dr. Patricia Chen, PhD",
    "metadata": {
      "batch_number": "MC-LIS10-2024-001",
      "tests_performed": [
        {"test": "Assay (HPLC)", "result": "10.2 mg", "spec": "9.5-10.5 mg", "status": "PASS"},
        {"test": "Dissolution", "result": "98% @ 30min", "spec": "≥80% @ 30min", "status": "PASS"},
        {"test": "Weight Variation", "result": "±2.1%", "spec": "±5%", "status": "PASS"},
        {"test": "Hardness", "result": "8.5 kP", "spec": "6-12 kP", "status": "PASS"},
        {"test": "Friability", "result": "0.3%", "spec": "≤1.0%", "status": "PASS"},
        {"test": "Microbial Limits", "result": "<10 CFU/g", "spec": "<100 CFU/g", "status": "PASS"}
      ],
      "stability_protocol": "40C/75%RH (ongoing)",
      "expiration_date": "2027-02-06",
      "release_status": "APPROVED_FOR_DISTRIBUTION",
      "ndc_number": "12345-678-90",
      "approved_by": "Dr. Patricia Chen",
      "approval_signature": "sha256:8f3e9a7b..."
    }
  }')
echo "✅ Final QC release: APPROVED FOR DISTRIBUTION"
echo "   Expiration: 2027-02-06 (3 years)"
echo "   NDC: 12345-678-90"
echo "   Approved by: Dr. Patricia Chen (digitally signed)"
echo ""

# Step 5: Distribution to Pharmacy Chain
echo "🚚 Step 5: Distribution"
echo "----------------------------------------"

echo "Creating wholesale customer (CVS Pharmacy)..."
CUSTOMER=$(curl -s -X POST "$FIN_BASE/customers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "CVS Pharmacy - Distribution Center",
    "email": "pharmaceuticals@cvshealth.com",
    "type": "pharmaceutical_wholesaler",
    "payment_terms_days": 60
  }')
CUSTOMER_ID=$(echo "$CUSTOMER" | jq -r '.id')
echo "✅ Customer: CVS Pharmacy ($CUSTOMER_ID)"
echo ""

sleep 1

echo "Recording shipment to CVS..."
SHIPMENT=$(curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "DISTRIBUTION",
    "product_id": "lisinopril-10mg-batch-2024-MC001",
    "location": "In Transit to CVS DC - Memphis, TN",
    "timestamp": "2024-02-07T09:00:00Z",
    "actor": "MediCure Logistics",
    "metadata": {
      "batch_number": "MC-LIS10-2024-001",
      "quantity_bottles": 5000,
      "tablets_per_bottle": 100,
      "total_tablets": 500000,
      "shipping_temp_c": 23,
      "carrier": "FedEx Healthcare",
      "tracking": "FX-2024-PHM-890123",
      "pedigree_attached": true,
      "customer": "CVS Pharmacy",
      "po_number": "CVS-PO-2024-078901"
    }
  }')
echo "✅ Shipment to CVS: 5,000 bottles (500,000 tablets)"
echo "   Tracking: FX-2024-PHM-890123"
echo "   Pedigree attached for state compliance"
echo ""

# Step 6: Create Invoice
echo "💰 Step 6: Wholesale Invoice"
echo "----------------------------------------"

INVOICE=$(curl -s -X POST "$FIN_BASE/invoices" \
  -H "Content-Type: application/json" \
  -d "{
    \"customer_id\": \"$CUSTOMER_ID\",
    \"items\": [
      {
        \"description\": \"Lisinopril 10mg Tablets - 100/bottle (5000 bottles)\",
        \"quantity\": 1,
        \"unit_price\": \"125000.00\",
        \"account_code\": \"4000\"
      }
    ],
    \"due_date\": \"2024-04-07\",
    \"notes\": \"Batch MC-LIS10-2024-001. NDC 12345-678-90. Net 60 terms. Pedigree attached.\"
  }")
INVOICE_ID=$(echo "$INVOICE" | jq -r '.id')
INVOICE_TOTAL=$(echo "$INVOICE" | jq -r '.total')
echo "✅ Invoice: $INVOICE_ID"
echo "   Amount: $$INVOICE_TOTAL"
echo "   Unit price: \$25.00 per bottle (wholesale)"
echo ""

# Step 7: Create API Cost (COGS)
echo "📝 Step 7: Recording API Cost"
echo "----------------------------------------"

API_BILL=$(curl -s -X POST "$FIN_BASE/bills" \
  -H "Content-Type: application/json" \
  -d "{
    \"vendor_id\": \"$API_SUPPLIER_ID\",
    \"items\": [
      {
        \"description\": \"Lisinopril API (100 kg @ \$450/kg)\",
        \"quantity\": 1,
        \"unit_price\": \"45000.00\",
        \"account_code\": \"5300\"
      }
    ],
    \"due_date\": \"2024-04-02\",
    \"notes\": \"Lot TEVA-LIS-2024-L001. Used 5 kg for batch MC-LIS10-2024-001\"
  }")
API_COST=$(echo "$API_BILL" | jq -r '.total')
echo "✅ API cost recorded: $$API_COST"
echo "   Cost per kg: \$450"
echo "   Amount used: 5 kg = \$2,250 COGS for this batch"
echo ""

# Step 8: Regulatory Audit Trail
echo "📋 Step 8: Regulatory Audit Trail (FDA Ready)"
echo "----------------------------------------"

echo "Retrieving complete batch history (21 CFR Part 11 compliant)..."
AUDIT_TRAIL=$(curl -s "$API_BASE/provenance/lisinopril-10mg-batch-2024-MC001")

echo ""
echo "BATCH RECORD: MC-LIS10-2024-001"
echo "========================================"
echo "$AUDIT_TRAIL" | jq -r '
  .events[] |
  "[\(.timestamp)] \(.event_type) - \(.actor)\n  Location: \(.location)\n  Status: \(.metadata.release_status // .metadata.status // "N/A")\n"
'

echo "Cryptographic Verification:"
echo "$AUDIT_TRAIL" | jq -r '.events[] | "  \(.event_type): \(.signature[0:16])..."'
echo ""

# Step 9: Profitability Analysis
echo "💹 Step 9: Batch Profitability"
echo "----------------------------------------"

COGS_BATCH=$(echo "2250 + 5000" | bc)  # API + excipients
GROSS_PROFIT=$(echo "$INVOICE_TOTAL - $COGS_BATCH" | bc)
MARGIN=$(echo "scale=1; ($GROSS_PROFIT / $INVOICE_TOTAL) * 100" | bc)

echo "Batch Profitability Analysis:"
echo "  Revenue (wholesale): $$INVOICE_TOTAL"
echo "  COGS:"
echo "    - API: \$2,250 (5 kg Lisinopril)"
echo "    - Excipients: \$5,000 (estimated)"
echo "    - Total COGS: $$COGS_BATCH"
echo "  Gross Profit: $$GROSS_PROFIT"
echo "  Gross Margin: ${MARGIN}%"
echo ""
echo "  Per-Tablet Economics:"
echo "    - Cost: \$$(echo "scale=4; $COGS_BATCH / 500000" | bc) per tablet"
echo "    - Wholesale price: \$0.25 per tablet"
echo "    - Retail (typical): \$0.50-1.00 per tablet"
echo ""

# Summary
echo "✅ Demo Complete!"
echo "================="
echo ""
echo "Summary of Operations:"
echo "  🧪 API received with full chain of custody (100 kg Lisinopril)"
echo "  🔬 QC testing: 4/4 tests PASSED (HPLC, KF, ICP-MS, LAL)"
echo "  ⚙️  Manufacturing: 500,000 tablets produced (98.5% yield)"
echo "  ✅ Final QC: 6/6 tests PASSED → RELEASED"
echo "  🚚 Distribution: 5,000 bottles shipped to CVS"
echo "  💰 Invoice: $$INVOICE_TOTAL (Net 60)"
echo "  📊 Profitability: ${MARGIN}% gross margin"
echo "  📋 Audit trail: 100% compliant with 21 CFR Part 11"
echo ""
echo "Regulatory Compliance:"
echo "  ✅ Complete chain of custody (supplier → manufacturing → distribution)"
echo "  ✅ Every event cryptographically signed (tamper-proof)"
echo "  ✅ QC release at every stage (quarantine → approved)"
echo "  ✅ Environmental monitoring (cleanroom standards)"
echo "  ✅ Batch genealogy (can trace to exact API lot)"
echo "  ✅ 24-hour recall capability (if needed)"
echo ""
echo "💊 Mycelix ERP: Pharmaceutical Manufacturing Excellence!"
echo "   FDA Audit-Ready | 21 CFR Part 11 Compliant | Blockchain-Verified"
