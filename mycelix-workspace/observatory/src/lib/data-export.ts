/**
 * CSV/JSON data export utilities for resilience kit domains.
 *
 * Uses the same downloadExport() pattern as tax-export.ts.
 */

import type {
  ExchangeRecord,
  HarvestRecord,
  FoodPlot,
  PlantingRecord,
  EmergencyMessage,
  AidOffer,
  AidRequest,
  WaterReading,
  InventoryItem,
} from './resilience-client';

// ============================================================================
// Download Helper
// ============================================================================

function downloadExport(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function escCsv(val: unknown): string {
  const s = String(val ?? '');
  if (s.includes(',') || s.includes('"') || s.includes('\n')) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

export function toCsv(headers: string[], rows: string[][]): string {
  const lines = [headers.join(',')];
  for (const row of rows) {
    lines.push(row.map(escCsv).join(','));
  }
  return lines.join('\n');
}

function datestamp(): string {
  return new Date().toISOString().slice(0, 10);
}

// ============================================================================
// TEND Exports
// ============================================================================

export function exportTendExchangesCsv(exchanges: ExchangeRecord[]): void {
  const headers = ['ID', 'Provider', 'Receiver', 'Hours', 'Service', 'Category', 'Status', 'Date'];
  const rows = exchanges.map(e => [
    e.id,
    e.provider_did,
    e.receiver_did,
    String(e.hours),
    e.service_description,
    e.service_category,
    e.status,
    new Date(e.timestamp).toISOString(),
  ]);
  downloadExport(toCsv(headers, rows), `tend-exchanges-${datestamp()}.csv`, 'text/csv');
}

// ============================================================================
// Food Production Exports
// ============================================================================

export function exportHarvestsCsv(harvests: HarvestRecord[]): void {
  const headers = ['ID', 'Plot ID', 'Crop', 'Quantity (kg)', 'Harvested At', 'Notes'];
  const rows = harvests.map(h => [
    h.id,
    h.plot_id,
    h.crop_name,
    String(h.quantity_kg),
    new Date(h.harvested_at).toISOString(),
    h.notes,
  ]);
  downloadExport(toCsv(headers, rows), `harvests-${datestamp()}.csv`, 'text/csv');
}

export function exportPlotsCsv(plots: FoodPlot[]): void {
  const headers = ['ID', 'Owner', 'Name', 'Location', 'Area (sqm)', 'Type', 'Created'];
  const rows = plots.map(p => [
    p.id,
    p.owner_did,
    p.name,
    p.location,
    String(p.area_sqm),
    p.plot_type,
    new Date(p.created_at).toISOString(),
  ]);
  downloadExport(toCsv(headers, rows), `food-plots-${datestamp()}.csv`, 'text/csv');
}

export function exportPlantingsCsv(plantings: PlantingRecord[]): void {
  const headers = ['ID', 'Plot ID', 'Crop', 'Planted At', 'Expected Harvest', 'Area (sqm)'];
  const rows = plantings.map(p => [
    p.id,
    p.plot_id,
    p.crop_name,
    new Date(p.planted_at).toISOString(),
    new Date(p.expected_harvest).toISOString(),
    String(p.area_sqm),
  ]);
  downloadExport(toCsv(headers, rows), `plantings-${datestamp()}.csv`, 'text/csv');
}

// ============================================================================
// Mutual Aid Exports
// ============================================================================

export function exportAidOffersCsv(offers: AidOffer[]): void {
  const headers = ['Title', 'Description', 'Category', 'Hours Available', 'Provider', 'Recurring', 'Created'];
  const rows = offers.map(o => [
    o.title,
    o.description,
    o.category,
    String(o.hours_available),
    o.provider_did,
    o.recurring ? 'Yes' : 'No',
    new Date(o.created_at).toISOString(),
  ]);
  downloadExport(toCsv(headers, rows), `aid-offers-${datestamp()}.csv`, 'text/csv');
}

export function exportAidRequestsCsv(requests: AidRequest[]): void {
  const headers = ['Title', 'Description', 'Category', 'Hours Needed', 'Urgency', 'Requester', 'Fulfilled', 'Created'];
  const rows = requests.map(r => [
    r.title,
    r.description,
    r.category,
    String(r.hours_needed),
    r.urgency,
    r.requester_did,
    r.fulfilled ? 'Yes' : 'No',
    new Date(r.created_at).toISOString(),
  ]);
  downloadExport(toCsv(headers, rows), `aid-requests-${datestamp()}.csv`, 'text/csv');
}

// ============================================================================
// Emergency Comms Exports
// ============================================================================

export function exportEmergencyMessagesCsv(messages: EmergencyMessage[]): void {
  const headers = ['Sender', 'Content', 'Priority', 'Sent At', 'Synced'];
  const rows = messages.map(m => [
    m.sender_did,
    m.content,
    m.priority,
    new Date(m.sent_at).toISOString(),
    m.synced ? 'Yes' : 'No',
  ]);
  downloadExport(toCsv(headers, rows), `emergency-messages-${datestamp()}.csv`, 'text/csv');
}

// ============================================================================
// Water Exports
// ============================================================================

export function exportWaterReadingsCsv(readings: WaterReading[]): void {
  const headers = ['Source ID', 'Parameter', 'Value', 'Unit', 'Location', 'Recorded At', 'Recorder'];
  const rows = readings.map(r => [
    r.source_id,
    r.parameter,
    String(r.value),
    r.unit,
    r.location || '',
    new Date(r.recorded_at).toISOString(),
    r.recorder_did,
  ]);
  downloadExport(toCsv(headers, rows), `water-readings-${datestamp()}.csv`, 'text/csv');
}

// ============================================================================
// Supplies Exports
// ============================================================================

export function exportInventoryCsv(items: InventoryItem[]): void {
  const headers = ['Name', 'SKU', 'Category', 'Unit', 'Reorder Point', 'Reorder Qty', 'Description'];
  const rows = items.map(i => [
    i.name,
    i.sku,
    i.category,
    i.unit,
    String(i.reorder_point),
    String(i.reorder_quantity),
    i.description || '',
  ]);
  downloadExport(toCsv(headers, rows), `inventory-${datestamp()}.csv`, 'text/csv');
}
