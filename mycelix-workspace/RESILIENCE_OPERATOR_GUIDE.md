# Mycelix Resilience Kit — Operator Training Guide

How to use each feature of the Resilience Kit day-to-day. For installation, see `RESILIENCE_QUICKSTART.md`.

---

## Getting Started

Open the Observatory in your browser:
- **Local**: `http://localhost:5173/resilience`
- **Docker**: `http://localhost/resilience`

The **Resilience Dashboard** (`/resilience`) shows all 11 domains at a glance with live status indicators. Green = connected to DHT, yellow = demo mode.

> **Demo mode**: If no conductor is running, the Observatory uses simulated Roodepoort data. All features work — data just isn't saved. Use demo mode for training before going live.

---

## 1. TEND Mutual Credit (`/tend`)

TEND is time-based mutual credit: 1 TEND = 1 hour of labor. No money changes hands — it's a ledger of who helped whom.

### Record an Exchange

1. Go to `/tend`
2. Click **"+ Record Exchange"**
3. Enter:
   - **Receiver**: The person who received the service (their DID or name)
   - **Hours**: How long the work took (e.g., 2.5)
   - **Description**: What was done (e.g., "Repaired roof guttering")
4. Click **Submit**

The sender's balance decreases, the receiver's increases. Balances can go negative — that's normal for mutual credit.

### Reading Balances

- **Raw Balance**: Total TEND earned minus spent
- **Effective Balance**: After demurrage (anti-hoarding decay). TEND loses ~2% value per month to encourage circulation
- **Oracle Tier**: Shows current credit limits (Normal ±40, Elevated ±60, High ±80, Emergency ±120)

### Service Marketplace

Browse **Listings** (services people offer) and **Requests** (services people need). Match requests to listings to facilitate exchanges.

---

## 2. Food Production (`/food`)

Track what the community grows, where, and how much.

### Register a Plot

1. Go to `/food`
2. Click **"+ Register Plot"**
3. Enter plot name, crop type, area (m²), and location
4. Click **Register**

### Record a Harvest

1. Select a plot from the list
2. Click **"Record Harvest"**
3. Enter yield in kg, date, and any notes
4. Click **Submit**

### Log Nutrient Inputs

Track compost, biochar, mulch, and other inputs going into your plots. Click **"Log Input"** on any plot card.

The **Nutrient Summary** panel (right side) shows totals by resource type across all plots.

---

## 3. Mutual Aid (`/mutual-aid`)

Timebanking: post what you can offer and what you need.

### Post a Service Offer

1. Go to `/mutual-aid`
2. Click **"+ New Offer"**
3. Enter: title, category (Repair, Gardening, Childcare, Transport, Teaching, Other), description, and estimated hours
4. Click **Submit**

### Post a Service Request

1. Click **"+ New Request"**
2. Enter: title, category, urgency (Low/Medium/High/Critical), description
3. Click **Submit**

### Browse and Match

Filter by category or urgency. When someone's offer matches your request, contact them to arrange the exchange, then record it in TEND.

---

## 4. Emergency Communications (`/emergency`)

Priority-based messaging for community alerts.

### Send a Message

1. Go to `/emergency`
2. Select or create a channel (e.g., "Block 7 Water", "Community-wide")
3. Choose priority:
   - **Routine**: General updates
   - **Priority**: Important, non-urgent
   - **Immediate**: Urgent action needed
   - **Flash**: Life-threatening emergency
4. Type your message and click **Send**

Flash and Immediate messages are relayed over the mesh network even when internet is down.

### Create a Channel

Click **"+ Create Channel"** to set up topic-specific channels (e.g., "Night Watch", "Water Status", "Load Shedding").

---

## 5. Value Anchor (`/value-anchor`)

Gives TEND real purchasing power meaning by tracking local prices.

### Set Up the Basket

1. Go to `/value-anchor`
2. Click **"+ Add Item"**
3. Enter a staple item (e.g., "Bread 700g"), its unit, and current TEND price
4. Repeat for key staples: mealie meal, diesel, electricity (kWh), transport (taxi fare)

### Update Prices

When prices change, click the edit icon next to any item and update the TEND price. The basket index recalculates automatically.

### Reading the Dashboard

- **Basket Index**: Weighted average cost of your basket items in TEND. Rising index = inflation.
- **Purchasing Power**: "Your 40 TEND buys: 267 loaves bread, 16L diesel..."
- **Oracle Tier**: When basket volatility exceeds thresholds, TEND credit limits auto-expand

### Export for Tax

Click **"Export Tax Record"** to download a SARS-compatible CSV of barter transactions (Income Tax Act Section 1 compliance).

---

## 6. Water Systems (`/water`)

Track rainwater harvesting, quality readings, and contamination alerts.

### Register a System

1. Go to `/water`
2. Click **"+ Register System"**
3. Enter: name, type (Roof Rainwater, Ground Catchment, Fog Collection, Dew Collection, Snowmelt), capacity in litres, catchment area
4. Click **Register**

### Report a Water Reading

1. Click **"+ Report Reading"**
2. Select the system, parameter (pH, Turbidity, TDS, Chlorine, E. coli), enter the measured value and sample location
3. Click **Submit**

### Alerts

Active contamination alerts appear at the top in red. These are also relayed over the mesh network to nearby nodes.

---

## 7. Household Emergency Planning (`/household`)

Per-hearth emergency preparedness: plans, contacts, shared resources, alerts.

### Select Your Hearth

If you belong to multiple hearths (family units), select the active one from the tabs at the top.

### Create an Emergency Plan

1. Click **"+ Create Plan"**
2. Enter plan name and type (Fire, Flood, Medical, Security, Load Shedding, General)
3. Add emergency contacts: name, phone, relationship, email
4. Click **Save Plan**

### Share a Resource

Click **"+ Share Resource"** to list resources your hearth can share during emergencies (e.g., "Generator — 2kW, available weekdays").

### Raise an Alert

Click **"Raise Alert"** to notify your hearth of an active emergency. Choose severity and describe the situation. Alerts propagate over mesh to nearby nodes.

---

## 8. Community Knowledge (`/knowledge`)

Shared knowledge graph for community claims and local expertise.

### Submit a Claim

1. Go to `/knowledge`
2. Click **"+ Submit Claim"**
3. Enter: title, content (the claim itself), confidence level (0-100%), and tags (comma-separated, e.g., "water, gardening, Roodepoort")
4. Click **Submit**

Claims are stored on the DHT and searchable by the whole community. The **Graph Stats** panel shows total claims, connections, and domains tracked.

---

## 9. Care Circles (`/care-circles`)

Neighbourhood care groups for mutual support.

### Browse Circles

Go to `/care-circles` and browse existing circles. Filter by type: Elder Care, Child Care, Mental Health, Disability, General.

### Join a Circle

Click **"Join"** on any circle card (if capacity allows). Your name is added to the membership list.

Each circle shows: current members, capacity, focus area, and meeting schedule.

---

## 10. Shelter & Housing (`/shelter`)

Track available housing units and emergency shelter.

### Browse Units

Go to `/shelter` to see available housing units with occupancy status, capacity, and amenities.

### Request Placement

1. Click **"Request Placement"** on any available unit
2. Fill in: your name, phone, number of people, and accessibility needs
3. Click **Submit Request**

Requests are queued for the housing coordinator to review.

---

## 11. Supply Chain (`/supplies`)

Community inventory tracking for shared supplies.

### Add an Item

1. Go to `/supplies`
2. Click **"+ Add Item"**
3. Enter: name, category, SKU, unit, reorder point, reorder quantity
4. Click **Add**

### Update Stock

Each item card has an **"Update Stock"** form. Enter the new quantity, location, and optional notes, then click **Update**.

### Low Stock Alerts

Items below their reorder point appear highlighted. The summary panel shows total items, locations, and low-stock count.

---

## Daily Operator Checklist

1. Open `/resilience` — check all domains show green status
2. Review `/emergency` for any Flash/Immediate messages
3. Check `/water` for contamination alerts
4. Glance at `/supplies` for low-stock items
5. Update `/value-anchor` prices if any staples changed this week
6. Check mesh bridge health: `curl http://localhost:9100/health`

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| All pages show "Demo Mode" | Conductor not running. Run `just resilience-up` or check Docker: `docker ps` |
| Exchange fails to submit | Check conductor logs: `just logs` or `docker logs mycelix-conductor` |
| Mesh bridge offline | Check `curl http://localhost:9100/health`. For LoRa: verify SPI device exists (`ls /dev/spidev0.0`) |
| Prices not updating | Value anchor uses localStorage — check you're on the same browser/device |
| Balance seems wrong | Demurrage applies continuously. Raw balance is the ledger total; effective balance includes decay |

## Getting Help

- **Technical**: `just resilience-test` runs the full test suite (76 tests)
- **Status check**: `./scripts/resilience-bootstrap.sh --status`
- **Mycelix**: https://mycelix.net
