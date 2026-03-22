# FERC Data Import Guide

## Overview

This script imports real FERC interconnection queue data into the Terra Atlas database.

**Expected Result**: 11,547 energy projects imported from the FERC interconnection queue.

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements-data-import.txt
```

### 2. Set Environment Variables

Required:
- `NEXT_PUBLIC_SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Service role key (has full database access)

```bash
# Option 1: Export variables
export NEXT_PUBLIC_SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"

# Option 2: Use .env.local (already configured)
# The script will read from Next.js environment
```

### 3. Get FERC Data

**Option A: Manual Download** (Required for first run)

1. Visit [FERC Electric Power Markets](https://www.ferc.gov/industries-data/electric/overview)
2. Download the "Interconnection Queue" dataset (Excel format)
3. Convert to CSV using Excel or LibreOffice
4. Save as `data/ferc/ferc_queue_YYYYMMDD_HHMMSS.csv`

**Option B: Automated Download** (TODO)

Future enhancement will auto-download from FERC API.

## Usage

### Run Import

```bash
python3 scripts/import_ferc_data.py
```

### Output

```
============================================================
FERC Interconnection Queue Data Import
============================================================

📥 Downloading FERC interconnection queue data...
📊 Parsing FERC data...
🔄 Transforming data for database...
💾 Importing to database...
   Imported batch 1: 1000/11547
   Imported batch 2: 2000/11547
   ...
✅ Imported 11,547 records to database

============================================================
IMPORT SUMMARY
============================================================
Parsed:   11,547 records
Imported: 11,547 records
Errors:   0 errors
Time:     45.2s
Rate:     255 records/sec
============================================================
```

## Data Mapping

FERC fields → `energy_projects` table:

| FERC Field | Database Field | Transform |
|------------|----------------|-----------|
| Queue ID | metadata.queue_id | Direct |
| Project Name | name | Direct |
| Developer | developer | Direct |
| State | metadata.state | Direct |
| County | metadata.county | Direct |
| Capacity (MW) | capacity_mw | Float |
| Energy Source | project_type | Mapped (see below) |
| Latitude | latitude | Float |
| Longitude | longitude | Float |
| Status | status | Direct |
| Queue Date | metadata.queue_date | Direct |
| Estimated Cost | estimated_cost | Float (M → actual) |

### Energy Source Mapping

- "Solar" → `solar`
- "Wind" / "Offshore Wind" → `wind`
- "Hydro" → `hydro`
- "Nuclear" / "SMR" → `nuclear`
- "Battery Storage" → `storage`
- Other → `other`

## Troubleshooting

### ModuleNotFoundError: pandas

**Solution**: Install dependencies
```bash
pip install -r requirements-data-import.txt
```

### Supabase credentials not found

**Solution**: Set environment variables or check `.env.local`

### No data file found

**Solution**: Download FERC data manually (see Setup step 3)

### Import fails mid-way

**Solution**: Script will report which batch failed. Check database logs and re-run.

## Next Steps

After successful import:

1. Verify data: `SELECT COUNT(*) FROM energy_projects;`
2. Check distribution: `SELECT project_type, COUNT(*) FROM energy_projects GROUP BY project_type;`
3. Test API: `curl http://localhost:3000/api/projects`
4. Update dashboard stats

## Development

To modify the import script:

1. Edit `scripts/import_ferc_data.py`
2. Test with small dataset first
3. Run full import
4. Verify database state

## Status

- ✅ Script foundation complete
- ⏳ Awaiting manual data download
- ⏳ Ready for Day 2 full import

## Resources

- [FERC Electric Power Markets](https://www.ferc.gov/industries-data/electric/overview)
- [Supabase Python Client](https://github.com/supabase-community/supabase-py)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
