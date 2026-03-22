# Terra Atlas Data Import Scripts

This directory contains scripts for importing real-world energy project data into the Terra Atlas platform.

## USACE Dam Data Import

Imports the **USACE National Inventory of Dams** dataset (~87,000 US dams) into the platform.

### Prerequisites

1. **Install dependencies**:
   ```bash
   npm install csv-parse node-fetch
   npm install --save-dev @types/node ts-node
   ```

2. **Set environment variables**:
   ```bash
   export NEXT_PUBLIC_SUPABASE_URL="your-supabase-url"
   export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
   ```

   Or create a `.env.local` file:
   ```
   NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   ```

3. **Run the migration**:
   First, apply the sites table migration in Supabase SQL Editor:
   ```sql
   -- Run the contents of: supabase/migrations/003_create_sites_table.sql
   ```

### Running the Import

```bash
# Run the import script
npx ts-node scripts/import-usace-dams.ts
```

The script will:
1. Download the USACE dataset (~50MB)
2. Cache it locally in `data/usace-dams.csv`
3. Parse and convert dam records to site format
4. Insert sites in batches of 1000
5. Refresh the materialized view for globe data

### Expected Output

```
🚀 Starting USACE Dam Data Import
==================================================
✅ Using cached USACE data (age: 0 days)
📊 Parsed 87,000 dam records
🔄 Converting dam records to site format...
📊 Converted 12,543 dams with hydroelectric capacity
📤 Inserting 12,543 sites into database...
✅ Progress: 100.0% (12543 inserted, 0 skipped, 0 errors)

✅ Import complete:
   - Inserted: 12,543
   - Skipped: 0
   - Errors: 0
🔄 Refreshing materialized view...
✅ Materialized view refreshed

🎉 Import completed successfully!
```

### Data Quality Notes

- **Capacity Estimation**: The USACE database doesn't include power capacity (MW), so we estimate it based on dam height and storage volume. This is a rough approximation.
- **Hydroelectric Only**: We filter for dams with hydroelectric purposes (about 14% of all dams)
- **Location Validated**: Only includes dams with valid US coordinates
- **Data Freshness**: USACE data is cached for 7 days, then auto-refreshed

### Performance

- **Download**: ~30-60 seconds (first run)
- **Processing**: ~10-20 seconds
- **Database Insert**: ~2-5 minutes for 12K records
- **Total Time**: ~3-6 minutes

### Troubleshooting

**Error: ECONNRESET or timeout downloading data**
- The USACE API can be slow. The script will retry with cached data if available.
- Try running again - the data will be cached after first successful download.

**Error: Permission denied (RLS policy)**
- Make sure you're using the `SUPABASE_SERVICE_ROLE_KEY`, not the anon key
- The service role key has full access and bypasses RLS

**Error: duplicate key value violates unique constraint**
- The script doesn't handle duplicates yet. To re-import:
  ```sql
  -- In Supabase SQL Editor:
  DELETE FROM sites WHERE data_source = 'usace';
  ```

## Future Import Scripts

### Global Solar Atlas
```bash
# Coming soon
npx ts-node scripts/import-solar-atlas.ts
```

### US Wind Turbine Database
```bash
# Coming soon
npx ts-node scripts/import-wind-turbines.ts
```

### IAEA Nuclear Facilities
```bash
# Coming soon
npx ts-node scripts/import-nuclear-facilities.ts
```

## Data Sources

| Dataset | Records | Coverage | Update Frequency |
|---------|---------|----------|------------------|
| **USACE National Inventory of Dams** | 87,000 | US Only | Monthly |
| Global Solar Atlas | ~50,000 | Global | Annual |
| US Wind Turbine Database (USWTDB) | ~70,000 | US Only | Quarterly |
| IAEA Power Reactor Information System | ~440 | Global | Real-time |
| Global Energy Observatory | ~30,000 | Global | Irregular |

## Contributing

To add a new data import script:

1. Create a new file: `scripts/import-[source].ts`
2. Follow the pattern in `import-usace-dams.ts`
3. Map source data to the `sites` table schema
4. Include data quality indicators
5. Add progress logging and error handling
6. Update this README

## License

All imported data should be properly attributed and respect the original source's licensing terms.
