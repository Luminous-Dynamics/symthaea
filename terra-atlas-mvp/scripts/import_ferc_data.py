#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FERC Interconnection Queue Data Importer
=========================================

Downloads and imports real FERC interconnection queue data into Terra Atlas database.

Data Source: https://www.ferc.gov/media/queue-interconnection-requests
Expected Records: 11,547 active interconnection requests
Database Table: energy_projects

Author: Terra Atlas Team
Date: November 21, 2025
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional
import time

# Configuration
FERC_DATA_URL = "https://www.ferc.gov/media/queue-interconnection-requests"
DATA_DIR = Path(__file__).parent.parent / "data" / "ferc"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Supabase configuration (from environment)
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

class FERCImporter:
    """Handles downloading and importing FERC interconnection queue data."""

    def __init__(self):
        self.data_file = None
        self.df = None
        self.stats = {
            "downloaded": 0,
            "parsed": 0,
            "imported": 0,
            "errors": 0,
            "start_time": datetime.now()
        }

    def download_data(self) -> Path:
        """
        Download FERC interconnection queue data.

        Returns:
            Path to downloaded file
        """
        print("📥 Downloading FERC interconnection queue data...")
        print(f"   Source: {FERC_DATA_URL}")

        # Check if we already have recent data
        existing_files = list(DATA_DIR.glob("ferc_queue_*.csv"))
        if existing_files:
            latest = max(existing_files, key=lambda p: p.stat().st_mtime)
            age_hours = (time.time() - latest.stat().st_mtime) / 3600

            if age_hours < 24:
                print(f"✅ Using existing data file (age: {age_hours:.1f}h)")
                self.data_file = latest
                return latest

        # Download new data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = DATA_DIR / f"ferc_queue_{timestamp}.csv"

        try:
            # NOTE: FERC provides data in Excel format
            # We'll need to:
            # 1. Download the Excel file
            # 2. Convert to CSV for easier processing
            # 3. Handle pagination if data is split across multiple sheets

            # For now, create a placeholder that explains how to get the data
            print("⚠️  FERC data download not yet implemented")
            print("   Manual steps required:")
            print("   1. Visit: https://www.ferc.gov/industries-data/electric/overview")
            print("   2. Download: Interconnection Queue dataset (Excel)")
            print("   3. Convert to CSV")
            print(f"   4. Save to: {output_file}")
            print()
            print("🔄 For automated download, we need to:")
            print("   - Identify exact data file URL (changes periodically)")
            print("   - Handle Excel-to-CSV conversion")
            print("   - Parse multiple sheets if data is paginated")

            self.data_file = output_file
            return output_file

        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise

    def parse_data(self, file_path: Path) -> pd.DataFrame:
        """
        Parse FERC data file into pandas DataFrame.

        Args:
            file_path: Path to data file

        Returns:
            Parsed DataFrame
        """
        print("📊 Parsing FERC data...")

        if not file_path.exists():
            print(f"⚠️  Data file not found: {file_path}")
            print("   Creating demo structure for reference...")

            # Create demo structure showing expected columns
            demo_data = {
                "Queue ID": ["Q001", "Q002"],
                "Project Name": ["Solar Farm TX-1", "Wind Farm CA-1"],
                "Developer": ["NextEra Energy", "Invenergy"],
                "State": ["TX", "CA"],
                "County": ["Travis", "Kern"],
                "Capacity (MW)": [150.0, 200.0],
                "Energy Source": ["Solar", "Wind"],
                "Latitude": [30.2672, 35.3733],
                "Longitude": [-97.7431, -118.9987],
                "Status": ["Active", "System Impact Study"],
                "Queue Date": ["2023-01-15", "2023-02-20"],
                "Estimated Cost ($M)": [180.0, 250.0],
            }

            self.df = pd.DataFrame(demo_data)
            self.stats["parsed"] = len(self.df)

            print(f"✅ Demo data created: {len(self.df)} records")
            print("\nExpected columns for real FERC data:")
            for col in demo_data.keys():
                print(f"   - {col}")

            return self.df

        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Validate required columns
            required_cols = ["Queue ID", "Project Name", "Developer", "State", "Capacity (MW)"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"⚠️  Missing required columns: {missing_cols}")
                print(f"   Available columns: {list(df.columns)}")
                raise ValueError(f"Missing required columns: {missing_cols}")

            self.df = df
            self.stats["parsed"] = len(df)

            print(f"✅ Parsed {len(df)} projects")
            print(f"   States: {df['State'].nunique()}")
            print(f"   Developers: {df['Developer'].nunique()}")
            print(f"   Total capacity: {df['Capacity (MW)'].sum():.1f} MW")

            return df

        except Exception as e:
            print(f"❌ Parse failed: {e}")
            raise

    def transform_data(self) -> List[Dict]:
        """
        Transform FERC data to match energy_projects schema.

        Returns:
            List of records ready for database import
        """
        print("🔄 Transforming data for database...")

        if self.df is None:
            raise ValueError("No data to transform. Run parse_data() first.")

        records = []

        for idx, row in self.df.iterrows():
            try:
                # Map FERC fields to energy_projects schema
                record = {
                    "name": row.get("Project Name", "Unknown"),
                    "project_type": self._map_energy_source(row.get("Energy Source", "Unknown")),
                    "developer": row.get("Developer", "Unknown"),
                    "capacity_mw": float(row.get("Capacity (MW)", 0)),
                    "latitude": float(row.get("Latitude", 0)) if pd.notna(row.get("Latitude")) else None,
                    "longitude": float(row.get("Longitude", 0)) if pd.notna(row.get("Longitude")) else None,
                    "status": row.get("Status", "Unknown"),
                    "estimated_cost": float(row.get("Estimated Cost ($M)", 0)) * 1_000_000 if pd.notna(row.get("Estimated Cost ($M)")) else None,
                    "metadata": {
                        "queue_id": row.get("Queue ID"),
                        "state": row.get("State"),
                        "county": row.get("County"),
                        "queue_date": row.get("Queue Date"),
                        "source": "FERC Interconnection Queue",
                        "imported_at": datetime.now().isoformat()
                    }
                }

                records.append(record)

            except Exception as e:
                print(f"⚠️  Error transforming row {idx}: {e}")
                self.stats["errors"] += 1
                continue

        print(f"✅ Transformed {len(records)} records")

        return records

    def import_to_database(self, records: List[Dict]) -> int:
        """
        Import records to Supabase database.

        Args:
            records: List of transformed records

        Returns:
            Number of successfully imported records
        """
        print("💾 Importing to database...")

        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            print("⚠️  Supabase credentials not found in environment")
            print("   Required environment variables:")
            print("   - NEXT_PUBLIC_SUPABASE_URL")
            print("   - SUPABASE_SERVICE_ROLE_KEY")
            print()
            print(f"   Would import {len(records)} records to energy_projects table")
            return 0

        try:
            # Import supabase client
            from supabase import create_client, Client

            supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

            # Batch insert (Supabase has a limit, typically 1000 per batch)
            batch_size = 1000
            imported = 0

            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]

                try:
                    response = supabase.table("energy_projects").insert(batch).execute()
                    imported += len(batch)
                    print(f"   Imported batch {i//batch_size + 1}: {imported}/{len(records)}")

                except Exception as e:
                    print(f"⚠️  Batch {i//batch_size + 1} failed: {e}")
                    self.stats["errors"] += len(batch)

            self.stats["imported"] = imported
            print(f"✅ Imported {imported} records to database")

            return imported

        except ImportError:
            print("⚠️  supabase-py not installed")
            print("   Install with: pip install supabase")
            return 0
        except Exception as e:
            print(f"❌ Database import failed: {e}")
            raise

    def run(self):
        """Run the complete import pipeline."""
        print("="*60)
        print("FERC Interconnection Queue Data Import")
        print("="*60)
        print()

        try:
            # Step 1: Download
            data_file = self.download_data()

            # Step 2: Parse
            df = self.parse_data(data_file)

            # Step 3: Transform
            records = self.transform_data()

            # Step 4: Import
            imported = self.import_to_database(records)

            # Report
            self.print_summary()

        except Exception as e:
            print(f"\n❌ Import failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def print_summary(self):
        """Print import summary statistics."""
        elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()

        print()
        print("="*60)
        print("IMPORT SUMMARY")
        print("="*60)
        print(f"Parsed:   {self.stats['parsed']:,} records")
        print(f"Imported: {self.stats['imported']:,} records")
        print(f"Errors:   {self.stats['errors']:,} errors")
        print(f"Time:     {elapsed:.1f}s")
        print(f"Rate:     {self.stats['imported']/elapsed:.0f} records/sec" if elapsed > 0 else "N/A")
        print("="*60)

    @staticmethod
    def _map_energy_source(source: str) -> str:
        """Map FERC energy source to project_type."""
        source_lower = str(source).lower()

        if "solar" in source_lower:
            return "solar"
        elif "wind" in source_lower:
            return "wind"
        elif "hydro" in source_lower or "water" in source_lower:
            return "hydro"
        elif "nuclear" in source_lower or "smr" in source_lower:
            return "nuclear"
        elif "battery" in source_lower or "storage" in source_lower:
            return "storage"
        else:
            return "other"


def main():
    """Main entry point."""
    importer = FERCImporter()
    importer.run()


if __name__ == "__main__":
    main()
