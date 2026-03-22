#!/usr/bin/env python3
"""
Download Interconnection Queue Data using gridstatus library
Downloads data from major ISOs/RTOs in the United States
"""

import sys
import pandas as pd
from pathlib import Path
import gridstatus

def download_interconnection_data():
    """Download interconnection queue data from multiple ISOs"""

    output_dir = Path("data/ferc/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # List of ISOs to download from
    isos = [
        ('CAISO', 'California ISO'),
        ('ERCOT', 'Electric Reliability Council of Texas'),
        ('ISONE', 'ISO New England'),
        ('MISO', 'Midcontinent ISO'),
        ('NYISO', 'New York ISO'),
        ('PJM', 'PJM Interconnection'),
        ('SPP', 'Southwest Power Pool'),
    ]

    all_data = []
    stats = {'total': 0, 'by_iso': {}, 'errors': []}

    print("🔄 Downloading interconnection queue data from all ISOs...")
    print("=" * 70)

    for iso_code, iso_name in isos:
        print(f"\n📡 Fetching {iso_name} ({iso_code})...")

        try:
            # Get the ISO instance
            iso = getattr(gridstatus, iso_code)()

            # Download interconnection queue
            queue_data = iso.get_interconnection_queue()

            if queue_data is not None and len(queue_data) > 0:
                # Add ISO identifier
                queue_data['iso'] = iso_code
                queue_data['iso_name'] = iso_name

                # Save individual ISO file
                iso_file = output_dir / f"{iso_code.lower()}_queue.csv"
                queue_data.to_csv(iso_file, index=False)

                count = len(queue_data)
                stats['by_iso'][iso_code] = count
                stats['total'] += count
                all_data.append(queue_data)

                print(f"  ✅ Downloaded {count:,} projects")
                print(f"  💾 Saved to {iso_file}")
            else:
                print(f"  ⚠️  No data returned")
                stats['errors'].append(f"{iso_code}: No data")

        except Exception as e:
            error_msg = f"{iso_code}: {str(e)}"
            print(f"  ❌ Error: {error_msg}")
            stats['errors'].append(error_msg)
            continue

    # Combine all data
    if all_data:
        print(f"\n📊 Combining data from all ISOs...")
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save combined file
        combined_file = output_dir / "interconnection_queue_combined.csv"
        combined_df.to_csv(combined_file, index=False)

        print(f"✅ Combined dataset: {len(combined_df):,} total projects")
        print(f"💾 Saved to {combined_file}")

        # Print statistics
        print("\n" + "=" * 70)
        print("📈 DOWNLOAD STATISTICS")
        print("=" * 70)
        print(f"Total projects: {stats['total']:,}")
        print(f"\nBy ISO:")
        for iso_code, count in sorted(stats['by_iso'].items(), key=lambda x: x[1], reverse=True):
            iso_name = next(name for code, name in isos if code == iso_code)
            print(f"  {iso_name:40s} {count:>8,}")

        if stats['errors']:
            print(f"\n⚠️  Errors encountered: {len(stats['errors'])}")
            for error in stats['errors']:
                print(f"  - {error}")

        print("\n✅ Download complete!")
        return combined_file
    else:
        print("\n❌ No data downloaded from any ISO")
        return None

if __name__ == "__main__":
    try:
        output_file = download_interconnection_data()
        if output_file:
            print(f"\n🎉 Success! Data ready for import at: {output_file}")
            sys.exit(0)
        else:
            print("\n💥 Failed to download any data")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
