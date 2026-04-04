#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Cluster USACE dam sites by US state and combine with SMR nuclear projects
to produce a pre-clustered energy sites JSON for the WebGL globe.

Output: assets/data/sites-clustered.json
"""

import json
import os
import sys

# US state FIPS codes to full names
STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico", "VI": "Virgin Islands", "GU": "Guam",
    "AS": "American Samoa", "MP": "Northern Mariana Islands",
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "/srv/luminous-dynamics/terra-atlas-mvp/data"
OUT_FILE = os.path.join(BASE_DIR, "assets", "data", "sites-clustered.json")


def load_smr_projects():
    """Load and transform SMR nuclear projects."""
    with open(os.path.join(DATA_DIR, "smr-pipeline.json")) as f:
        data = json.load(f)

    results = []
    for p in data["projects"]:
        loc = p["location"]
        # Map NRC statuses to simpler categories
        status = p.get("status", "unknown")
        if "Potential" in status:
            simple_status = "planned"
        elif "Planning" in status or "Letter" in status or "Agreement" in status:
            simple_status = "planned"
        elif "Application" in status or "Certification" in status or "Licensing" in status:
            simple_status = "licensing"
        elif "Construction" in status or "Preparation" in status:
            simple_status = "construction"
        elif "Approved" in status or "Issued" in status:
            simple_status = "approved"
        elif "Restart" in status:
            simple_status = "restart"
        else:
            simple_status = "planned"

        results.append({
            "id": p["id"],
            "name": p["name"],
            "lat": round(loc["latitude"], 4),
            "lon": round(loc["longitude"], 4),
            "energy_type": "nuclear",
            "capacity_mw": p["capacityMw"],
            "status": simple_status,
            "country": "US",
        })

    return results


def load_and_cluster_dams():
    """Load USACE dams and cluster by US state."""
    print("Loading USACE dams...", file=sys.stderr)
    with open(os.path.join(DATA_DIR, "usace-dams.json")) as f:
        data = json.load(f)

    projects = data["projects"]
    print(f"  Loaded {len(projects)} dam sites", file=sys.stderr)

    # Cluster by state
    clusters = {}  # state_code -> {lats, lons, capacities, count}
    no_state = []

    for p in projects:
        loc = p.get("location", {})
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        cap = p.get("capacityMw", 0) or 0
        state = loc.get("state", "")

        # Skip entries without coordinates
        if lat is None or lon is None:
            continue

        if state and len(state) == 2:
            if state not in clusters:
                clusters[state] = {"lats": [], "lons": [], "cap": 0.0, "count": 0}
            clusters[state]["lats"].append(lat)
            clusters[state]["lons"].append(lon)
            clusters[state]["cap"] += cap
            clusters[state]["count"] += 1
        else:
            no_state.append((lat, lon, cap))

    # Build cluster entries
    results = []
    for state_code, c in sorted(clusters.items()):
        centroid_lat = sum(c["lats"]) / len(c["lats"])
        centroid_lon = sum(c["lons"]) / len(c["lons"])
        state_name = STATE_NAMES.get(state_code, state_code)
        cap_mw = round(c["cap"], 1)
        count = c["count"]

        results.append({
            "id": f"hydro-{state_code.lower()}",
            "name": f"{state_name} Hydro ({count:,} dams)",
            "lat": round(centroid_lat, 4),
            "lon": round(centroid_lon, 4),
            "energy_type": "hydro",
            "capacity_mw": cap_mw,
            "count": count,
            "status": "operational",
            "country": "US",
        })

    # Handle dams without a state via grid clustering (5-degree cells)
    if no_state:
        print(f"  {len(no_state)} dams without state, grid-clustering...", file=sys.stderr)
        grid = {}
        for lat, lon, cap in no_state:
            cell = (int(lat // 5) * 5, int(lon // 5) * 5)
            if cell not in grid:
                grid[cell] = {"lats": [], "lons": [], "cap": 0.0, "count": 0}
            grid[cell]["lats"].append(lat)
            grid[cell]["lons"].append(lon)
            grid[cell]["cap"] += cap
            grid[cell]["count"] += 1

        for (glat, glon), c in sorted(grid.items()):
            centroid_lat = sum(c["lats"]) / len(c["lats"])
            centroid_lon = sum(c["lons"]) / len(c["lons"])
            results.append({
                "id": f"hydro-grid-{glat}-{glon}",
                "name": f"Hydro Grid ({glat}N, {abs(glon)}W, {c['count']} dams)",
                "lat": round(centroid_lat, 4),
                "lon": round(centroid_lon, 4),
                "energy_type": "hydro",
                "capacity_mw": round(c["cap"], 1),
                "count": c["count"],
                "status": "operational",
                "country": "US",
            })

    print(f"  Created {len(results)} hydro clusters", file=sys.stderr)
    return results


def main():
    smr = load_smr_projects()
    print(f"Loaded {len(smr)} SMR nuclear projects", file=sys.stderr)

    hydro = load_and_cluster_dams()

    combined = hydro + smr
    print(f"Total entries: {len(combined)} ({len(hydro)} hydro + {len(smr)} nuclear)", file=sys.stderr)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    with open(OUT_FILE, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Written to {OUT_FILE}", file=sys.stderr)
    print(f"File size: {os.path.getsize(OUT_FILE) / 1024:.1f} KB", file=sys.stderr)


if __name__ == "__main__":
    main()
