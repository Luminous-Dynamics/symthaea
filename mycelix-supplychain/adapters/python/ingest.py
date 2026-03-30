#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Python Adapter for Mycelix Supply Chain

Ingest supply chain events from various data sources (CSV, Excel, ERP exports)
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv


class SupplyChainClient:
    """Client for Mycelix Supply Chain API"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def ingest_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a supply chain event"""
        response = self.session.post(f"{self.base_url}/v1/events", json=event)
        response.raise_for_status()
        return response.json()


def csv_to_events(csv_path: str) -> List[Dict[str, Any]]:
    """Convert CSV to supply chain events"""
    df = pd.read_csv(csv_path)

    events = []
    for _, row in df.iterrows():
        event = {
            "@context": [
                "https://www.w3.org/2018/credentials/v1",
                "https://mycelix.org/contexts/supply-chain/v1",
            ],
            "type": ["VerifiableCredential", "SupplyChainEvent"],
            "issuer": row["issuer"],
            "issuanceDate": datetime.utcnow().isoformat() + "Z",
            "credentialSubject": {
                "eventType": row["eventType"],
                "productId": row["productId"],
                "batchId": row["batchId"],
                "quantity": float(row["quantity"]),
                "unit": row["unit"],
                "facility": {
                    "id": row["facilityId"],
                    "name": row["facilityName"],
                },
                "timestamp": row.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            },
        }

        # Optional fields
        if "prevBatchIds" in row and pd.notna(row["prevBatchIds"]):
            event["credentialSubject"]["prevBatchIds"] = [
                b.strip() for b in row["prevBatchIds"].split(",")
            ]

        events.append(event)

    return events


def main():
    parser = argparse.ArgumentParser(description="Ingest supply chain events from CSV")
    parser.add_argument("-f", "--file", required=True, help="CSV file to ingest")
    parser.add_argument(
        "-u", "--url", default="http://localhost:8080", help="API base URL"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse without sending to API"
    )

    args = parser.parse_args()

    load_dotenv()

    client = SupplyChainClient(args.url)

    # Test connection
    if not args.dry_run:
        try:
            health = client.health()
            print(f"Connected to API (version {health['version']})")
        except Exception as e:
            print(f"Failed to connect to API: {e}", file=sys.stderr)
            sys.exit(1)

    # Parse CSV
    try:
        events = csv_to_events(args.file)
        print(f"Parsed {len(events)} events from {args.file}")
    except Exception as e:
        print(f"Failed to parse CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Ingest events
    processed = 0
    errors = 0

    for event in events:
        try:
            if args.dry_run:
                print(
                    f"[DRY RUN] Would ingest: {event['credentialSubject']['eventType']} "
                    f"for batch {event['credentialSubject']['batchId']}"
                )
            else:
                result = client.ingest_event(event)
                print(
                    f"✓ Ingested {event['credentialSubject']['eventType']} "
                    f"for batch {event['credentialSubject']['batchId']} "
                    f"→ claim {result['claim_id']}"
                )
            processed += 1
        except Exception as e:
            print(f"✗ Failed to ingest event: {e}", file=sys.stderr)
            errors += 1

    print(f"\nSummary: {processed} processed, {errors} errors")


if __name__ == "__main__":
    main()
