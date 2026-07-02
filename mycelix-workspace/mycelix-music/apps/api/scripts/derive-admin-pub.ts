// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { ethers } from 'ethers';
import * as dotenv from 'dotenv';

dotenv.config();

const key = process.env.ADMIN_SIGNER_KEY;
if (!key) {
  console.error('ADMIN_SIGNER_KEY is required');
  process.exit(1);
}

const wallet = new ethers.Wallet(key);
console.log(`ADMIN_SIGNER_PUBLIC_KEY=${wallet.address}`);
