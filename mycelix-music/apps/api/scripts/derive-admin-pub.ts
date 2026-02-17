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
