// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { json, type RequestHandler } from '@sveltejs/kit';
import { env } from '$env/dynamic/private';

const PINATA_ENDPOINT = 'https://api.pinata.cloud/pinning/pinFileToIPFS';

export const POST: RequestHandler = async ({ request }) => {
  const form = await request.formData();
  const file = form.get('file');

  if (!(file instanceof File)) {
    return json({ error: 'Missing file upload.' }, { status: 400 });
  }

  const pinataJwt = env.PINATA_JWT;
  const ipfsApiUrl = env.IPFS_API_URL?.replace(/\/$/, '');

  let uploadUrl: string;
  let headers: Record<string, string> | undefined;

  if (pinataJwt) {
    uploadUrl = PINATA_ENDPOINT;
    headers = { Authorization: `Bearer ${pinataJwt}` };
  } else if (ipfsApiUrl) {
    uploadUrl = `${ipfsApiUrl}/api/v0/add?pin=true`;
  } else {
    return json(
      { error: 'No IPFS upload backend configured. Set PINATA_JWT or IPFS_API_URL.' },
      { status: 500 }
    );
  }

  const outboundForm = new FormData();
  outboundForm.append('file', file, file.name);

  const response = await fetch(uploadUrl, {
    method: 'POST',
    headers,
    body: outboundForm,
  });

  if (!response.ok) {
    const details = await response.text();
    return json(
      { error: 'IPFS upload failed.', details },
      { status: 502 }
    );
  }

  const data = await response.json() as { IpfsHash?: string; Hash?: string; cid?: string };
  const cid = data.IpfsHash ?? data.Hash ?? data.cid;

  if (!cid) {
    return json({ error: 'IPFS upload failed: missing CID in response.' }, { status: 502 });
  }

  return json({ cid });
};
