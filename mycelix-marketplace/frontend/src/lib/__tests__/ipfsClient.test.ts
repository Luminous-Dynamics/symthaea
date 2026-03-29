// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { describe, it, expect, vi } from 'vitest';
import { resolvePinataConfig, uploadFile } from '../ipfs/ipfsClient';

const mockConfig = {
  jwt: '',
  gateway: 'https://gateway.pinata.cloud',
  allowMock: false,
};

describe('ipfsClient configuration', () => {
  it('resolves config from env variables', () => {
    const config = resolvePinataConfig({
      VITE_PINATA_JWT: 'jwt_token',
      VITE_PINATA_GATEWAY: 'https://example.gateway',
      VITE_ALLOW_MOCK_IPFS: 'true',
    });

    expect(config).toEqual({
      jwt: 'jwt_token',
      gateway: 'https://example.gateway',
      allowMock: true,
    });
  });
});

describe('uploadFile', () => {
  const dummyFile = new File([new Blob(['hello world'])], 'hello.txt', {
    type: 'text/plain',
  });

  it('throws when Pinata is not configured and mock uploads are disabled', async () => {
    await expect(uploadFile(dummyFile, mockConfig)).rejects.toThrow(/Pinata JWT missing/);
  });

  it('falls back to mock CIDs when allowed', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch');

    const cid = await uploadFile(dummyFile, { ...mockConfig, allowMock: true });

    expect(cid.startsWith('Qm')).toBe(true);
    expect(cid).toHaveLength(46);
    expect(fetchSpy).not.toHaveBeenCalled();

    fetchSpy.mockRestore();
  });
});
