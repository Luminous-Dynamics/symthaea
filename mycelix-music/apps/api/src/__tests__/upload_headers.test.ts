import request from 'supertest';
import { app } from '../index';

describe('upload header validation', () => {
  it('rejects upload with missing content-type header when validation enabled', async () => {
    const res = await request(app)
      .post('/api/upload-to-ipfs')
      .attach('file', Buffer.from('abc'), { filename: 'test.mp3', contentType: undefined as any })
      .expect((r) => {
        if (r.status !== 400 && r.status !== 503 && r.status !== 403) {
          throw new Error(`unexpected status ${r.status}`);
        }
      });
    if (res.status === 400) {
      expect(res.body.error).toBe('invalid_request');
    }
  });
});
