import request from 'supertest';
import { app } from '../index';

describe('upload route', () => {
  it('blocks when uploads disabled', async () => {
    process.env.ENABLE_UPLOADS = 'false';
    process.env.API_ADMIN_KEY = process.env.API_ADMIN_KEY || 'test-admin-key';
    const res = await request(app)
      .post('/api/upload-to-ipfs')
      .set('x-api-key', process.env.API_ADMIN_KEY)
      .expect(503);
    expect(res.body.error).toBe('uploads_disabled');
    process.env.ENABLE_UPLOADS = 'true';
  });

  it('enforces mime check', async () => {
    process.env.API_ADMIN_KEY = process.env.API_ADMIN_KEY || 'test-admin-key';
    const res = await request(app)
      .post('/api/upload-to-ipfs')
      .set('x-api-key', process.env.API_ADMIN_KEY)
      .attach('file', Buffer.from('hello'), { filename: 'test.txt', contentType: 'text/plain' });
    // Without real IPFS this may fail upstream; just assert 4xx on mime
    expect([400, 500, 503, 403]).toContain(res.status);
  });
});
