import request from 'supertest';
import { app } from '../index';

describe('upload route protection', () => {
  it('blocks uploads without admin key', async () => {
    const res = await request(app)
      .post('/api/upload-to-ipfs')
      .expect((r) => {
        if (r.status !== 403 && r.status !== 503) {
          throw new Error(`unexpected status ${r.status}`);
        }
      });
    expect(res.body.error).toBeDefined();
  });
});

const runHealth = String(process.env.RUN_HEALTH_TESTS || '').toLowerCase() === 'true';
(runHealth ? describe : describe.skip)('health/indexer endpoint', () => {
  it('responds with ok flag', async () => {
    const res = await request(app).get('/health/indexer');
    // Endpoint may be 200 or 500 depending on RPC/redis availability, but should include ok flag
    expect(res.body).toHaveProperty('ok');
  });
});

describe('strategy config access control', () => {
  it('denies unpublished latest fetch without admin key', async () => {
    const res = await request(app)
      .get('/api/strategy-configs/latest?admin=true')
      .expect(403);
    expect(res.body.error).toBe('forbidden');
  });

  it('denies unpublished fetch by id without admin key', async () => {
    const res = await request(app)
      .get('/api/strategy-configs/123?admin=true')
      .expect(403);
    expect(res.body.error).toBe('forbidden');
  });
});
