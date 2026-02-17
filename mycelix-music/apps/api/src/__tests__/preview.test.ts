import request from 'supertest';
import { app, pool } from '../index';

const hasDb = !!process.env.DATABASE_URL;

(hasDb ? describe : describe.skip)('strategy preview', () => {
  let configId: number;

  beforeAll(async () => {
    const inserted = await pool.query(
      `INSERT INTO strategy_configs (name, payload, published, hash) VALUES ($1, '{}'::jsonb, TRUE, $2)
       RETURNING id`,
      ['Test', '0xhash'],
    );
    configId = inserted.rows[0].id;
  });

  it('rejects preview when admin required', async () => {
    process.env.ENABLE_PREVIEW_PUBLIC = 'false';
    const res = await request(app)
      .post(`/api/strategy-configs/${configId}/preview`)
      .send({ days: 1 });
    expect(res.status).toBe(403);
  });

  it('allows preview when public enabled', async () => {
    process.env.ENABLE_PREVIEW_PUBLIC = 'true';
    const res = await request(app)
      .post(`/api/strategy-configs/${configId}/preview`)
      .send({ days: 1, modules: [] });
    expect([200, 404]).toContain(res.status);
  });
});
