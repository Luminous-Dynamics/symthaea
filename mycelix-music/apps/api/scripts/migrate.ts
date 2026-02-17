import fs from 'fs';
import path from 'path';
import * as url from 'url';
import { Pool } from 'pg';
import * as dotenv from 'dotenv';

dotenv.config();

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const MIGRATIONS_DIR = path.join(__dirname, '..', 'migrations');

async function main() {
  const dbUrl = process.env.DATABASE_URL;
  if (!dbUrl) {
    console.error('DATABASE_URL is required');
    process.exit(1);
  }
  const pool = new Pool({ connectionString: dbUrl });
  const client = await pool.connect();
  try {
    await client.query(`
      CREATE TABLE IF NOT EXISTS schema_migrations (
        id SERIAL PRIMARY KEY,
        filename TEXT UNIQUE NOT NULL,
        applied_at TIMESTAMP DEFAULT NOW()
      )
    `);

    const files = fs.readdirSync(MIGRATIONS_DIR)
      .filter((f) => f.endsWith('.sql'))
      .sort();

    for (const file of files) {
      const seen = await client.query('SELECT 1 FROM schema_migrations WHERE filename = $1', [file]);
      if (seen.rowCount > 0) {
        console.log(`Skip ${file} (already applied)`);
        continue;
      }
      const sql = fs.readFileSync(path.join(MIGRATIONS_DIR, file), 'utf8');
      console.log(`Applying ${file}...`);
      await client.query('BEGIN');
      await client.query(sql);
      await client.query('INSERT INTO schema_migrations (filename) VALUES ($1)', [file]);
      await client.query('COMMIT');
      console.log(`✓ ${file}`);
    }
    console.log('Migrations complete');
  } catch (e: any) {
    await client.query('ROLLBACK').catch(() => {});
    console.error('Migration failed:', e?.message || e);
    process.exitCode = 1;
  } finally {
    client.release();
    await pool.end();
  }
}

main();
