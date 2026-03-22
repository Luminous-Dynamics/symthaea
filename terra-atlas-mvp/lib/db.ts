import { Pool } from 'pg'
import bcrypt from 'bcryptjs'

type QueryParams = Array<string | number | boolean | null | Record<string, unknown>>

const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://tstoltz@localhost:5434/terra_atlas?host=/srv/luminous-dynamics/terra-atlas-mvp/postgres-data',
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
})

pool.on('connect', () => {
  // eslint-disable-next-line no-console
  console.log('✅ Database connected')
})

pool.on('error', (err) => {
  // eslint-disable-next-line no-console
  console.error('❌ Database error:', err)
})

async function query(text: string, params?: QueryParams) {
  const client = await pool.connect()
  try {
    return await client.query(text, params)
  } finally {
    client.release()
  }
}

const db = {
  query,
  user: {
    async findByEmail(email: string) {
      const result = await query('SELECT * FROM users WHERE email = $1 LIMIT 1', [email])
      return result.rows[0]
    },
    async findByUsername(username: string) {
      const result = await query('SELECT * FROM users WHERE username = $1 LIMIT 1', [username])
      return result.rows[0]
    },
    async findById(id: string) {
      const result = await query('SELECT * FROM users WHERE id = $1 LIMIT 1', [id])
      return result.rows[0]
    },
    async create({ email, username, password, fullName }: { email: string; username: string; password: string; fullName?: string }) {
      const passwordHash = await bcrypt.hash(password, 10)
      const result = await query(
        `INSERT INTO users (email, username, password_hash, full_name)
         VALUES ($1, $2, $3, $4)
         RETURNING id, email, username, full_name, reputation_score, trust_level, created_at`,
        [email, username, passwordHash, fullName]
      )
      return result.rows[0]
    },
    async updateLastLogin(id: string) {
      await query('UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = $1', [id])
    },
    async updateReputation(userId: string) {
      const result = await query('SELECT calculate_user_reputation($1::uuid) as reputation', [userId])
      const reputation = result.rows[0]?.reputation ?? 0
      let trustLevel = 'novice'
      if (reputation > 2000) trustLevel = 'guardian'
      else if (reputation > 500) trustLevel = 'expert'
      else if (reputation > 100) trustLevel = 'contributor'
      await query('UPDATE users SET reputation_score = $1, trust_level = $2 WHERE id = $3', [reputation, trustLevel, userId])
      return { reputation, trustLevel }
    },
  },
  validation: {
    async create({ userId, dataPointId, validationType, comment, evidenceUrls, ipAddress, userAgent }: Record<string, any>) {
      const existing = await query('SELECT * FROM validations WHERE user_id = $1 AND data_point_id = $2', [userId, dataPointId])
      let result
      if (existing.rows.length > 0) {
        result = await query(
          `UPDATE validations SET
             validation_type = $3, comment = $4, evidence_urls = $5,
             ip_address = $6, user_agent = $7, updated_at = CURRENT_TIMESTAMP
           WHERE user_id = $1 AND data_point_id = $2 RETURNING *`,
          [userId, dataPointId, validationType, comment, evidenceUrls, ipAddress, userAgent]
        )
      } else {
        result = await query(
          `INSERT INTO validations (user_id, data_point_id, validation_type, comment, evidence_urls, ip_address, user_agent)
           VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING *`,
          [userId, dataPointId, validationType, comment, evidenceUrls, ipAddress, userAgent]
        )
      }
      await query('SELECT update_data_point_trust_score($1::uuid)', [dataPointId])
      return result.rows[0]
    },
    async findByDataPoint(dataPointId: string, limit = 100, offset = 0) {
      const result = await query(
        `SELECT v.*, u.username, u.avatar_url, u.trust_level, u.reputation_score
         FROM validations v
         JOIN users u ON v.user_id = u.id
         WHERE v.data_point_id = $1
         ORDER BY v.created_at DESC
         LIMIT $2 OFFSET $3`,
        [dataPointId, limit, offset]
      )
      return result.rows
    },
    async delete(userId: string, dataPointId: string) {
      const result = await query('DELETE FROM validations WHERE user_id = $1 AND data_point_id = $2 RETURNING *', [userId, dataPointId])
      if (result.rows.length > 0) {
        await query('SELECT update_data_point_trust_score($1::uuid)', [dataPointId])
      }
      return result.rows[0]
    },
  },
  session: {
    async create({ userId, refreshTokenHash, ipAddress, userAgent, expiresIn = 30 }: Record<string, any>) {
      const expiresAt = new Date()
      expiresAt.setDate(expiresAt.getDate() + expiresIn)
      const result = await query(
        `INSERT INTO sessions (user_id, refresh_token_hash, ip_address, user_agent, expires_at)
         VALUES ($1, $2, $3, $4, $5) RETURNING *`,
        [userId, refreshTokenHash, ipAddress, userAgent, expiresAt]
      )
      return result.rows[0]
    },
    async findByToken(refreshTokenHash: string) {
      const result = await query(
        'SELECT * FROM sessions WHERE refresh_token_hash = $1 AND is_active = true AND expires_at > NOW()',
        [refreshTokenHash]
      )
      return result.rows[0]
    },
    async revoke(id: string, reason: string) {
      await query('UPDATE sessions SET is_active = false, revoked_at = CURRENT_TIMESTAMP, revoked_reason = $2 WHERE id = $1', [id, reason])
    },
  },
  apiKey: {
    async create({ userId, keyHash, keyPrefix, name, description, scopes, rateLimit, expiresAt }: Record<string, any>) {
      const result = await query(
        `INSERT INTO user_api_keys (user_id, key_hash, key_prefix, name, description, scopes, rate_limit, expires_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING *`,
        [userId, keyHash, keyPrefix, name, description, scopes, rateLimit, expiresAt]
      )
      return result.rows[0]
    },
  },
  dataPoint: {
    async create(data: Record<string, any>) {
      const result = await query(
        `INSERT INTO data_points (latitude, longitude, data_type, source_id, source_name,
                                  title, description, properties, severity, confidence, observed_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) RETURNING *`,
        [
          data.latitude,
          data.longitude,
          data.dataType,
          data.sourceId,
          data.sourceName,
          data.title,
          data.description,
          data.properties,
          data.severity,
          data.confidence,
          data.observedAt,
        ]
      )
      return result.rows[0]
    },
    async findById(id: string) {
      const result = await query('SELECT * FROM data_points WHERE id = $1', [id])
      return result.rows[0]
    },
    async findByArea(minLat: number, maxLat: number, minLng: number, maxLng: number, limit = 1000) {
      const result = await query(
        `SELECT * FROM data_points
         WHERE latitude BETWEEN $1 AND $2
         AND longitude BETWEEN $3 AND $4
         ORDER BY trust_score DESC
         LIMIT $5`,
        [minLat, maxLat, minLng, maxLng, limit]
      )
      return result.rows
    },
  },
}

export default db
