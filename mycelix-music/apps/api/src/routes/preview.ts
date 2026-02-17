import { Pool } from 'pg';
import { isAdmin } from '../security';

type PreviewResult = {
  totalNet: number;
  totalPlays: number;
};

type ModuleImpact = {
  name: string;
  lift: number;
};

export function registerPreviewRoute(app: any, pool: Pool) {
  app.post('/api/strategy-configs/:id/preview', async (req: any, res: any) => {
    try {
      if (!isAdmin(req) && String(process.env.ENABLE_PREVIEW_PUBLIC || 'false').toLowerCase() !== 'true') {
        return res.status(403).json({ error: 'forbidden' });
      }
      const { id } = req.params;
      const { days = 30, modules } = req.body || {};
      const cfg = await pool.query('SELECT payload FROM strategy_configs WHERE id = $1 AND published = TRUE', [id]);
      if (!cfg.rows.length) {
        return res.status(404).json({ error: 'not_found' });
      }
      const daysInt = Math.min(90, Math.max(1, parseInt(String(days), 10) || 30));
      const plays = await pool.query(
        `SELECT payment_type, amount::float AS amount
         FROM plays
         WHERE timestamp >= NOW() - ($1::text || ' days')::interval`,
        [String(daysInt)],
      );
      const totalNet = plays.rows.reduce((sum: number, r: any) => sum + (r.amount || 0), 0);
      const totalPlays = plays.rowCount || 0;
      const perType = await pool.query(
        `SELECT payment_type, COUNT(*)::int AS plays, COALESCE(SUM(amount),0)::float AS net
         FROM plays
         WHERE timestamp >= NOW() - ($1::text || ' days')::interval
         GROUP BY payment_type`,
        [String(daysInt)],
      );
      const base: PreviewResult = { totalNet, totalPlays };

      // Simple simulation toggles for dynamic/loyalty modules
      const activeModules: string[] = Array.isArray(modules) ? modules.map(String) : [];
      const config = cfg.rows[0].payload as any;
      const impacts: ModuleImpact[] = [];

      // Heuristic lifts based on config and activity
      const avgPerDay = totalPlays / daysInt;
      let dynamicLift = 1;
      if (activeModules.includes('dynamic')) {
        // Surge pricing: log curve capped at 20%, scaled by config.pricing.dynamicLift if provided
        const cfgLift = Number(config?.pricing?.dynamicLift ?? 0);
        const maxSurge = Number(config?.pricing?.dynamicMax ?? 0.2);
        const curveFactor = Number(config?.pricing?.dynamicCurve ?? 15);
        const heuristic = 1 + Math.min(maxSurge, Math.log1p(Math.max(1, avgPerDay)) / Math.max(1, curveFactor));
        dynamicLift = heuristic * (1 + cfgLift);
        impacts.push({ name: 'dynamic', lift: dynamicLift });
      }

      let loyaltyLift = 1;
      if (activeModules.includes('loyalty')) {
        // Loyalty lift: modest baseline depending on activity, scaled by config.pricing.loyaltyLift if provided
        const cfgLift = Number(config?.pricing?.loyaltyLift ?? 0);
        const maxLoyalty = Number(config?.pricing?.loyaltyMax ?? 0.08);
        const baseHeuristic = 1 + Math.min(maxLoyalty, (avgPerDay / 1000));
        loyaltyLift = baseHeuristic * (1 + cfgLift);
        impacts.push({ name: 'loyalty', lift: loyaltyLift });
      }

      const simulatedNet = totalNet * dynamicLift * loyaltyLift;

      const compare = {
        base,
        simulated: {
          totalNet: simulatedNet,
          totalPlays,
          delta: simulatedNet - totalNet,
        },
        modules: activeModules,
        impacts: impacts.map((imp) => ({
          name: imp.name,
          lift: imp.lift,
          netDelta: totalNet * imp.lift - totalNet,
        })),
        perType: perType.rows,
      };

      return res.json({ preview: compare, config: cfg.rows[0].payload });
    } catch (e: any) {
      console.error('Preview failed', e);
      return res.status(500).json({ error: 'preview_failed', message: e?.message || 'unknown' });
    }
  });
}
