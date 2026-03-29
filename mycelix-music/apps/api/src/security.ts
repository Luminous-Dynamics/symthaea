// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { Request, Response, NextFunction } from 'express';

export function requireAdminKey(req: Request, res: Response, next: NextFunction) {
  const adminKey = process.env.API_ADMIN_KEY;
  if (!adminKey) {
    return res.status(503).json({ error: 'admin_key_not_configured' });
  }
  const provided = req.headers['x-api-key'];
  if (provided && String(provided) === adminKey) {
    return next();
  }
  return res.status(403).json({ error: 'auth_failed', reason: 'missing_admin_key' });
}

export function isAdmin(req: Request): boolean {
  const adminKey = process.env.API_ADMIN_KEY;
  const provided = req.headers['x-api-key'];
  return Boolean(adminKey && provided && String(provided) === adminKey);
}
