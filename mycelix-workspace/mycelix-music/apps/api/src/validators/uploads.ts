// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { z } from 'zod';

export const uploadQuerySchema = z.object({
  // No query params currently; placeholder for future preflight toggles
});

export const uploadHeadersSchema = z.object({
  'content-type': z.string().optional(),
});

export function validateUploadHeaders(req: any, res: any, next: any) {
  const result = uploadHeadersSchema.safeParse(req.headers);
  if (!result.success) {
    return res.status(400).json({
      error: 'invalid_request',
      issues: result.error.issues.map((i) => ({
        path: i.path.join('.'),
        message: i.message,
      })),
    });
  }
  return next();
}
