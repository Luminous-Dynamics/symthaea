// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Email Templates Module
 *
 * Rich email template system with variable interpolation
 */

export {
  EmailTemplateEngine,
  type Template,
  type TemplateCategory,
  type TemplateVariable,
  type TemplateFilter,
  type RenderOptions,
  type RenderResult,
} from './EmailTemplateEngine';

export default EmailTemplateEngine;
