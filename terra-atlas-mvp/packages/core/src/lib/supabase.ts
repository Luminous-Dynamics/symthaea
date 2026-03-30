// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { createClient } from '@supabase/supabase-js'

export const createSupabaseBrowserClient = (url?: string, anonKey?: string) => {
  const supabaseUrl =
    url || process.env.NEXT_PUBLIC_SUPABASE_URL || ''
  const supabaseAnonKey =
    anonKey || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''
  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error('Supabase URL or anon key missing')
  }
  return createClient(supabaseUrl, supabaseAnonKey, {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: true,
      flowType: 'pkce'
    }
  })
}
