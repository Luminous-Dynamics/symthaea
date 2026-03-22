import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { env } from '@/lib/env'
import logger from '@/lib/logger'

export async function GET(request: NextRequest) {
  const requestUrl = new URL(request.url)
  const code = requestUrl.searchParams.get('code')

  if (code) {
    const supabaseUrl = env.NEXT_PUBLIC_SUPABASE_URL
    const supabaseAnonKey = env.NEXT_PUBLIC_SUPABASE_ANON_KEY
    
    const supabase = createClient(supabaseUrl, supabaseAnonKey)
    
    const { error } = await supabase.auth.exchangeCodeForSession(code)
    
    if (error) {
      logger.error('Auth callback error:', error)
      return NextResponse.redirect(`${requestUrl.origin}/auth/login?error=${error.message}`)
    }
  }

  // URL to redirect to after sign in process completes
  return NextResponse.redirect(`${requestUrl.origin}/dashboard`)
}
