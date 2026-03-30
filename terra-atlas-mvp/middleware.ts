// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// Define public routes that don't require authentication
const PUBLIC_ROUTES = [
  '/',
  '/explore',
  '/landing',
  '/homepage',
  '/horizon',
  '/smr',
  '/west-texas-corridor',
  '/auth/login',
  '/auth/signup',
  '/auth/reset-password',
  '/auth/verify-email',
  '/auth/update-password',
  '/auth/callback',
]

// Define public API routes
const PUBLIC_API_ROUTES = [
  '/api/stats',
  '/api/sites',
  '/api/projects',
  '/api/discovery',
  '/api/content',
]

// Routes that require authentication
const PROTECTED_ROUTES = [
  '/dashboard',
  '/portfolio',
  // Note: /invest pages are public for viewing, auth required only for pledge action
]

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  // Allow all static files, images, and Next.js internals
  if (
    pathname.startsWith('/_next') ||
    pathname.startsWith('/static') ||
    pathname.startsWith('/favicon.ico') ||
    pathname.startsWith('/icon') ||
    pathname.startsWith('/manifest.json') ||
    pathname.match(/\.(png|jpg|jpeg|gif|svg|ico|webp|woff|woff2|ttf|eot)$/)
  ) {
    return NextResponse.next()
  }

  // Check if the route is explicitly public
  const isPublicRoute = PUBLIC_ROUTES.some(route => pathname === route || pathname.startsWith(route + '/'))
  const isPublicApiRoute = PUBLIC_API_ROUTES.some(route => pathname.startsWith(route))

  // Allow public routes and public API routes
  if (isPublicRoute || isPublicApiRoute) {
    return NextResponse.next()
  }

  // Check if route requires authentication
  const isProtectedRoute = PROTECTED_ROUTES.some(route => pathname.startsWith(route))

  if (isProtectedRoute) {
    // Check for auth token in cookies
    const authToken = request.cookies.get('sb-access-token') || request.cookies.get('sb-refresh-token')

    if (!authToken) {
      // Redirect to login with return URL
      const loginUrl = new URL('/auth/login', request.url)
      loginUrl.searchParams.set('returnUrl', pathname)
      return NextResponse.redirect(loginUrl)
    }
  }

  // Allow all other routes by default (open platform)
  return NextResponse.next()
}

// Configure middleware to run on all routes
export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
}
