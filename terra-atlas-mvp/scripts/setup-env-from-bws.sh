#!/bin/bash
# Terra Atlas - Set Environment Variables from BWS
# This script retrieves all API keys from Bitwarden Secrets Manager

set -a # Auto-export all variables

echo "🔑 Terra Atlas Environment Setup from BWS"
echo "=========================================="
echo ""

# Core Database
echo "📊 Setting up database credentials..."
export NEXT_PUBLIC_SUPABASE_URL=$(bws get supabase-prod-url)
export NEXT_PUBLIC_SUPABASE_ANON_KEY=$(bws get supabase-prod-anon-key)
export SUPABASE_SERVICE_ROLE_KEY=$(bws get supabase-prod-service-key)
export DATABASE_URL=$(bws get postgres-url)
export REDIS_URL=$(bws get redis-url)

# Mapping & Visualization
echo "🗺️  Setting up mapping services..."
export NEXT_PUBLIC_MAPBOX_TOKEN=$(bws get mapbox-prod-public-token)
export NEXT_PUBLIC_CESIUM_TOKEN=$(bws get cesium-prod-ion-token)
export GOOGLE_MAPS_API_KEY=$(bws get google-maps-api-key)
export PLANET_API_KEY=$(bws get planet-api-key)

# Energy Data APIs
echo "⚡ Setting up energy data APIs..."
export EIA_API_KEY=$(bws get eia-api-key)
export NREL_API_KEY=$(bws get nrel-api-key)
export OPENWEATHER_API_KEY=$(bws get openweather-api-key)
export GOOGLE_EARTH_ENGINE_KEY=$(bws get google-earth-engine-key)
export COPERNICUS_API_KEY=$(bws get copernicus-api-key)

# Financial & Payments
echo "💳 Setting up payment processing..."
export STRIPE_PUBLISHABLE_KEY=$(bws get stripe-publishable-key)
export STRIPE_SECRET_KEY=$(bws get stripe-secret-key)
export PLAID_CLIENT_ID=$(bws get plaid-client-id)
export PLAID_SECRET=$(bws get plaid-secret)
export ALPHA_VANTAGE_API_KEY=$(bws get alpha-vantage-api-key)
export IEX_CLOUD_API_KEY=$(bws get iex-cloud-api-key)

# Authentication
echo "🔐 Setting up authentication..."
export NEXTAUTH_SECRET=$(bws get nextauth-secret)
export GOOGLE_CLIENT_ID=$(bws get google-client-id)
export GOOGLE_CLIENT_SECRET=$(bws get google-client-secret)
export GITHUB_CLIENT_ID=$(bws get github-client-id)
export GITHUB_CLIENT_SECRET=$(bws get github-client-secret)
export GITHUB_TOKEN=$(bws get github-token)

# Communications
echo "📧 Setting up communication services..."
export TWILIO_ACCOUNT_SID=$(bws get twilio-account-sid)
export TWILIO_AUTH_TOKEN=$(bws get twilio-auth-token)
export SENDGRID_API_KEY=$(bws get sendgrid-api-key)
export DISCORD_TOKEN=$(bws get discord-token)

# AI Services
echo "🤖 Setting up AI services..."
export OPENAI_API_KEY=$(bws get openai-api-key)
export ANTHROPIC_API_KEY=$(bws get anthropic-api-key)

# Infrastructure
echo "🏗️  Setting up infrastructure..."
export CLOUDFLARE_API_TOKEN=$(bws get cloudflare-api-token)
export VERCEL_TOKEN=$(bws get vercel-token)

# Regional Grids
echo "🌍 Setting up regional grid APIs..."
export OPENNEM_API_KEY=$(bws get opennem-api-key)
export ENTSO_E_API_KEY=$(bws get entso-e-api-key)

# Local Services
echo "🏠 Setting up local services..."
export PYTHON_BACKEND_URL=http://localhost:8000
export SITE_ANALYSIS_API=http://localhost:8000/api/analyze-site
export WEBSOCKET_URL=ws://localhost:3002
export NODE_ENV=development
export NEXT_TELEMETRY_DISABLED=1

set +a # Stop auto-export

echo ""
echo "✅ Environment configured with $(env | grep -E "API_KEY|TOKEN|SECRET" | wc -l) API keys"
echo ""
echo "Available services:"
echo "  ✅ Supabase Database"
echo "  ✅ Mapbox + Cesium + Google Maps"
echo "  ✅ Planet Labs Satellite Imagery"
echo "  ✅ EIA + NREL Energy Data"
echo "  ✅ OpenWeather + Earth Engine"
echo "  ✅ Stripe + Plaid Payments"
echo "  ✅ OAuth (Google, GitHub)"
echo "  ✅ Twilio + SendGrid Communications"
echo "  ✅ OpenAI + Anthropic AI"
echo "  ✅ Cloudflare + Vercel Infrastructure"
echo ""
echo "🚀 Ready to start Terra Atlas!"
echo ""
echo "Next steps:"
echo "  1. npm install          # Install dependencies"
echo "  2. npm run migrate:data # Import 79,193 projects"
echo "  3. npm run dev          # Start development server"
echo ""