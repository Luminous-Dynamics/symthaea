#!/usr/bin/env bash

# Setup script for production Stripe keys
# This script helps securely store Stripe production keys in BWS and Vercel

echo "🔐 Terra Atlas - Production Stripe Setup"
echo "======================================="
echo ""
echo "This script will help you securely store your Stripe production keys."
echo "You'll need your Stripe dashboard open to get these keys."
echo ""
echo "⚠️  IMPORTANT: Never commit these keys to git!"
echo ""

# Function to validate Stripe keys format
validate_stripe_key() {
    local key=$1
    local type=$2
    
    if [[ $type == "publishable" && $key == pk_live_* ]]; then
        return 0
    elif [[ $type == "secret" && $key == sk_live_* ]]; then
        return 0
    elif [[ $type == "webhook" && $key == YOUR_STRIPE_WEBHOOK_SECRET* ]]; then
        return 0
    else
        return 1
    fi
}

# Get publishable key
echo "1. From Stripe Dashboard > Developers > API Keys"
echo "   Copy your PUBLISHABLE KEY (starts with pk_live_)"
echo ""
read -p "Enter your Stripe Publishable Key: " STRIPE_PK

if ! validate_stripe_key "$STRIPE_PK" "publishable"; then
    echo "❌ Invalid publishable key format. Should start with pk_live_"
    exit 1
fi

# Get secret key
echo ""
echo "2. Copy your SECRET KEY (starts with sk_live_)"
echo "   ⚠️  This key gives full access to your Stripe account - keep it secure!"
echo ""
read -s -p "Enter your Stripe Secret Key: " STRIPE_SK
echo ""

if ! validate_stripe_key "$STRIPE_SK" "secret"; then
    echo "❌ Invalid secret key format. Should start with sk_live_"
    exit 1
fi

# Optional webhook secret
echo ""
echo "3. (Optional) From Stripe Dashboard > Developers > Webhooks"
echo "   Copy your webhook signing secret (starts with YOUR_STRIPE_WEBHOOK_SECRET)"
echo "   Press Enter to skip if you haven't set up webhooks yet."
echo ""
read -s -p "Enter your Stripe Webhook Secret (or press Enter to skip): " STRIPE_WEBHOOK
echo ""

# Store in BWS (Bitwarden Secrets Manager)
echo ""
echo "📝 Storing in Bitwarden Secrets Manager..."

# Store publishable key
echo "$STRIPE_PK" | bws create stripe-pk-live > /dev/null 2>&1 && echo "✅ Publishable key stored" || echo "⚠️  Could not store in BWS (may already exist)"

# Store secret key
echo "$STRIPE_SK" | bws create stripe-sk-live > /dev/null 2>&1 && echo "✅ Secret key stored" || echo "⚠️  Could not store in BWS (may already exist)"

# Store webhook secret if provided
if [[ ! -z "$STRIPE_WEBHOOK" ]]; then
    if validate_stripe_key "$STRIPE_WEBHOOK" "webhook"; then
        echo "$STRIPE_WEBHOOK" | bws create stripe-webhook-live > /dev/null 2>&1 && echo "✅ Webhook secret stored" || echo "⚠️  Could not store in BWS"
    fi
fi

# Update Vercel environment variables
echo ""
echo "📝 Updating Vercel environment variables..."

# Update production environment
echo "$STRIPE_PK" | vercel env add NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY production --force
echo "$STRIPE_SK" | vercel env add STRIPE_SECRET_KEY production --force

if [[ ! -z "$STRIPE_WEBHOOK" ]] && validate_stripe_key "$STRIPE_WEBHOOK" "webhook"; then
    echo "$STRIPE_WEBHOOK" | vercel env add STRIPE_WEBHOOK_SECRET production --force
fi

echo ""
echo "✅ Production Stripe keys configured!"
echo ""
echo "Next steps:"
echo "1. Redeploy to production: vercel --prod"
echo "2. Test a real payment at https://atlas.luminousdynamics.io"
echo "3. Monitor payments at https://dashboard.stripe.com"
echo ""
echo "⚠️  Remember:"
echo "- Never share your secret key"
echo "- Rotate keys regularly"
echo "- Use webhook signatures to validate events"
echo "- Enable Stripe Radar for fraud protection"