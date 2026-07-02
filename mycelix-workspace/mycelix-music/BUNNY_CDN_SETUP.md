# Bunny CDN Setup Guide for Mycelix Music

## Overview
Bunny CDN provides global audio delivery at $0.01/GB with 114 PoPs worldwide.

**API Key**: Securely stored in Bitwarden Secrets Manager as `bunny-cdn-api-key`

## Step 1: Create Storage Zone

1. Go to [Bunny CDN Dashboard](https://panel.bunny.net/)
2. Navigate to **Storage** → **Add Storage Zone**
3. Configure:
   - **Name**: `mycelix-music` (or your preferred name)
   - **Region**: Choose based on audience (e.g., `New York` for US)
   - **Replication**: Enable for better global performance
   - **Price Tier**: Standard ($0.01/GB storage + $0.01/GB bandwidth)
4. Click **Add Storage Zone**
5. **Save the FTP/API credentials** shown after creation

## Step 2: Create Pull Zone (CDN)

1. Navigate to **CDN** → **Add Pull Zone**
2. Configure:
   - **Name**: `mycelix-music-cdn`
   - **Origin URL**: Select your storage zone (`mycelix-music`)
   - **Type**: Select "Standard"
3. Advanced Settings:
   - **Edge Rules**: Enable "Cache Everything"
   - **Cache Expiry**: 31536000 seconds (1 year for audio files)
   - **CORS**: Enable for web player access
4. Click **Add Pull Zone**
5. **Copy the Pull Zone URL** (e.g., `mycelix-music-cdn.b-cdn.net`)

## Step 3: Configure CORS (Critical!)

In your Pull Zone settings:
1. Go to **Edge Rules** → **CORS Configuration**
2. Add headers:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, HEAD, OPTIONS
Access-Control-Allow-Headers: Range, Content-Type
```

## Step 4: Add Environment Variables to Vercel

1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Add:
```
BUNNY_CDN_HOSTNAME=mycelix-music-cdn.b-cdn.net
BUNNY_CDN_STORAGE_ZONE=mycelix-music
BUNNY_CDN_API_KEY=<your-api-key>
BUNNY_STORAGE_API_KEY=<your-storage-api-key>
```

## Step 5: Upload Audio Files

### Option A: Web Interface
1. Go to **Storage** → `mycelix-music`
2. Click **Upload Files**
3. Organize in folders:
   - `/music/artist-name/song-title.mp3`
   - `/music/covers/song-id.jpg`

### Option B: FTP Client (Recommended for bulk uploads)
```bash
# Using curl
curl -X PUT \
  --data-binary @song.mp3 \
  -H "AccessKey: YOUR_STORAGE_API_KEY" \
  https://storage.bunnycdn.com/mycelix-music/music/artist/song.mp3
```

### Option C: Via Script
```bash
#!/bin/bash
STORAGE_API_KEY=$(bws get bunny-cdn-api-key)
STORAGE_ZONE="mycelix-music"

# Upload a file
upload_to_bunny() {
  local file=$1
  local path=$2

  curl -X PUT \
    --data-binary "@$file" \
    -H "AccessKey: $STORAGE_API_KEY" \
    "https://storage.bunnycdn.com/$STORAGE_ZONE/$path"
}

# Example
upload_to_bunny "digital-dreams.mp3" "music/nova-synthesis/digital-dreams.mp3"
```

## Step 6: Update Application Code

### Add CDN Helper Function
Create `apps/web/src/lib/cdn.ts`:
```typescript
const BUNNY_CDN_HOSTNAME = process.env.NEXT_PUBLIC_BUNNY_CDN_HOSTNAME || 'mycelix-music-cdn.b-cdn.net';

export function getCDNUrl(path: string): string {
  // Remove leading slash if present
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  return `https://${BUNNY_CDN_HOSTNAME}/${cleanPath}`;
}

export function getAudioUrl(song: Song): string {
  // Try local demo file first (for development)
  if (song.audioUrl && song.audioUrl.startsWith('/demo-music/')) {
    return song.audioUrl;
  }

  // Use CDN for production
  if (song.audioUrl) {
    return getCDNUrl(song.audioUrl);
  }

  // Fallback: construct from IPFS hash
  return getCDNUrl(`music/${song.artist.toLowerCase().replace(/\s+/g, '-')}/${song.id}.mp3`);
}
```

### Update MusicPlayer Component
```typescript
import { getAudioUrl } from '@/lib/cdn';

// In the component
const audioUrl = getAudioUrl(song);
<audio ref={audioRef} src={audioUrl} />
```

## Step 7: Test CDN Delivery

1. Upload a test MP3 to your storage zone
2. Access via Pull Zone URL:
   ```
   https://mycelix-music-cdn.b-cdn.net/music/test.mp3
   ```
3. Verify headers include CORS
4. Test playback in the web player

## Monitoring & Analytics

- **Dashboard**: https://panel.bunny.net/
- **Traffic Stats**: Real-time bandwidth and request metrics
- **Alerts**: Set up billing alerts for unexpected usage
- **Logs**: Enable access logs for debugging

## Cost Estimation

**Example Monthly Costs**:
- 1,000 songs × 5MB = 5GB storage = **$0.05/month**
- 100,000 streams × 5MB = 500GB bandwidth = **$5.00/month**
- **Total**: ~$5.05/month for 100K streams

Compare to traditional hosting:
- AWS S3: ~$20-30/month for same usage
- Azure: ~$25-35/month
- **Bunny Savings**: 4-5x cheaper

## Troubleshooting

### CORS Errors
- Verify CORS headers in Pull Zone settings
- Check browser console for specific error
- Test with curl: `curl -I https://your-cdn.b-cdn.net/file.mp3`

### 404 Not Found
- Verify file uploaded to correct path
- Check storage zone name matches Pull Zone origin
- Wait 60 seconds for cache propagation

### Slow Loading
- Enable replication in storage zone settings
- Verify closest PoP being used (check response headers)
- Consider enabling "Smart Cache" in Pull Zone

## Security

### Signed URLs (Optional for premium content)
```typescript
import crypto from 'crypto';

function generateSignedUrl(path: string, expiresIn: number = 3600): string {
  const expires = Math.floor(Date.now() / 1000) + expiresIn;
  const securityKey = process.env.BUNNY_CDN_SECURITY_KEY;

  const signatureData = `${securityKey}${path}${expires}`;
  const signature = crypto.createHash('sha256').update(signatureData).digest('hex');

  return `https://${BUNNY_CDN_HOSTNAME}${path}?expires=${expires}&signature=${signature}`;
}
```

## Next Steps

1. ✅ Create Storage Zone
2. ✅ Create Pull Zone
3. ✅ Configure CORS
4. ✅ Add environment variables to Vercel
5. ✅ Upload test audio file
6. ✅ Test playback
7. ✅ Update application code
8. ✅ Deploy to production

## Resources

- [Bunny CDN Dashboard](https://panel.bunny.net/)
- [API Documentation](https://docs.bunny.net/reference/storage-api)
- [Support](https://support.bunny.net/)
