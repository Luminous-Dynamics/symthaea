# 🚀 Terra Atlas MVP Deployment Guide

## Current Status
✅ **Terra Atlas MVP is running locally on port 3002**
- API: http://localhost:3002/api
- App: http://localhost:3002
- Data: 325 features (150 fires, 75 earthquakes, 100 weather stations)

## Option 1: Deploy to Vercel (Recommended)

### Prerequisites
- Create account at [vercel.com](https://vercel.com)
- Install Vercel CLI: `npm i -g vercel`

### Deployment Steps
```bash
# From terra-atlas-mvp directory
cd /srv/luminous-dynamics/terra-atlas-mvp

# Login to Vercel
vercel login

# Deploy (follow prompts)
vercel

# For production deployment
vercel --prod
```

### Expected Prompts
1. **Set up and deploy?** → Yes
2. **Which scope?** → Choose your account
3. **Link to existing project?** → No (create new)
4. **Project name?** → terra-atlas-mvp
5. **Directory?** → ./
6. **Override settings?** → No

### Post-Deployment
- Your app will be live at: `https://terra-atlas-mvp.vercel.app`
- Custom domain: Add `terra-atlas.earth` in Vercel dashboard
- Environment variables: None required for MVP

## Option 2: Deploy to Netlify

### Steps
```bash
# Build the project
npm run build

# Install Netlify CLI
npm i -g netlify-cli

# Deploy
netlify deploy --dir=.next

# For production
netlify deploy --prod --dir=.next
```

## Option 3: Self-Host with Docker

### Create Dockerfile
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Build and Run
```bash
docker build -t terra-atlas .
docker run -p 3000:3000 terra-atlas
```

## Environment Variables (Future)

For production with real data sources:
```env
# NASA FIRMS
FIRMS_API_KEY=your_api_key_here

# OpenWeatherMap  
OPENWEATHER_API_KEY=your_api_key_here

# Carbon Monitor
CARBON_API_KEY=your_api_key_here
```

## Domain Setup

### For terra-atlas.earth
1. Add domain in Vercel/Netlify dashboard
2. Update DNS records:
   - A Record: Points to Vercel/Netlify IP
   - CNAME: www → terra-atlas.earth

## GitHub Actions (Already Configured)

The `.github/workflows/update-data.yml` will:
- Run every 3 hours
- Fetch latest data from all sources
- Commit updates to repository
- Trigger automatic redeploy

## Monitoring

### Vercel Analytics (Free tier)
- Page views
- API usage
- Performance metrics
- Error tracking

### Custom Analytics
Add to `app/layout.tsx`:
```javascript
// Google Analytics, Plausible, etc.
```

## Cost Analysis

### Current (MVP)
- **Hosting**: $0 (Vercel/Netlify free tier)
- **Data**: $0 (Demo data)
- **Maps**: $0 (Mapbox 50k free loads)
- **Domain**: $10/year
- **Total**: $0.83/month

### Scaling Triggers
Move to paid tier when:
- >100GB bandwidth/month
- >100K API requests/day
- >50K map loads/month
- Need real-time data processing

## Success Metrics

Track these KPIs:
- [ ] Unique visitors
- [ ] API calls
- [ ] Time on site
- [ ] Data accuracy
- [ ] User feedback

## Troubleshooting

### Build Errors
- Clear `.next` folder: `rm -rf .next`
- Clear cache: `npm cache clean --force`
- Reinstall: `rm -rf node_modules && npm install`

### Runtime Errors
- Check browser console
- Verify API endpoints
- Check data file permissions

### Performance Issues
- Enable caching headers
- Optimize images
- Use CDN for static assets

## Security Checklist

- [ ] API rate limiting enabled
- [ ] CORS configured properly
- [ ] Environment variables secured
- [ ] No sensitive data in repo
- [ ] HTTPS enforced

## Next Steps After Deployment

1. **Week 1**: Monitor performance, gather feedback
2. **Week 2**: Add real data sources
3. **Week 3**: Implement Trust Layer
4. **Week 4**: Launch API documentation
5. **Month 2**: Add premium features

## Support

- GitHub Issues: [terra-atlas/mvp/issues](https://github.com/terra-atlas/mvp/issues)
- Discord: [discord.gg/terra-atlas](https://discord.gg/terra-atlas)
- Email: support@terra-atlas.earth

---

**Ready to make the inevitable happen!** 🌍

When you're ready to deploy, just run:
```bash
npx vercel --prod
```

Your Planetary Nervous System awaits! 🚀