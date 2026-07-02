# Demo Music Files

This folder contains demo audio files for local development and testing.

## Quick Setup with Royalty-Free Music

### Option 1: YouTube Audio Library (Recommended)
1. Visit [YouTube Audio Library](https://www.youtube.com/audiolibrary)
2. Download royalty-free tracks (MP3 format)
3. Rename files to match the expected naming:
   - `neon-dreams.mp3`
   - `electric-sunset.mp3`
   - `cosmic-journey.mp3`
   - `midnight-echoes.mp3`
   - `digital-rain.mp3`
   - `ethereal-whispers.mp3`

### Option 2: Free Music Archive
1. Visit [Free Music Archive](https://freemusicarchive.org/)
2. Filter by "CC BY" license (attribution required)
3. Download MP3 files
4. Rename to match expected names above

### Option 3: Incompetech
1. Visit [Incompetech](https://incompetech.com/music/royalty-free/)
2. Download tracks under Creative Commons license
3. Rename files as needed

## File Requirements
- **Format**: MP3 (recommended) or OGG
- **Bitrate**: 128kbps or higher
- **Size**: Ideally under 10MB per file
- **Naming**: Use lowercase with hyphens (e.g., `neon-dreams.mp3`)

## Integration with Mycelix Music

Files in this folder are served at: `/demo-music/filename.mp3`

The app automatically falls back to Bunny CDN if demo files aren't available.

## License Attribution

Remember to credit the artists! Add attribution to:
`/srv/luminous-dynamics/04-infinite-play/core/mycelix-music/apps/web/MUSIC_CREDITS.md`

Example:
```
# Music Credits

- "Neon Dreams" by Artist Name - [License](link)
- "Electric Sunset" by Artist Name - CC BY 4.0
```

## For Production

Demo files are for **development only**. Production uses:
- **Bunny CDN** for global audio delivery ($0.01/GB)
- **IPFS** for decentralized storage
- **Artist uploads** through the platform

## Need Help?

Check the main README for CDN setup instructions.
