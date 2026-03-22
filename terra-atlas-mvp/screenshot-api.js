const https = require('https');
const fs = require('fs');

// Using shot.screenshotapi.net - free screenshot service
const url = 'https://atlas.luminousdynamics.io';
const apiUrl = `https://shot.screenshotapi.net/screenshot?url=${encodeURIComponent(url)}&width=1920&height=1080&output=image&file_type=png&wait_for_event=load`;

console.log('📸 Requesting professional screenshot from API...');
console.log(`URL: ${url}`);

https.get(apiUrl, (response) => {
  if (response.statusCode === 200) {
    const file = fs.createWriteStream('screenshots/homepage-professional.png');
    response.pipe(file);
    file.on('finish', () => {
      file.close();
      console.log('✅ Professional screenshot saved to screenshots/homepage-professional.png');
      console.log(`Size: ${fs.statSync('screenshots/homepage-professional.png').size} bytes`);
    });
  } else {
    console.error(`❌ HTTP ${response.statusCode}: ${response.statusMessage}`);
  }
}).on('error', (err) => {
  console.error('❌ Error:', err.message);
});