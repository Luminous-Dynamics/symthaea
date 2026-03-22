#!/usr/bin/env python3
"""
Create simple icon placeholders for Terra Atlas PWA
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create a simple Terra Atlas icon"""
    # Create a new image with a gradient background
    img = Image.new('RGB', (size, size), color='#030712')
    draw = ImageDraw.Draw(img)
    
    # Draw a circle representing Earth
    margin = size // 8
    earth_color = '#0ea5e9'
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=earth_color,
        outline='#ffffff',
        width=2
    )
    
    # Add some "data points" on the earth
    points = [
        (size * 0.3, size * 0.4),
        (size * 0.7, size * 0.3),
        (size * 0.5, size * 0.6),
        (size * 0.6, size * 0.7),
        (size * 0.4, size * 0.5),
    ]
    
    point_size = size // 40
    for x, y in points:
        draw.ellipse(
            [x - point_size, y - point_size, x + point_size, y + point_size],
            fill='#ffd700',
            outline='#ffffff',
            width=1
        )
    
    # Save the image
    img.save(output_path, 'PNG')
    print(f"✅ Created {output_path}")

def main():
    """Generate PWA icons"""
    public_dir = os.path.join(os.path.dirname(__file__), '..', 'public')
    
    # Create icons
    create_icon(192, os.path.join(public_dir, 'icon-192.png'))
    create_icon(512, os.path.join(public_dir, 'icon-512.png'))
    
    print("✨ Icons created successfully!")

if __name__ == "__main__":
    # Check if PIL is available
    try:
        from PIL import Image, ImageDraw
        main()
    except ImportError:
        print("⚠️  PIL not available, creating placeholder icons...")
        # Create empty placeholder files
        public_dir = os.path.join(os.path.dirname(__file__), '..', 'public')
        open(os.path.join(public_dir, 'icon-192.png'), 'w').close()
        open(os.path.join(public_dir, 'icon-512.png'), 'w').close()
        print("✅ Placeholder icons created")