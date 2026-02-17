import { PaymentModel } from '@/lib';

export interface Song {
  id: string;
  title: string;
  artist: string;
  genre: string;
  description: string;
  ipfsHash: string;
  paymentModel: PaymentModel;
  plays: number;
  earnings: string;
  coverArt: string;  // Cover art image URL
  audioUrl?: string;  // Audio file URL (demo or CDN)
  freePlaysRemaining?: number;  // For freemium model
  tipAmount?: string;  // For pay-what-you-want
}

/**
 * Mock song data for UI demo
 * Sources: Free Music Archive, Incompetech, ccMixter
 * All music used is Creative Commons licensed
 *
 * Economic Models:
 * - PAY_PER_STREAM: $0.01 per play (40% of songs)
 * - FREEMIUM: 3 free plays, then $0.01/stream (30% of songs)
 * - PAY_WHAT_YOU_WANT: Listen free, tip what you want (20% of songs)
 * - PATRONAGE: Monthly subscription for unlimited plays (10% of songs)
 */
export const mockSongs: Song[] = [
  // Electronic - Freemium (try before you buy)
  {
    id: '1',
    title: 'Digital Dreams',
    artist: 'Nova Synthesis',
    genre: 'Electronic',
    description: 'Ethereal electronic soundscape with ambient textures',
    ipfsHash: 'QmX8R3mZ9kL5pN2wQ7vB4sT6yH8jF1cD3aE9gU0iO5mP7',
    paymentModel: PaymentModel.FREEMIUM,
    plays: 12847,
    earnings: '89.23',
    coverArt: 'https://images.unsplash.com/photo-1571330735066-03aaa9429d89?w=600&h=600&fit=crop',
    audioUrl: '/demo-music/digital-dreams.mp3',
    freePlaysRemaining: 3,
  },
  {
    id: '2',
    title: 'Neon Waves',
    artist: 'Synthwave Collective',
    genre: 'Electronic',
    description: 'Retro synthwave with driving basslines',
    ipfsHash: 'QmY9S4nA0lM6qO3xR8wC5tU7zI9kG2dE4bF0hV1jP6nQ8',
    paymentModel: PaymentModel.PAY_WHAT_YOU_WANT,
    plays: 8234,
    earnings: '2,047.50',
    coverArt: 'https://images.unsplash.com/photo-1514320291840-2e0a9bf2a9ae?w=600&h=600&fit=crop',
    audioUrl: '/demo-music/neon-waves.mp3',
    tipAmount: 'avg $0.25',
  },

  // Rock - Pay Per Stream
  {
    id: '3',
    title: 'Thunder Road',
    artist: 'The Voltage',
    genre: 'Rock',
    description: 'High-energy rock anthem with powerful vocals',
    ipfsHash: 'QmZ0T5oB1mN7rP4yS9xD6uV8aJ0lH3eF5cG1iW2kQ7oR9',
    paymentModel: PaymentModel.PAY_PER_STREAM,
    plays: 15692,
    earnings: '156.92',
    coverArt: 'https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=600&h=600&fit=crop',
    audioUrl: '/demo-music/thunder-road.mp3',
  },
  {
    id: '4',
    title: 'Midnight Drive',
    artist: 'Urban Echo',
    genre: 'Rock',
    description: 'Indie rock with atmospheric guitars',
    ipfsHash: 'QmA1U6pC2nO8sQ5zT0yE7vW9bK1mI4fG6dH2jX3lR8pS0',
    paymentModel: PaymentModel.PAY_PER_STREAM,
    plays: 9876,
    earnings: '98.76',
    coverArt: 'https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=600&h=600&fit=crop',
    audioUrl: '/demo-music/midnight-drive.mp3',
  },

  // Ambient - Pay What You Want (tip-based, true gift economy)
  {
    id: '5',
    title: 'Morning Mist',
    artist: 'Serene Soundscapes',
    genre: 'Ambient',
    description: 'Peaceful ambient meditation music',
    ipfsHash: 'QmB2V7qD3oP9tR6aU1zF8wY0cL2nJ5gH7eI3kY4mS9qT1',
    paymentModel: PaymentModel.PAY_WHAT_YOU_WANT,
    plays: 23451,
    earnings: '3,517.65',
    coverArt: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&h=600&fit=crop',
    audioUrl: '/demo-music/morning-mist.mp3',
    tipAmount: 'avg $0.15',
  },
  {
    id: '6',
    title: 'Ocean Depths',
    artist: 'Aquatic Resonance',
    genre: 'Ambient',
    description: 'Deep underwater ambient journey',
    ipfsHash: 'QmC3W8rE4pQ0uS7bV2aG9xZ1dM3oK6hI8fJ4lZ5nT0rU2',
    paymentModel: PaymentModel.FREEMIUM,
    plays: 18723,
    earnings: '154.89',
    coverArt: 'https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=600&h=600&fit=crop',
    audioUrl: '/demo-music/ocean-depths.mp3',
    freePlaysRemaining: 3,
  },

  // Jazz - Patronage
  {
    id: '7',
    title: 'Blue Note Variations',
    artist: 'Marcus Jazz Trio',
    genre: 'Jazz',
    description: 'Classic jazz trio with modern improvisation',
    ipfsHash: 'QmD4X9sF5qR1vT8cW3bH0yA2eN4pL7iJ9gK5mA6oU1sV3',
    paymentModel: PaymentModel.PATRONAGE,
    plays: 6432,
    earnings: '512.45',
    coverArt: 'https://images.unsplash.com/photo-1415201364774-f6f0bb35f28f?w=600&h=600&fit=crop',
  },
  {
    id: '8',
    title: 'Smooth Serenade',
    artist: 'Luna Fitzgerald',
    genre: 'Jazz',
    description: 'Sultry jazz vocals with piano accompaniment',
    ipfsHash: 'QmE5Y0tG6rS2wU9dX4cI1zA3fO5qM8jK0hL6nB7pV2tW4',
    paymentModel: PaymentModel.PATRONAGE,
    plays: 11234,
    earnings: '892.67',
    coverArt: 'https://images.unsplash.com/photo-1511192336575-5a79af67a629?w=600&h=600&fit=crop',
  },

  // Hip-Hop - Pay Per Stream
  {
    id: '9',
    title: 'City Lights',
    artist: 'MC Cosmos',
    genre: 'Hip-Hop',
    description: 'Urban poetry over atmospheric beats',
    ipfsHash: 'QmF6Z1uH7sT3xV0eY5dJ2aB4gP6rN9kL1iM7oC8qW3uX5',
    paymentModel: PaymentModel.PAY_PER_STREAM,
    plays: 34567,
    earnings: '345.67',
    coverArt: 'https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=600&h=600&fit=crop',
  },
  {
    id: '10',
    title: 'Rise Up',
    artist: 'The Conscious Crew',
    genre: 'Hip-Hop',
    description: 'Empowering hip-hop with positive message',
    ipfsHash: 'QmG7A2vI8tU4yW1fZ6eK3bC5hQ7sO0mM2jN8pD9rX4vY6',
    paymentModel: PaymentModel.FREEMIUM,
    plays: 28901,
    earnings: '260.19',
    coverArt: 'https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=600&h=600&fit=crop',
    freePlaysRemaining: 3,
  },

  // Classical - Patronage
  {
    id: '11',
    title: 'Moonlight Sonata Reimagined',
    artist: 'Elena Petrova',
    genre: 'Classical',
    description: 'Modern interpretation of Beethoven classic',
    ipfsHash: 'QmH8B3wJ9uV5zX2gA7fL4cD6iR8tP1nN3kO9qE0sY5wZ7',
    paymentModel: PaymentModel.PATRONAGE,
    plays: 14562,
    earnings: '1,165.00',
    coverArt: 'https://images.unsplash.com/photo-1520523839897-bd0b52f945a0?w=600&h=600&fit=crop',
  },
  {
    id: '12',
    title: 'String Quartet No. 5',
    artist: 'Resonance Ensemble',
    genre: 'Classical',
    description: 'Contemporary classical composition',
    ipfsHash: 'QmI9C4xK0vW6aY3hB8gM5dE7jS9uQ2oO4lP0rF1tZ6xA8',
    paymentModel: PaymentModel.PATRONAGE,
    plays: 7891,
    earnings: '631.28',
    coverArt: 'https://images.unsplash.com/photo-1507838153414-b4b713384a76?w=600&h=600&fit=crop',
  },

  // Electronic - Freemium
  {
    id: '13',
    title: 'Bass Cathedral',
    artist: 'Subsonic Architecture',
    genre: 'Electronic',
    description: 'Deep bass and intricate sound design',
    ipfsHash: 'QmJ0D5yL1wX7bZ4iC9hN6eF8kT0vR3pP5mQ1sG2uA7yB9',
    paymentModel: PaymentModel.FREEMIUM,
    plays: 19234,
    earnings: '162.07',
    coverArt: 'https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=600&h=600&fit=crop',
    freePlaysRemaining: 3,
  },

  // Ambient - Patronage
  {
    id: '14',
    title: 'Forest Whispers',
    artist: 'Nature\'s Symphony',
    genre: 'Ambient',
    description: 'Field recordings mixed with ambient textures',
    ipfsHash: 'QmK1E6zM2xY8cA5jD0iO7fG9lU1wS4qQ6nR2tH3vB8zC0',
    paymentModel: PaymentModel.PATRONAGE,
    plays: 31245,
    earnings: '2,499.60',
    coverArt: 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=600&h=600&fit=crop',
  },

  // Rock - Pay What You Want (tip-based)
  {
    id: '15',
    title: 'Revolution Song',
    artist: 'The Free Spirits',
    genre: 'Rock',
    description: 'Protest rock with message of unity',
    ipfsHash: 'QmL2F7aO3yZ9dB6kE1jP8gH0mV2xT5rR7oS3uI4wC9aD1',
    paymentModel: PaymentModel.PAY_WHAT_YOU_WANT,
    plays: 42156,
    earnings: '6,323.40',
    coverArt: 'https://images.unsplash.com/photo-1429962714451-bb934ecdc4ec?w=600&h=600&fit=crop',
    tipAmount: 'avg $0.15',
  },
];
