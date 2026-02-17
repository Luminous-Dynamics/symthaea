import type { UserProfile } from '$types';

const mockProfile: UserProfile = {
  agent_id: 'mock-agent',
  username: 'Offline Voyager',
  trust_score: 85,
  total_listings: 4,
  total_sales: 18,
  total_purchases: 12,
  average_rating: 4.6,
  total_reviews: 9,
  member_since: Date.now() - 1000 * 60 * 60 * 24 * 365,
  is_verified: false,
  roles: [],
};

export function getMockProfile(): UserProfile {
  return mockProfile;
}
