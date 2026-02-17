export type SongAuthParams = {
  id: string;
  artistAddress: string;
  ipfsHash: string;
  paymentModel: string;
  nonce?: string;
  timestamp?: string | number;
};

export type PlayAuthParams = {
  songId: string;
  listener: string;
  amount: string | number;
  paymentType: string;
  nonce?: string;
  timestamp?: string | number;
};

export type ClaimAuthParams = {
  songId: string;
  artistAddress: string;
  ipfsHash: string;
  title: string;
  nonce?: string;
  timestamp?: string | number;
};

export function messageForSong(params: SongAuthParams): string[] {
  return [
    'mycelix-song',
    params.id || '',
    params.artistAddress || '',
    params.ipfsHash || '',
    params.paymentModel || '',
    params.nonce || '',
    String(params.timestamp ?? ''),
  ];
}

export function messageForPlay(params: PlayAuthParams): string[] {
  return [
    'mycelix-play',
    params.songId || '',
    params.listener || '',
    String(params.amount ?? ''),
    params.paymentType || '',
    params.nonce || '',
    String(params.timestamp ?? ''),
  ];
}

export function messageForClaim(params: ClaimAuthParams): string[] {
  return [
    'mycelix-claim',
    params.songId || '',
    params.artistAddress || '',
    params.ipfsHash || '',
    params.title || '',
    params.nonce || '',
    String(params.timestamp ?? ''),
  ];
}
