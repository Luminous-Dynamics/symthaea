import { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, SkipBack, SkipForward, Volume2, VolumeX, X, Heart, Share2, List, Loader2, AlertCircle } from 'lucide-react';
import { Song } from '../data/mockSongs';

interface MusicPlayerProps {
  song: Song;
  onClose: () => void;
  onNext?: () => void;
  onPrevious?: () => void;
  onPaymentRequired?: (song: Song) => Promise<boolean>;
}

// IPFS gateways in priority order
const IPFS_GATEWAYS = [
  'https://cloudflare-ipfs.com/ipfs/',
  'https://ipfs.io/ipfs/',
  'https://gateway.pinata.cloud/ipfs/',
  'https://dweb.link/ipfs/',
];

export default function MusicPlayer({ song, onClose, onNext, onPrevious, onPaymentRequired }: MusicPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(70);
  const [isMuted, setIsMuted] = useState(false);
  const [isLiked, setIsLiked] = useState(false);
  const [buffered, setBuffered] = useState(0);
  const [paymentPending, setPaymentPending] = useState(false);

  // Get audio URL - prefer direct URL, fall back to IPFS gateway
  const getAudioUrl = useCallback(() => {
    if (song.audioUrl) {
      return song.audioUrl;
    }
    if (song.ipfsHash) {
      // Try first gateway, component will retry others on error
      return `${IPFS_GATEWAYS[0]}${song.ipfsHash}`;
    }
    return null;
  }, [song.audioUrl, song.ipfsHash]);

  // Initialize audio on song change
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    setIsLoading(true);
    setLoadError(null);
    setCurrentTime(0);
    setDuration(0);
    setBuffered(0);

    const audioUrl = getAudioUrl();
    if (audioUrl) {
      audio.src = audioUrl;
      audio.load();
    } else {
      setLoadError('No audio source available');
      setIsLoading(false);
    }

    return () => {
      audio.pause();
    };
  }, [song.id, getAudioUrl]);

  // Audio event handlers
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
      setIsLoading(false);
      // Auto-play when loaded
      audio.play().catch(() => {
        // Autoplay blocked, user needs to click play
        setIsPlaying(false);
      });
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleProgress = () => {
      if (audio.buffered.length > 0) {
        const bufferedEnd = audio.buffered.end(audio.buffered.length - 1);
        setBuffered((bufferedEnd / audio.duration) * 100);
      }
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => {
      setIsPlaying(false);
      if (onNext) onNext();
    };

    const handleError = () => {
      // Try alternate IPFS gateway
      const currentSrc = audio.src;
      const currentGatewayIndex = IPFS_GATEWAYS.findIndex(g => currentSrc.startsWith(g));

      if (song.ipfsHash && currentGatewayIndex < IPFS_GATEWAYS.length - 1) {
        const nextGateway = IPFS_GATEWAYS[currentGatewayIndex + 1];
        audio.src = `${nextGateway}${song.ipfsHash}`;
        audio.load();
      } else {
        setLoadError('Failed to load audio. The file may not be available.');
        setIsLoading(false);
      }
    };

    const handleCanPlay = () => {
      setIsLoading(false);
    };

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('progress', handleProgress);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('error', handleError);
    audio.addEventListener('canplay', handleCanPlay);

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('progress', handleProgress);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('error', handleError);
      audio.removeEventListener('canplay', handleCanPlay);
    };
  }, [song.ipfsHash, onNext]);

  // Handle play/pause with payment check
  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      // Check if payment is required
      if (onPaymentRequired) {
        setPaymentPending(true);
        try {
          const approved = await onPaymentRequired(song);
          if (!approved) {
            setPaymentPending(false);
            return;
          }
        } catch (error) {
          console.error('Payment failed:', error);
          setPaymentPending(false);
          return;
        }
        setPaymentPending(false);
      }

      try {
        await audio.play();
      } catch (error) {
        console.error('Playback failed:', error);
      }
    }
  };

  // Volume control
  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.volume = isMuted ? 0 : volume / 100;
    }
  }, [volume, isMuted]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;
    const newTime = parseFloat(e.target.value);
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseInt(e.target.value);
    setVolume(newVolume);
    if (newVolume > 0) setIsMuted(false);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: song.title,
          text: `Check out "${song.title}" by ${song.artist} on Mycelix Music!`,
          url: window.location.href,
        });
      } catch (err) {
        console.log('Share cancelled');
      }
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(window.location.href);
      alert('Link copied to clipboard!');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      {/* Hidden Audio Element */}
      <audio ref={audioRef} preload="metadata" />

      <div className="bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 rounded-2xl max-w-2xl w-full shadow-2xl border border-white/10 overflow-hidden relative">

        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20 transition z-10"
        >
          <X className="w-6 h-6 text-white" />
        </button>

        {/* Album Art */}
        <div className="relative w-full aspect-square">
          {/* Real cover art */}
          <img
            src={song.coverArt}
            alt={`${song.title} by ${song.artist}`}
            className="w-full h-full object-cover"
          />

          {/* Loading Overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-12 h-12 text-purple-400 animate-spin mx-auto mb-2" />
                <p className="text-white text-sm">Loading audio...</p>
              </div>
            </div>
          )}

          {/* Error Overlay */}
          {loadError && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
              <div className="text-center p-4">
                <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-2" />
                <p className="text-white text-sm">{loadError}</p>
                <p className="text-gray-400 text-xs mt-2">This is demo mode - real IPFS audio will work in production</p>
              </div>
            </div>
          )}

          {/* Waveform/Progress Overlay */}
          <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-black/60 to-transparent flex items-end justify-center space-x-0.5 p-4">
            {Array.from({ length: 60 }).map((_, i) => {
              const height = 20 + Math.sin(i * 0.5) * 30 + Math.random() * 20;
              const progress = duration > 0 ? (currentTime / duration) * 60 : 0;
              const isActive = i < progress;
              const isBuffered = i < (buffered / 100) * 60;
              return (
                <div
                  key={i}
                  className={`w-1 rounded-full transition-colors ${
                    isActive ? 'bg-purple-400' : isBuffered ? 'bg-white/40' : 'bg-white/20'
                  }`}
                  style={{ height: `${height}%` }}
                />
              );
            })}
          </div>
        </div>

        {/* Player Controls */}
        <div className="p-6 space-y-4">

          {/* Song Info */}
          <div className="text-center">
            <h2 className="text-2xl font-bold text-white mb-1">{song.title}</h2>
            <p className="text-gray-400">{song.artist}</p>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <input
              type="range"
              min="0"
              max={duration}
              value={currentTime}
              onChange={handleSeek}
              className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer
                         [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4
                         [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-purple-500
                         [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer
                         [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4
                         [&::-moz-range-thumb]:bg-purple-500 [&::-moz-range-thumb]:rounded-full
                         [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-0"
            />
            <div className="flex justify-between text-sm text-gray-400">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>

          {/* Playback Controls */}
          <div className="flex items-center justify-center space-x-6">
            <button
              onClick={onPrevious}
              className="p-3 rounded-full hover:bg-white/10 transition disabled:opacity-50"
              disabled={!onPrevious}
            >
              <SkipBack className="w-6 h-6 text-white" />
            </button>

            <button
              onClick={togglePlay}
              disabled={isLoading || !!loadError || paymentPending}
              className="p-5 rounded-full bg-purple-600 hover:bg-purple-700 transition shadow-lg shadow-purple-500/50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {paymentPending ? (
                <Loader2 className="w-8 h-8 text-white animate-spin" />
              ) : isPlaying ? (
                <Pause className="w-8 h-8 text-white" />
              ) : (
                <Play className="w-8 h-8 text-white ml-1" />
              )}
            </button>

            <button
              onClick={onNext}
              className="p-3 rounded-full hover:bg-white/10 transition disabled:opacity-50"
              disabled={!onNext}
            >
              <SkipForward className="w-6 h-6 text-white" />
            </button>
          </div>

          {/* Secondary Controls */}
          <div className="flex items-center justify-between">

            {/* Volume */}
            <div className="flex items-center space-x-2 flex-1">
              <button
                onClick={toggleMute}
                className="p-2 rounded-full hover:bg-white/10 transition"
              >
                {isMuted || volume === 0 ? (
                  <VolumeX className="w-5 h-5 text-gray-400" />
                ) : (
                  <Volume2 className="w-5 h-5 text-gray-400" />
                )}
              </button>
              <input
                type="range"
                min="0"
                max="100"
                value={isMuted ? 0 : volume}
                onChange={handleVolumeChange}
                className="w-24 h-1 bg-white/20 rounded-lg appearance-none cursor-pointer
                           [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                           [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-purple-500
                           [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer
                           [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:h-3
                           [&::-moz-range-thumb]:bg-purple-500 [&::-moz-range-thumb]:rounded-full
                           [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-0"
              />
            </div>

            {/* Actions */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setIsLiked(!isLiked)}
                className={`p-2 rounded-full hover:bg-white/10 transition ${
                  isLiked ? 'text-red-500' : 'text-gray-400'
                }`}
              >
                <Heart className={`w-5 h-5 ${isLiked ? 'fill-current' : ''}`} />
              </button>
              <button
                onClick={handleShare}
                className="p-2 rounded-full hover:bg-white/10 transition text-gray-400"
              >
                <Share2 className="w-5 h-5" />
              </button>
              <button
                className="p-2 rounded-full hover:bg-white/10 transition text-gray-400"
              >
                <List className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Song Stats */}
          <div className="flex items-center justify-between pt-4 border-t border-white/10">
            <div className="text-sm">
              <span className="text-gray-400">{song.plays.toLocaleString()} plays</span>
              {' • '}
              <span className="text-gray-400">{song.genre}</span>
            </div>
            <div className="text-sm">
              <span className="text-green-400 font-semibold">${song.earnings} earned</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
